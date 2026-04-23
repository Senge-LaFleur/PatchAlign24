from __future__ import print_function, division
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import os
import skimage
from skimage import io
import warnings
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import lr_scheduler
import time
import copy
import random
import cv2
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from torch.utils.tensorboard import SummaryWriter
from Models.got_losses import Network, Confusion_Loss, Supervised_Contrastive_Loss
from Masked_GOT_NewSinkhorn import cost_matrix_batch_torch, GW_distance_uniform, IPOT_distance_torch_batch_uniform
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings('ignore')
print('Imports complete.')

device_global = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device_global)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')
bert_model.to(device_global)

labels_top = ['drug induced pigmentary changes', 'photodermatoses',
       'dermatofibroma', 'psoriasis', 'kaposi sarcoma',
       'neutrophilic dermatoses', 'granuloma annulare',
       'nematode infection', 'allergic contact dermatitis',
       'necrobiosis lipoidica', 'hidradenitis', 'melanoma',
       'acne vulgaris', 'sarcoidosis', 'xeroderma pigmentosum',
       'actinic keratosis', 'scleroderma', 'syringoma', 'folliculitis',
       'pityriasis lichenoides chronica', 'porphyria',
       'dyshidrotic eczema', 'seborrheic dermatitis', 'prurigo nodularis',
       'acne', 'neurofibromatosis', 'eczema', 'pediculosis lids',
       'basal cell carcinoma', 'pityriasis rubra pilaris',
       'pityriasis rosea', 'livedo reticularis',
       'stevens johnson syndrome', 'erythema multiforme',
       'acrodermatitis enteropathica', 'epidermolysis bullosa',
       'dermatomyositis', 'urticaria', 'basal cell carcinoma morpheiform',
       'vitiligo', 'erythema nodosum', 'lupus erythematosus',
       'lichen planus', 'sun damaged skin', 'drug eruption', 'scabies',
       'cheilitis', 'urticaria pigmentosa', 'behcets disease',
       'nevocytic nevus', 'mycosis fungoides',
       'superficial spreading melanoma ssm', 'porokeratosis of mibelli',
       'juvenile xanthogranuloma', 'milia', 'granuloma pyogenic',
       'papilomatosis confluentes and reticulate',
       'neurotic excoriations', 'epidermal nevus', 'naevus comedonicus',
       'erythema annulare centrifigum', 'pilar cyst',
       'pustular psoriasis', 'ichthyosis vulgaris', 'lyme disease',
       'striae', 'rhinophyma', 'calcinosis cutis', 'stasis edema',
       'neurodermatitis', 'congenital nevus', 'squamous cell carcinoma',
       'mucinosis', 'keratosis pilaris', 'keloid', 'tuberous sclerosis',
       'acquired autoimmune bullous diseaseherpes gestationis',
       'fixed eruptions', 'lentigo maligna', 'lichen simplex',
       'dariers disease', 'lymphangioma', 'pilomatricoma',
       'lupus subacute', 'perioral dermatitis',
       'disseminated actinic porokeratosis', 'erythema elevatum diutinum',
       'halo nevus', 'aplasia cutis', 'incontinentia pigmenti',
       'tick bite', 'fordyce spots', 'telangiectases',
       'solid cystic basal cell carcinoma', 'paronychia', 'becker nevus',
       'pyogenic granuloma', 'langerhans cell histiocytosis',
       'port wine stain', 'malignant melanoma', 'factitial dermatitis',
       'xanthomas', 'nevus sebaceous of jadassohn',
       'hailey hailey disease', 'scleromyxedema', 'porokeratosis actinic',
       'rosacea', 'acanthosis nigricans', 'myiasis',
       'seborrheic keratosis', 'mucous cyst', 'lichen amyloidosis',
       'ehlers danlos syndrome', 'tungiasis', 'eudermic']

print(f'Labels loaded: {len(labels_top)} conditions')

def calculate_probabilities(string_list):
    n = len(string_list)
    counts = {}
    probabilities = []
    for string in string_list:
        counts[string] = counts.get(string, 0) + 1
    summation = 0
    for string in string_list:
        probability = counts[string] / n
        probabilities.append(probability)
        summation += probability
    probabilities = [i / summation for i in probabilities]
    return probabilities


def got_loss(p, q, Mask, lamb):
    cos_distance = cost_matrix_batch_torch(p.transpose(2, 1), q.transpose(2, 1)).transpose(1, 2)
    beta = 0.1
    min_score = cos_distance.min()
    max_score = cos_distance.max()
    threshold = min_score + beta * (max_score - min_score)
    cos_dist = torch.nn.functional.relu(cos_distance - threshold)
    wd, T = IPOT_distance_torch_batch_uniform(cos_dist, Mask, p.size(0), p.size(1), q.size(1), 30)
    gwd = GW_distance_uniform(p.transpose(2, 1), q.transpose(2, 1), Mask)
    twd = lamb * torch.mean(gwd) + (1 - lamb) * torch.mean(wd)
    return twd


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


print('Helper functions defined.')

# ============================================================
# Modality Invariance Loss (L_mi) Components
# ============================================================

class ProjectionHead(nn.Module):
    """MLP projection head for L_mi.
    Projects mean-pooled patch features into a normalised embedding
    space for cosine-similarity alignment between views.
    """
    def __init__(self, input_dim=512, proj_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim)
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


# Pseudo-dermoscopic augmentation: simulates dermoscopic imaging
# characteristics from a clinical image to create a cross-modal view.
pseudo_derm_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0)),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def modality_invariance_loss(feat_orig, feat_aug, proj_head):
    """Modality Invariance Loss (L_mi).

    Measures cosine dissimilarity between projected features of the
    original image and its pseudo-dermoscopic augmented view.
    Minimising this encourages modality-invariant representations.

    Args:
        feat_orig (Tensor): (B, D) mean-pooled patch features, original view.
        feat_aug  (Tensor): (B, D) mean-pooled patch features, augmented view.
        proj_head (nn.Module): ProjectionHead module.

    Returns:
        Scalar loss.
    """
    z1 = proj_head(feat_orig)
    z2 = proj_head(feat_aug)
    cos_sim = nn.functional.cosine_similarity(z1, z2, dim=-1)
    return (1.0 - cos_sim).mean()


print('L_mi components defined: ProjectionHead | pseudo_derm_transform | modality_invariance_loss')

def train_model_lmi(label, dataloaders, device, dataset_sizes, model, proj_head,
                    criterion, optimizer, scheduler,
                    num_epochs=2, alpha=1.0, beta=0.8, lambda_mi=0.1):
    """Training loop with multi-objective loss including L_mi.

    Total loss = L_cls + 0.5*L_conf + L_ce + L_GOT + lambda_mi * L_mi
    L_mi is computed only during the training phase (not validation).
    """
    print(f'Hyper-params | alpha={alpha}  beta={beta}  lambda_mi={lambda_mi}')
    since = time.time()
    training_results = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_proj_wts  = copy.deepcopy(proj_head.state_dict())
    best_acc = 0.0
    train_step = 0
    leading_epoch = 0

    text_embeddings = np.load('./text_embeddings_3_large_consecutive_averaged.npy')
    text_embeddings = np.array(text_embeddings, dtype=np.double)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                proj_head.train()
            else:
                model.eval()
                proj_head.eval()

            running_loss = 0.0
            running_lmi  = 0.0
            running_corrects = 0.0
            running_balanced_acc_sum = 0.0
            print(phase)

            loop = tqdm(dataloaders[phase], leave=True,
                        desc=f' {phase}-ing Epoch {epoch + 1}/{num_epochs}')
            for n_iter, batch in enumerate(loop):
                bs = len(batch['mid'])
                textual_embeddings = torch.cat(
                    tuple([torch.tensor(text_embeddings).unsqueeze(0)] * bs)
                ).to(device).double()

                inputs     = batch['image'].to(device)
                inputs_aug = batch['image_aug'].to(device)
                label_c    = torch.from_numpy(np.asarray(batch[label])).to(device)
                label_t    = torch.from_numpy(np.asarray(batch['fitzpatrick']) - 1).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    inputs     = inputs.float()
                    inputs_aug = inputs_aug.float()

                    # --- Original forward pass ---
                    output = model(inputs)
                    l_got  = got_loss(output[-1], textual_embeddings, output[3], lamb=0.9)

                    _, preds = torch.max(output[0], 1)
                    loss0 = criterion[0](output[0], label_c)  # CE (condition)
                    loss1 = criterion[1](output[1], label_t)  # Confusion (skin type)
                    loss2 = criterion[2](output[2], label_t)  # CE (skin type)
                    loss3 = torch.tensor(0)

                    # --- Modality Invariance Loss (train phase only) ---
                    if phase == 'train':
                        output_aug = model(inputs_aug)
                        feat_orig = output[-1].mean(dim=1).float()
                        feat_aug  = output_aug[-1].mean(dim=1).float()
                        l_mi = modality_invariance_loss(feat_orig, feat_aug, proj_head)
                    else:
                        l_mi = torch.tensor(0.0, device=device)

                    # --- Multi-objective loss ---
                    loss = loss0 + loss1 * 0.5 + loss2 + 1 * l_got + lambda_mi * l_mi

                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(
                            list(model.parameters()) + list(proj_head.parameters()),
                            max_norm=1.0
                        )
                        optimizer.step()

                # --- TensorBoard ---
                if phase == 'train':
                    writer.add_scalar('Loss/' + phase,              loss.item(),  train_step)
                    writer.add_scalar('Loss/' + phase + '_loss0',    loss0.item(), train_step)
                    writer.add_scalar('Loss/' + phase + '_loss1_conf', loss1.item(), train_step)
                    writer.add_scalar('Loss/' + phase + '_loss2',    loss2.item(), train_step)
                    writer.add_scalar('Loss/' + phase + '_l_mi',    l_mi.item(),  train_step)
                    writer.add_scalar('Loss/' + phase + '_l_got',   l_got.item(), train_step)
                    writer.add_scalar('Accuracy/' + phase,
                                      (torch.sum(preds == label_c.data)).item() / inputs.size(0),
                                      train_step)
                    writer.add_scalar('Balanced-Accuracy/' + phase,
                                      balanced_accuracy_score(label_c.data.cpu(), preds.cpu()),
                                      train_step)
                    train_step += 1

                running_loss += loss.item() * inputs.size(0)
                running_lmi  += l_mi.item() * inputs.size(0)
                running_corrects += torch.sum(preds == label_c.data)
                running_balanced_acc_sum += (
                    balanced_accuracy_score(label_c.data.cpu(), preds.cpu()) * inputs.size(0)
                )

            epoch_loss         = running_loss / dataset_sizes[phase]
            epoch_lmi          = running_lmi  / dataset_sizes[phase]
            epoch_acc          = running_corrects / dataset_sizes[phase]
            epoch_balanced_acc = running_balanced_acc_sum / dataset_sizes[phase]

            print(f'Accuracy: {running_corrects}/{dataset_sizes[phase]}')
            print(f'{phase} Loss: {epoch_loss:.4f} | L_mi: {epoch_lmi:.4f} | '
                  f'Acc: {epoch_acc:.4f} | Balanced-Acc: {epoch_balanced_acc:.4f}')

            writer.add_scalar('lr/' + phase, scheduler.get_last_lr()[0], epoch)
            if phase == 'val':
                writer.add_scalar('Loss/' + phase,            epoch_loss,         epoch)
                writer.add_scalar('Accuracy/' + phase,        epoch_acc,          epoch)
                writer.add_scalar('Balanced-Accuracy/' + phase, epoch_balanced_acc, epoch)

            training_results.append(
                [phase, epoch, epoch_loss, epoch_lmi, epoch_acc.item(), epoch_balanced_acc]
            )

            if epoch > 0:
                if phase == 'val' and epoch_acc > best_acc:
                    print(f'New leading accuracy: {epoch_acc}')
                    best_acc       = epoch_acc
                    leading_epoch  = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_proj_wts  = copy.deepcopy(proj_head.state_dict())
            elif phase == 'val':
                best_acc = epoch_acc

        scheduler.step()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}  (epoch {leading_epoch})')
    model.load_state_dict(best_model_wts)
    proj_head.load_state_dict(best_proj_wts)
    training_results = pd.DataFrame(training_results)
    training_results.columns = ['phase', 'epoch', 'loss', 'l_mi', 'accuracy', 'balanced-accuracy']
    return model, proj_head, training_results


print('train_model_lmi defined.')

class SkinDatasetLmi():
    """SkinDataset extended to return both a standard and a pseudo-derm augmented view.

    The 'image_aug' view is produced by pseudo_derm_transform from the raw numpy image.
    During validation, pseudo_derm_transform is None so image_aug == image.
    """
    def __init__(self, dataset_name, csv_file, root_dir,
                 transform=None, pseudo_derm_transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.pseudo_derm_transform = pseudo_derm_transform
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.dataset_name == 'ddi':
            img_name = os.path.join(self.root_dir,
                                    str(self.df.loc[self.df.index[idx], 'hasher']))
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_name = os.path.join(self.root_dir,
                                    str(self.df.loc[self.df.index[idx], 'hasher'])) + '.jpg'
            image = io.imread(img_name)

        if len(image.shape) < 3:
            image = skimage.color.gray2rgb(image)

        image_raw = image  # keep the raw numpy array for augmentation

        hasher      = self.df.loc[self.df.index[idx], 'hasher']
        high        = self.df.loc[self.df.index[idx], 'high']
        low         = self.df.loc[self.df.index[idx], 'low']
        fitzpatrick = self.df.loc[self.df.index[idx], 'fitzpatrick']

        if self.dataset_name == 'fitzpatrick':
            mid       = self.df.loc[self.df.index[idx], 'mid']
            partition = self.df.loc[self.df.index[idx], 'label']
        else:
            mid       = 0
            partition = self.df.loc[self.df.index[idx], 'disease']

        if self.transform:
            image = self.transform(image_raw)

        if self.pseudo_derm_transform:
            image_aug = self.pseudo_derm_transform(image_raw)
        else:
            image_aug = image  # fallback for val set

        sample = {
            'image':     image,
            'image_aug': image_aug,
            'high':      high,
            'mid':       mid,
            'low':       low,
            'hasher':    hasher,
            'fitzpatrick': fitzpatrick,
            'partition': partition
        }
        return sample


print('SkinDatasetLmi defined.')

def custom_load_lmi(
        batch_size=32,
        num_workers=0,
        train_dir='',
        val_dir='',
        label='low',
        dataset_name='fitzpatrick',
        image_dir='C:\\Users\\asose\\OneDrive\\Desktop\\Senge Research\\datasets\\fitzpatrick17k\\data\\finalfitz17k\\'
):
    if dataset_name == 'ddi':
        image_dir = 'C:\\Users\\asose\\OneDrive\\Desktop\\Senge Research\\datasets\\ddidiversedermatologyimages\\'

    val   = pd.read_csv(val_dir)
    train = pd.read_csv(train_dir)

    class_sample_count = np.array(train[label].value_counts().sort_index())
    weight             = 1. / class_sample_count
    samples_weight     = np.array([weight[t] for t in train[label]])
    samples_weight     = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(
        samples_weight.type('torch.DoubleTensor'),
        len(samples_weight),
        replacement=True
    )
    dataset_sizes = {'train': train.shape[0], 'val': val.shape[0]}

    standard_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transformed_train = SkinDatasetLmi(
        dataset_name=dataset_name,
        csv_file=train_dir,
        root_dir=image_dir,
        transform=standard_transform,
        pseudo_derm_transform=pseudo_derm_transform
    )
    transformed_test = SkinDatasetLmi(
        dataset_name=dataset_name,
        csv_file=val_dir,
        root_dir=image_dir,
        transform=val_transform,
        pseudo_derm_transform=None  # no augmented view needed at validation
    )

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            transformed_train,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers
        ),
        'val': torch.utils.data.DataLoader(
            transformed_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    }
    return dataloaders, dataset_sizes


print('custom_load_lmi defined.')

# ============================================================
#  Configuration — edit before running
# ============================================================
n_epochs     = 20           # number of training epochs
dev_mode     = 'full'       # 'dev' (1000-sample quick test) or 'full'
dataset_name = 'fitzpatrick'
model_name   = 'PATCHALIGN_FITZ_INDOMAIN_LMI'
lambda_mi    = 0.1          # weight for the modality invariance loss

# In-domain: domain is always random_holdout (index 0)
domain_index = 0  # 0=random_holdout | 1=a12 | 2=a34 | 3=a56

# Reproducibility (matches base InDomain script)
torch.manual_seed(200709)
random.seed(200709)
np.random.seed(200709)

print(f'Config | dataset={dataset_name} | model={model_name} | epochs={n_epochs} | lambda_mi={lambda_mi}')

print(f'CUDA available: {torch.cuda.is_available()}')
device = device_global

# ------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------
if dev_mode == 'dev':
    df = pd.read_csv('fitzpatrick17k_known_code.csv').sample(1000)
else:
    df = pd.read_csv('fitzpatrick17k_known_code.csv')

# Fix column name if needed
if 'fitzpatrick_scale' in df.columns and 'fitzpatrick' not in df.columns:
    df = df.rename(columns={'fitzpatrick_scale': 'fitzpatrick'})

domain = ['random_holdout', 'a12', 'a34', 'a56'][domain_index]
print(f'Domain: {domain}')

for holdout_set in [domain]:
    if holdout_set == 'random_holdout':
        train, test, _, _ = train_test_split(
            df, df['low'],
            test_size=0.2,
            random_state=205504,
            stratify=df['low']
        )
    elif holdout_set == 'a12':
        train = df[(df.fitzpatrick == 1) | (df.fitzpatrick == 2)]
        test  = df[(df.fitzpatrick != 1) & (df.fitzpatrick != 2)]
        combo = set(train.label.unique()) & set(test.label.unique())
        train = train[train.label.isin(combo)].reset_index()
        test  = test[test.label.isin(combo)].reset_index()
        train['low'] = train['label'].astype('category').cat.codes
        test['low']  = test['label'].astype('category').cat.codes
    elif holdout_set == 'a34':
        train = df[(df.fitzpatrick == 3) | (df.fitzpatrick == 4)]
        test  = df[(df.fitzpatrick != 3) & (df.fitzpatrick != 4)]
        combo = set(train.label.unique()) & set(test.label.unique())
        train = train[train.label.isin(combo)].reset_index()
        test  = test[test.label.isin(combo)].reset_index()
        train['low'] = train['label'].astype('category').cat.codes
        test['low']  = test['label'].astype('category').cat.codes
    elif holdout_set == 'a56':
        train = df[(df.fitzpatrick == 5) | (df.fitzpatrick == 6)]
        test  = df[(df.fitzpatrick != 5) & (df.fitzpatrick != 6)]
        combo = set(train.label.unique()) & set(test.label.unique())
        train = train[train.label.isin(combo)].reset_index()
        test  = test[test.label.isin(combo)].reset_index()
        train['low'] = train['label'].astype('category').cat.codes
        test['low']  = test['label'].astype('category').cat.codes

    level = 'high'
    train_path = f'temp_train_{model_name}.csv'
    test_path  = f'temp_test_{model_name}.csv'
    train.to_csv(train_path, index=False)
    test.to_csv(test_path,  index=False)

    for indexer, label in enumerate([level]):
        writer = SummaryWriter(comment=f'logs_{model_name}_{n_epochs}_{label}_{holdout_set}.pth')
        print(f'\nLabel: {label}')

        weights      = np.array(max(train[label].value_counts()) / train[label].value_counts().sort_index())
        label_codes  = sorted(list(train[label].unique()))
        dataloaders, dataset_sizes = custom_load_lmi(
            32, 0, train_path, test_path,
            label=label, dataset_name=dataset_name
        )
        print(f'Dataset sizes: {dataset_sizes}')

        # Build base model
        model_ft = Network('sparse', [len(label_codes), 6], pretrained=True)
        total_params = sum(p.numel() for p in model_ft.feature_extractor.parameters())
        print(f'{total_params} total parameters')
        for i, p in enumerate(model_ft.feature_extractor.parameters()):
            p.requires_grad = (i >= 50)
        trainable = sum(p.numel() for p in model_ft.feature_extractor.parameters() if p.requires_grad)
        print(f'{trainable} trainable parameters')
        model_ft = model_ft.to(device)
        if torch.cuda.is_available():
            model_ft = nn.DataParallel(model_ft)

        # ---- Auto-detect feature dimension for ProjectionHead ----
        model_ft.eval()
        with torch.no_grad():
            _dummy = torch.randn(2, 3, 224, 224).float().to(device)
            _out   = model_ft(_dummy)
            feat_dim = _out[-1].mean(dim=1).shape[-1]
        print(f'Auto-detected feature dim: {feat_dim}')

        proj_head = ProjectionHead(input_dim=feat_dim, proj_dim=128).to(device)

        # Loss criteria and optimiser
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = [
            nn.CrossEntropyLoss(),
            Confusion_Loss(),
            nn.CrossEntropyLoss(),
            Supervised_Contrastive_Loss(0.1, device)
        ]
        # Optimise both backbone and projection head jointly
        optimizer_ft = optim.Adam(
            list(model_ft.parameters()) + list(proj_head.parameters()),
            lr=0.0001
        )
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.8)

        print(f'\nTraining {label} with L_mi ...')
        model_ft, proj_head, training_results = train_model_lmi(
            label, dataloaders, device, dataset_sizes,
            model_ft, proj_head, criterion,
            optimizer_ft, exp_lr_scheduler,
            num_epochs=n_epochs, lambda_mi=lambda_mi
        )
        print('Training Complete')

        torch.save(model_ft.state_dict(), f'model_path_{model_name}_{n_epochs}_{label}_{holdout_set}.pth')
        torch.save(model_ft,              f'model_path_{model_name}_{n_epochs}_{label}_{holdout_set}.pt')
        torch.save(proj_head.state_dict(),f'proj_head_{model_name}_{n_epochs}_{label}_{holdout_set}.pth')
        training_results.to_csv(f'training_{model_name}_{n_epochs}_{label}_{holdout_set}.csv')
        print('Model and results saved.')

        # ------------------------------------------------------------------
        # Evaluation on validation set
        # ------------------------------------------------------------------
        model = model_ft.eval()
        prediction_list   = []
        fitzpatrick_list   = []
        hasher_list        = []
        labels_list        = []
        p_list             = []

        with torch.no_grad():
            running_corrects        = 0
            running_balanced_acc_sum = 0
            total = 0
            for i, batch in enumerate(dataloaders['val']):
                inputs      = batch['image'].to(device)
                classes     = batch[label].to(device)
                fitzpatrick = batch['fitzpatrick']
                hasher      = batch['hasher']
                outputs     = model(inputs.float())
                probability = torch.nn.functional.softmax(outputs[0], dim=1)
                ppp, preds  = torch.topk(probability, 1)
                running_balanced_acc_sum += (
                    balanced_accuracy_score(classes.data.cpu(), preds.reshape(-1).cpu())
                    * inputs.shape[0]
                )
                running_corrects += torch.sum(preds.reshape(-1) == classes.data)
                p_list.append(ppp.cpu().tolist())
                prediction_list.append(preds.cpu().tolist())
                labels_list.append(classes.tolist())
                fitzpatrick_list.append(fitzpatrick.tolist())
                hasher_list.append(hasher)
                total += inputs.shape[0]

            acc          = float(running_corrects) / float(dataset_sizes['val'])
            balanced_acc = float(running_balanced_acc_sum) / float(dataset_sizes['val'])

        df_x = pd.DataFrame({
            'hasher':                flatten(hasher_list),
            'label':                 flatten(labels_list),
            'fitzpatrick':           flatten(fitzpatrick_list),
            'prediction_probability': flatten(p_list),
            'prediction':            flatten(prediction_list)
        })
        df_x.to_csv(f'results_{model_name}_{n_epochs}_{label}_{holdout_set}.csv', index=False)
        print(f'\n Accuracy: {acc:.4f}   Balanced Accuracy: {balanced_acc:.4f} \n')

print('Done.')

