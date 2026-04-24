Imports complete.
Using device: cuda
tokenizer_config.json: 100%|██████████████████| 48.0/48.0 [00:00<00:00, 239kB/s]
config.json: 100%|█████████████████████████████| 570/570 [00:00<00:00, 1.66MB/s]
vocab.txt: 232kB [00:00, 10.2MB/s]
tokenizer.json: 466kB [00:00, 8.96MB/s]
model.safetensors: 100%|██████████████████████| 440M/440M [00:02<00:00, 200MB/s]
Labels loaded: 115 conditions
Helper functions defined.
L_mi components defined: ProjectionHead | pseudo_derm_transform | modality_invariance_loss
train_model_lmi defined.
SkinDatasetLmi defined.
custom_load_lmi defined.
Config | dataset=fitzpatrick | model=PATCHALIGN_FITZ_INDOMAIN_LMI | epochs=20 | lambda_mi=0.1
CUDA available: True
Domain: random_holdout

Label: high
Dataset sizes: {'train': 12809, 'val': 3203}
config.json: 100%|█████████████████████████████| 502/502 [00:00<00:00, 1.52MB/s]
model.safetensors: 100%|██████████████████████| 346M/346M [00:01<00:00, 209MB/s]
86389248 total parameters
64384512 trainable parameters
Auto-detected feature dim: 768

Training high with L_mi ...
Hyper-params | alpha=1.0  beta=0.8  lambda_mi=0.1
Epoch 0/19
----------
train
 train-ing Epoch 1/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 8942.0/12809
train Loss: 3.4255 | L_mi: 0.0242 | Acc: 0.6981 | Balanced-Acc: 0.6993
val
 val-ing Epoch 1/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2264.0/3203
val Loss: 3.3691 | L_mi: 0.0000 | Acc: 0.7068 | Balanced-Acc: 0.7216
Epoch 1/19
----------
train
 train-ing Epoch 2/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 11262.0/12809
train Loss: 3.0108 | L_mi: 0.0016 | Acc: 0.8792 | Balanced-Acc: 0.8795
val
 val-ing Epoch 2/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2551.0/3203
val Loss: 3.2679 | L_mi: 0.0000 | Acc: 0.7964 | Balanced-Acc: 0.7429
New leading accuracy: 0.796440839767456
Epoch 2/19
----------
train
 train-ing Epoch 3/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12100.0/12809
train Loss: 2.8403 | L_mi: 0.0008 | Acc: 0.9446 | Balanced-Acc: 0.9450
val
 val-ing Epoch 3/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2639.0/3203
val Loss: 3.1878 | L_mi: 0.0000 | Acc: 0.8239 | Balanced-Acc: 0.7596
New leading accuracy: 0.8239150643348694
Epoch 3/19
----------
train
 train-ing Epoch 4/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12396.0/12809
train Loss: 2.7623 | L_mi: 0.0004 | Acc: 0.9678 | Balanced-Acc: 0.9679
val
 val-ing Epoch 4/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2742.0/3203
val Loss: 3.3479 | L_mi: 0.0000 | Acc: 0.8561 | Balanced-Acc: 0.7836
New leading accuracy: 0.8560724258422852
Epoch 4/19
----------
train
 train-ing Epoch 5/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12529.0/12809
train Loss: 2.7159 | L_mi: 0.0003 | Acc: 0.9781 | Balanced-Acc: 0.9787
val
 val-ing Epoch 5/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2795.0/3203
val Loss: 3.2122 | L_mi: 0.0000 | Acc: 0.8726 | Balanced-Acc: 0.7645
New leading accuracy: 0.8726193904876709
Epoch 5/19
----------
train
 train-ing Epoch 6/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12629.0/12809
train Loss: 2.6838 | L_mi: 0.0002 | Acc: 0.9859 | Balanced-Acc: 0.9853
val
 val-ing Epoch 6/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2777.0/3203
val Loss: 3.3800 | L_mi: 0.0000 | Acc: 0.8670 | Balanced-Acc: 0.7528
Epoch 6/19
----------
train
 train-ing Epoch 7/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12652.0/12809
train Loss: 2.6687 | L_mi: 0.0001 | Acc: 0.9877 | Balanced-Acc: 0.9875
val
 val-ing Epoch 7/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2806.0/3203
val Loss: 3.2224 | L_mi: 0.0000 | Acc: 0.8761 | Balanced-Acc: 0.7744
New leading accuracy: 0.8760536909103394
Epoch 7/19
----------
train
 train-ing Epoch 8/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12694.0/12809
train Loss: 2.6492 | L_mi: 0.0001 | Acc: 0.9910 | Balanced-Acc: 0.9914
val
 val-ing Epoch 8/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2821.0/3203
val Loss: 3.4105 | L_mi: 0.0000 | Acc: 0.8807 | Balanced-Acc: 0.7446
New leading accuracy: 0.880736768245697
Epoch 8/19
----------
train
 train-ing Epoch 9/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12716.0/12809
train Loss: 2.6415 | L_mi: 0.0001 | Acc: 0.9927 | Balanced-Acc: 0.9927
val
 val-ing Epoch 9/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2843.0/3203
val Loss: 3.2525 | L_mi: 0.0000 | Acc: 0.8876 | Balanced-Acc: 0.7676
New leading accuracy: 0.8876053690910339
Epoch 9/19
----------
train
 train-ing Epoch 10/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12730.0/12809
train Loss: 2.6325 | L_mi: 0.0000 | Acc: 0.9938 | Balanced-Acc: 0.9938
val
 val-ing Epoch 10/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2800.0/3203
val Loss: 3.4183 | L_mi: 0.0000 | Acc: 0.8742 | Balanced-Acc: 0.7600
Epoch 10/19
----------
train
 train-ing Epoch 11/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12770.0/12809
train Loss: 2.6178 | L_mi: 0.0000 | Acc: 0.9970 | Balanced-Acc: 0.9969
val
 val-ing Epoch 11/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2846.0/3203
val Loss: 3.4350 | L_mi: 0.0000 | Acc: 0.8885 | Balanced-Acc: 0.7452
New leading accuracy: 0.8885419368743896
Epoch 11/19
----------
train
 train-ing Epoch 12/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12757.0/12809
train Loss: 2.6187 | L_mi: 0.0000 | Acc: 0.9959 | Balanced-Acc: 0.9962
val
 val-ing Epoch 12/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2848.0/3203
val Loss: 3.4290 | L_mi: 0.0000 | Acc: 0.8892 | Balanced-Acc: 0.7562
New leading accuracy: 0.8891663551330566
Epoch 12/19
----------
train
 train-ing Epoch 13/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12761.0/12809
train Loss: 2.6148 | L_mi: 0.0000 | Acc: 0.9963 | Balanced-Acc: 0.9964
val
 val-ing Epoch 13/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2838.0/3203
val Loss: 3.3676 | L_mi: 0.0000 | Acc: 0.8860 | Balanced-Acc: 0.7652
Epoch 13/19
----------
train
 train-ing Epoch 14/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12785.0/12809
train Loss: 2.6064 | L_mi: 0.0000 | Acc: 0.9981 | Balanced-Acc: 0.9981
val
 val-ing Epoch 14/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2825.0/3203
val Loss: 3.4313 | L_mi: 0.0000 | Acc: 0.8820 | Balanced-Acc: 0.7830
Epoch 14/19
----------
train
 train-ing Epoch 15/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12776.0/12809
train Loss: 2.6057 | L_mi: 0.0000 | Acc: 0.9974 | Balanced-Acc: 0.9974
val
 val-ing Epoch 15/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2829.0/3203
val Loss: 3.4744 | L_mi: 0.0000 | Acc: 0.8832 | Balanced-Acc: 0.7723
Epoch 15/19
----------
train
 train-ing Epoch 16/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12784.0/12809
train Loss: 2.6049 | L_mi: 0.0000 | Acc: 0.9980 | Balanced-Acc: 0.9981
val
 val-ing Epoch 16/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2843.0/3203
val Loss: 3.4182 | L_mi: 0.0000 | Acc: 0.8876 | Balanced-Acc: 0.7657
Epoch 16/19
----------
train
 train-ing Epoch 17/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12784.0/12809
train Loss: 2.5999 | L_mi: 0.0000 | Acc: 0.9980 | Balanced-Acc: 0.9980
val
 val-ing Epoch 17/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2828.0/3203
val Loss: 3.3888 | L_mi: 0.0000 | Acc: 0.8829 | Balanced-Acc: 0.7756
Epoch 17/19
----------
train
 train-ing Epoch 18/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12785.0/12809
train Loss: 2.6010 | L_mi: 0.0000 | Acc: 0.9981 | Balanced-Acc: 0.9982
val
 val-ing Epoch 18/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2843.0/3203
val Loss: 3.4140 | L_mi: 0.0000 | Acc: 0.8876 | Balanced-Acc: 0.7740
Epoch 18/19
----------
train
 train-ing Epoch 19/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12787.0/12809
train Loss: 2.5960 | L_mi: 0.0000 | Acc: 0.9983 | Balanced-Acc: 0.9984
val
 val-ing Epoch 19/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2863.0/3203
val Loss: 3.4857 | L_mi: 0.0000 | Acc: 0.8938 | Balanced-Acc: 0.7709
New leading accuracy: 0.8938494920730591
Epoch 19/19
----------
train
 train-ing Epoch 20/20:   0%|          | 0/401 [00:00<?, ?it/s]
Accuracy: 12797.0/12809
train Loss: 2.5946 | L_mi: 0.0000 | Acc: 0.9991 | Balanced-Acc: 0.9992
val
 val-ing Epoch 20/20:   0%|          | 0/101 [00:00<?, ?it/s]
Accuracy: 2855.0/3203
val Loss: 3.5576 | L_mi: 0.0000 | Acc: 0.8914 | Balanced-Acc: 0.7716
Training complete in 344m 28s
Best val Acc: 0.893849  (epoch 18)
Training Complete
Model and results saved.

 Accuracy: 0.8938   Balanced Accuracy: 0.7709 

Done.