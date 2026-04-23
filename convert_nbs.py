import json
import os

notebooks = [
    "train_PatchAlign_FitzPatrick_InDomain_Lmi.ipynb",
    "train_PatchAlign_DDI_InDomain_Lmi.ipynb",
    "train_PatchAlign_FitzPatrick_OutDomain_Lmi.ipynb"
]

for nb in notebooks:
    if not os.path.exists(nb):
        print(f"File {nb} not found.")
        continue
        
    with open(nb, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    py_filename = nb.replace(".ipynb", ".py")
    
    with open(py_filename, "w", encoding="utf-8") as f:
        for cell in data.get("cells", []):
            if cell["cell_type"] == "code":
                source = cell.get("source", [])
                for line in source:
                    # Comment out ipython magic commands
                    if line.strip().startswith("!") or line.strip().startswith("%"):
                        f.write("# " + line)
                    else:
                        f.write(line)
                f.write("\n\n")

print("Conversion complete.")
