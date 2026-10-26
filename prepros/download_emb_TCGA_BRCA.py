from huggingface_hub import hf_hub_download, login
import os

os.makedirs("UNI2-h_features", exist_ok=True)

login(token="")

for filename in ["TCGA/TCGA-BRCA_IDC.tar.gz", "TCGA/TCGA-BRCA_OTHERS.tar.gz"]:
    path = hf_hub_download(
        repo_id="MahmoodLab/UNI2-h-features",
        filename=filename,
        repo_type="dataset",
        local_dir="UNI2-h_features"
    )