# Preprocessing : Get TCGA-BRCA UNI2-h embeddings

This folder contains scripts for downloading and preparing pre-extracted UNI2-h embeddings for the TCGA-BRCA project.

## Current Content

| File | Description |

|------|-------------|

| `download\_emb\_TCGA\_BRCA.py` | Downloads UNI2-h embedding archives for TCGA-BRCA (IDC + OTHERS) from Hugging Face |

## UNI2-h Embeddings

- **Model**: UNI2-h (ViT-H/14)

- **Feature dimension**: 1536

- **Patch size**: 256×256 at 20× magnification

- **Source**: \[MahmoodLab/UNI2-h-features](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features)



Downloaded files:

\- `TCGA/TCGA-BRCA\_IDC.tar.gz`

\- `TCGA/TCGA-BRCA\_OTHERS.tar.gz`



Each archive contains `.h5` files with:

\- `features`  shape `(1, N, 1536)`

\- `coords`  shape `(1, N, 2)`



## Usage

```bash

# Navigate to the prepros folder

cd prepros

source env\_ils\_xclass/bin/activate



# Run the download script (requires a valid Hugging Face token)

python download\_emb\_TCGA\_BRCA.py
```

Expected structure

prepros/
├── download_emb_TCGA_BRCA.py
├── UNI2-h_features/
│   └── TCGA/
│       ├── TCGA-BRCA_IDC.tar.gz
│       └── TCGA-BRCA_OTHERS.tar.gz
└── README.md
