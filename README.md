\# Ilc\_Xclassification 



Project for analysis and classification of \*\*TCGA-BRCA\*\* histopathology slides using pre-extracted \*\*UNI2-h\*\* embeddings (Mahmood Lab).



\## Objective



Explore and classify breast cancer cases (TCGA-BRCA) using UNI2-h features embeddings, with the option to generate heatmaps on whole-slide images.



\## Project StructureIlc\_Xclassification/

├── prepros/                 # Download \& preparation of embeddings

│   ├── download\_emb\_TCGA\_BRCA.py

│   └── README.md

├── analysis/                # Analyses, models, visualizations

│   └── README.md

└── README.md                # This filetext



\## Data Used



| Type | Source | Description |

|------|--------|-------------|

| Embeddings | \[MahmoodLab/UNI2-h-features](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features) | UNI2-h features (1536-dim) + patch coordinates |

| Images | GDC Portal | Whole-slide images `.svs` (only needed for heatmaps) |



Embeddings are split into:

\- `TCGA-BRCA\_IDC`

\- `TCGA-BRCA\_OTHERS`



\## Quick Setup



```bash

\# Clone the repository



git clone <repo-url>



cd Ilc\_Xclassification



\# Create a virtual environment



pip install -r requirements.txt



source env\_ils\_xclass/bin/activate   # Linux/Mac



pip install huggingface\_hub h5py numpy torch



Recommended Workflow



1- Download the embeddings : Go to prepros/ and run download\_emb\_TCGA\_BRCA.py, extract and organize the .h5 files



2- Run analyses in the analysis/ folder



```





\*\*Author H\_LARBI\*\*

