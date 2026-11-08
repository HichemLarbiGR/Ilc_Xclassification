"""Heatmaps d'attention + top/bottom patches (extraits à la volée du WSI).

Usage :
    python xai_heatmaps.py --config config/abmil_ilc.yaml --ids TCGA-A2-A04Y TCGA-A7-A426 TCGA-AC-A2FO TCGA-AC-A8OS TCGA-E9-A227
    python xai_heatmaps.py --config config/abmil_ilc.yaml --ids TCGA-C8-A3M8 TCGA-A2-A0EW TCGA-GM-A2DO TCGA-AR-A1AO TCGA-HN-A2NL
"""

import argparse
from pathlib import Path
import torch

from dataclass.config_class import TrainingConfig
from utils.utils import _get_ids_from_model_path
from xai.utils import load_model, get_slides, process_slide


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    args = parser.parse_args()
 
    cfg = TrainingConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # Sortie : <results>/<model_id>/xai/<slide_id>/
    out_root = Path(cfg.path_save_results) / _get_ids_from_model_path(cfg.infer_model) / "xai"
 
    model = load_model(cfg, device)
    
    for slide in get_slides(cfg, args.ids):
        process_slide(slide, model, cfg, device, out_root)
     
if __name__ == "__main__":
    main()