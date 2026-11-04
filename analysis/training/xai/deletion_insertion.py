import torch
import copy

import argparse
from pathlib import Path
import torch

from dataclass.config_class import TrainingConfig
from utils.utils import _get_ids_from_model_path,
from xai.utils import load_model, get_slides, process_slide, randomize_weights
from itertools import combinations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--wsi_ids", nargs="+", required=True)
    args = parser.parse_args()
 
    cfg = TrainingConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    spearman_coefs = {}
    jaccard_coefs = {}
    # Sortie : <results>/<model_id>/xai/<slide_id>/
    out_root = Path(cfg.path_save_results) / _get_ids_from_model_path(cfg.infer_model) / "xai"
    
    model = load_model(cfg, device)
    for slide in get_slides(cfg, args.wsi_ids):
        deletation, insertion, x_axis = compute_deletion_insertion_curves(slide, 10, model, cfg, device, out_root)
        del_rand, ins_rand = compute_random_baseline(model, features, device, n_steps=20, n_repeats=5
        )
        curve_plot(x_axis, deletation, del_rand, xlab="Proportion of removed patches", title="Deletation curve", out_root, file="curve_deletion.png")
        curve_plot(x_axis, insertion, ins_rand, xlab="Insertion Curve (Higher is better)", title="Insertion curve", out_root, file="curve_insertion.png")
        
if __name__ == "__main__":
    main()




