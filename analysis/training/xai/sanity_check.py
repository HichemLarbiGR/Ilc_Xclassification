import torch
import copy

import argparse
from pathlib import Path
import torch

from dataclass.config_class import TrainingConfig
from utils.utils import _get_ids_from_model_path
from xai.utils import load_model, get_slides, process_slide, randomize_weights, process_slide_sanity_check, plot_histogram


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

    spearman_coefs = {}

    for slide in get_slides(cfg, args.ids):

        coef = process_slide_sanity_check(slide, model, cfg, device, out_root)
        spearman_coefs[slide["id"]] = coef

    sp_list = [v for _,v in spearman_coefs.items()]
    
    plot_histogram(sp_list, 'Spearman coefficient histogram - sanity check (test set)', out_root, "sanity_check.png")

if __name__ == "__main__":
    main()




