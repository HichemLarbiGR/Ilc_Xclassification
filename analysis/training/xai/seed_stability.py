import torch
import copy

import argparse
from pathlib import Path

from dataclass.config_class import TrainingConfig
from utils.utils import _get_ids_from_model_path
from xai.utils import load_model, get_slides, process_slide, randomize_weights, find_model_by_id, load_model_seed, process_seed_analysis,plot_histogram
from itertools import combinations



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_ids", nargs="+", required=True)
    parser.add_argument("--wsi_ids", nargs="+", required=True)
    args = parser.parse_args()
 
    cfg = TrainingConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    list_models = find_model_by_id(args.model_ids, cfg.save_model_path)
    
    spearman_coefs = {}
    jaccard_coefs = {}
    # Sortie : <results>/<model_id>/xai/<slide_id>/
    out_root = Path(cfg.path_save_results) / _get_ids_from_model_path(cfg.infer_model) / "xai"
    
    for m1, m2 in combinations(list_models, 2):
        model1 = load_model_seed(cfg, device, Path(cfg.save_model_path) / m1)
        model2 = load_model_seed(cfg, device, Path(cfg.save_model_path) / m2)

        for slide in get_slides(cfg, args.wsi_ids):
            coef_sp, coef_jc = process_seed_analysis(slide, 0.1, model1, model2, cfg, device, out_root)
            comb_name = f"{m1}_vs_{m2}_slide_{slide}"
            spearman_coefs[comb_name] = coef_sp
            jaccard_coefs[comb_name] = coef_jc

    sp_list = [v for _,v in spearman_coefs.items()]
    jc_list = [v for _,v in jaccard_coefs.items()]
    
    plot_histogram(sp_list, f"Spearman coefficient histogram on attention - seed stability ({list_models[0]} test set)", out_root, "spearman_histogram.png")
    plot_histogram(jc_list, f"Jaccard coefficient histogram on attention - seed stability ({list_models[0]} test set)", out_root,"jaccard_histogram.png")


if __name__ == "__main__":
    main()




