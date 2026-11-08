import torch
import argparse
from pathlib import Path

from dataclass.config_class import TrainingConfig
from utils.utils import _get_ids_from_model_path
from xai.utils import load_model, get_slides, build_name_h5_path, load_h5, get_attention

from xai.utils import (
    compute_deletion_insertion_curves,
    compute_random_baseline,
    compute_inverted_attention_curves,
    curve_plot,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sortie : <results>/<model_id>/xai/<slide_id>/
    out_root = Path(cfg.path_save_results) / _get_ids_from_model_path(cfg.infer_model) / "xai"
    out_root.mkdir(parents=True, exist_ok=True)

    model = load_model(cfg, device)

    for slide in get_slides(cfg, args.ids):
        slide_id = str(slide["id"])
        out_dir = out_root / slide_id
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {slide_id} ===")

        # --- Courbes attention ---
        del_attn, ins_attn, x_axis = compute_deletion_insertion_curves(
            slide, model, device, cfg, out_root, n_steps=20
        )

        # --- Recharge les features pour la baseline aléatoire ---
        h5_path = Path(build_name_h5_path(cfg.h5_dir, cfg.foundation_model_name)) / slide["filename"]
        features, _, _, _ = load_h5(h5_path)

        # --- Baseline aléatoire ---
        del_rand, ins_rand = compute_random_baseline(
            features, model, device, n_steps=20, n_repeats=5
        )

        # --- Baseline attention inversée ---
        del_inv, ins_inv = compute_inverted_attention_curves(
            slide, model, device, cfg, out_root, n_steps=20
        )

        # --- AUC ---
        auc_del_attn = float(torch.trapz(torch.tensor(del_attn), torch.tensor(x_axis)))
        auc_del_rand = float(torch.trapz(torch.tensor(del_rand), torch.tensor(x_axis)))
        auc_del_inv  = float(torch.trapz(torch.tensor(del_inv),  torch.tensor(x_axis)))
        auc_ins_attn = float(torch.trapz(torch.tensor(ins_attn), torch.tensor(x_axis)))
        auc_ins_rand = float(torch.trapz(torch.tensor(ins_rand), torch.tensor(x_axis)))
        auc_ins_inv  = float(torch.trapz(torch.tensor(ins_inv),  torch.tensor(x_axis)))

        print(f"Deletion  AUC  attn={auc_del_attn:.4f}  rand={auc_del_rand:.4f}  inv={auc_del_inv:.4f}")
        print(f"Insertion AUC  attn={auc_ins_attn:.4f}  rand={auc_ins_rand:.4f}  inv={auc_ins_inv:.4f}")
        print(f"Δ deletion  (rand - attn) = {auc_del_rand - auc_del_attn:+.4f}")
        print(f"Δ insertion (attn - rand) = {auc_ins_attn - auc_ins_rand:+.4f}")

        # --- Plots ---
        curve_plot(
            x_axis, del_attn, del_rand, del_inv,
            xlab="Proportion of removed patches",
            title=f"{slide_id} — Deletion curve",
            out=out_dir, file="curve_deletion.png",
        )
        curve_plot(
            x_axis, ins_attn, ins_rand, ins_inv,
            xlab="Proportion of inserted patches",
            title=f"{slide_id} — Insertion curve",
            out=out_dir, file="curve_insertion.png",
        )


if __name__ == "__main__":
    main()