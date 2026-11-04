import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from matplotlib.patches import Rectangle


from dataclass.config_class import TrainingConfig
from models.MIL_models import build_model_class
from utils.utils import load_json, _get_ids_from_model_path, build_name_h5_path

from PIL import Image, PngImagePlugin
from scipy.spatial import distance
from scipy import stats
from operator import itemgetter
from pathlib import Path

import glob, re

PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024

def load_model(cfg, device):
    model = build_model_class(cfg.architecture_name)(
        num_classes=cfg.num_classes, in_dim=cfg.input_dim
    )
    model.load_state_dict(torch.load(cfg.infer_model, map_location=device))
    return model.to(device).eval()
 
 
def get_slides(cfg, ids):
    """Retrouve les entrées du JSON (id, filename, target) à partir des ids."""
    data = load_json(cfg.json_split_wsi_target_path)
    all_slides = [s for v in data.values() for s in v] if isinstance(data, dict) else data
    by_id = {str(s["id"]): s for s in all_slides}
 
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"ids absents du JSON : {missing}")
    return [by_id[i] for i in ids]
 
 
def load_h5(path):
    with h5py.File(path, "r") as f:
        features = f["features"][:]
        if "patch_size_level0" in f["coords"].attrs:
            # Format Trident
            coords = f["coords"][:]
            patch_size_level0 = int(f["coords"].attrs["patch_size_level0"])
            patch_size = int(f["coords"].attrs["patch_size"])
        else:
            # Format H-optimus-0 : attributs à la racine, coords flottantes au niveau 0
            coords = f["coords_level0_float"][:]
            patch_size = int(f.attrs["tile_px"])
            patch_size_level0 = round(patch_size * f.attrs["target_mpp"] / f.attrs["source_mpp_x"])
    if features.ndim == 3:
        features = features[0]
    if coords.ndim == 3:
        coords = coords[0]
    return features, coords, patch_size_level0, patch_size
 
 
def get_attention(model, features, device):
    """Retourne l'attention par patch (N,) et les probas de la slide."""
    x = torch.from_numpy(features).float().to(device)
    with torch.no_grad():
        logits, attn = model([x])
    return attn[0].cpu().numpy(), torch.softmax(logits, dim=1)[0].cpu().numpy()
 
 
def plot_heatmap(png_path, coords, patch_size_level0, attn, title, out_path, cfg):
    """Dessine les patches colorés par attention (min-max par slide, 0-1) sur le PNG 1/16.
 
    attn reste le softmax brut ; la normalisation ne sert qu'à l'affichage.
    score_threshold s'applique sur la valeur normalisée.
    """
    scale = cfg.scale
    size = patch_size_level0 * scale
    cmap = plt.get_cmap(cfg.cmap)
    norm = plt.Normalize(attn.min(), attn.max())
 
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(Image.open(png_path).convert("RGB"))
 
    for (x, y), a in zip(coords, attn):
        if norm(a) > cfg.score_threshold:
            ax.add_patch(Rectangle((x * scale, y * scale), size, size,
                                   facecolor=cmap(norm(a)), alpha=cfg.alpha, linewidth=0))
 
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, shrink=0.7, label="Attention (min-max par slide)")
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
 
 
def save_patches(wsi, coords, attn, indices, patch_size_level0, patch_size, out_dir):
    """Lit les patches directement dans le WSI (niveau 0) et les sauvegarde."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, i in enumerate(indices, start=1):
        x, y = int(coords[i][0]), int(coords[i][1])
        patch = wsi.read_region((x, y), 0, (patch_size_level0, patch_size_level0))
        patch = patch.convert("RGB").resize((patch_size, patch_size))
        patch.save(out_dir / f"{rank:02d}_attn{attn[i]:.2e}_x{x}_y{y}.png")
  
 
def process_slide(slide, model, cfg, device, out_root):
    slide_id = str(slide["id"])
    stem = Path(slide["filename"]).stem
    out_dir = out_root / slide_id
    out_dir.mkdir(parents=True, exist_ok=True)
 
    features, coords, ps0, ps = load_h5(Path(build_name_h5_path(cfg.h5_dir, cfg.foundation_model_name)) / slide["filename"])
    attn, probs = get_attention(model, features, device)
 
    pred = int(probs.argmax())
    title = (f"{slide_id} | vrai: {cfg.classes_name[slide['target']]} | "
             f"prédit: {cfg.classes_name[pred]} ({probs[pred]:.2f})")
 
    plot_heatmap(Path(cfg.png_path) / f"{stem}_1_{cfg.scale_factor}.png", coords, ps0, attn, title,
                 out_dir / f"heatmap.{cfg.figures_extension}", cfg)
 
    order = np.argsort(attn)
    wsi = openslide.OpenSlide(str(next(Path(cfg.wsi_path).glob(f"{stem}.*"))))
    save_patches(wsi, coords, attn, order[::-1][:cfg.top_k], ps0, ps, out_dir / "top")
    save_patches(wsi, coords, attn, order[:cfg.bottom_k], ps0, ps, out_dir / "bottom")
    wsi.close()
 
    print(f"{slide_id} : {title}")

def randomize_weights(model):
    model_rand = copy.deepcopy(model)
    for param in model_rand.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.01)
    return model_rand

def process_slide_sanity_check(slide, model, cfg, device, out_root):
    slide_id = str(slide["id"])
    stem = Path(slide["filename"]).stem
    out_dir = out_root / slide_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    #Inference not randomized weights
    features, coords, ps0, ps = load_h5(Path(build_name_h5_path(cfg.h5_dir, cfg.foundation_model_name)) / slide["filename"])
    attn, probs = get_attention(model, features, device)

    #Randomized weights
    model_rd = randomize_weights(model)
    features_rd, coords_rd, ps0_rd, ps_rd = load_h5(Path(build_name_h5_path(cfg.h5_dir, cfg.foundation_model_name)) / slide["filename"])
    attn_rd, probs_rd = get_attention(model, features, device)

    pred_rd = int(probs_rd.argmax())
    title_rd = (f"{slide_id} | vrai: {cfg.classes_name[slide['target']]} | "
             f"prédit: {cfg.classes_name[pred_rd]} ({probs_rd[pred_rd]:.2f})")
 
    plot_heatmap(Path(cfg.png_path) / f"{stem}_1_{cfg.scale_factor}.png", coords_rd, ps0_rd, attn_rd, title_rd,
                 out_dir / f"heatmap_sanity_check.{cfg.figures_extension}", cfg)

    

    
    return spearman_attn(attn, attn_rd)

def jaccard_coord(list_a, list_b):
    coef_jc = distance.jaccard(list_a, list_b)
    return coef_jc



def spearman_attn(attn_trained, attn_random):

    attn_trained = np.array(attn_trained).flatten()
    attn_random = np.array(attn_random).flatten()
    
    res = stats.spearmanr(attn_trained, attn_random)
    coef_sp = res.statistic
    
    if np.isnan(coef_sp):
        return 0.0
        
    return coef_sp

def plot_histogram(x, title, out, file):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')

    # Historigramme épuré (Indigo moderne avec contours blancs)
    ax.hist(
        x,
        bins=30,
        color='#4F46E5',       # Indigo moderne (ou #0D9488 pour Teal)
        edgecolor='white',     # Contours blancs pour une séparation propre
        linewidth=1.2,
        alpha=0.9,
        rwidth=0.88            # Espacement subtil entre les barres
    )

    # Textes et labels (Gris ardoise au lieu du noir pur)
    ax.set_xlabel('Spearman coefficients', fontsize=11, fontweight='medium', color='#334155', labelpad=10)
    ax.set_ylabel('Frequency', fontsize=11, fontweight='medium', color='#334155', labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', color='#0F172A', pad=15)

    # Grille d'arrière-plan discrète (placée DERRIÈRE les barres)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6, color='#E2E8F0')

    # Supression des bordures supérieures et droites
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')
    ax.spines['bottom'].set_color('#CBD5E1')

    out_path = out / file

    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def find_model_by_id(id_patterns, model_path):
    
    list_models = []

    for id_p in id_patterns:
        model_id_p_list = glob.glob(Path(model_path) / f'./.*{id_p}.*')
        list_models += list_models
    
    return list_models


def load_model_seed(cfg, device, model_path):
    model = build_model_class(cfg.architecture_name)(
        num_classes=cfg.num_classes, in_dim=cfg.input_dim
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()


def process_seed_analysis(
    slide, k_percent, model1, model2, cfg, device, out_root
):
    # Chargement et inférences
    h5_path = (
        Path(build_name_h5_path(cfg.h5_dir, cfg.foundation_model_name))
        / slide["filename"]
    )
    features, coords, _, _ = load_h5(h5_path)

    attn1, probs1 = get_attention(model1, features, device)
    attn2, probs2 = get_attention(model2, features, device)

    top_k_n = round(len(coords) * k_percent)

    # Top-K pour Model 1 (tri décroissant pour garder les plus fortes attentions)
    data1 = [(attn1[i], coords[i]) for i in range(len(coords))]
    data1.sort(key=itemgetter(0), reverse=True)
    top_k_coords1 = {coord for _, coord in data1[:top_k_n]}

    # Top-K pour Model 2
    data2 = [(attn2[i], coords[i]) for i in range(len(coords))]
    data2.sort(key=itemgetter(0), reverse=True)
    top_k_coords2 = {coord for _, coord in data2[:top_k_n]}

    # Création des masques binaires (0 ou 1)
    jc_data1 = [1 if coord in top_k_coords1 else 0 for coord in coords]
    jc_data2 = [1 if coord in top_k_coords2 else 0 for coord in coords]

    return spearman_attn(attn1, attn2), jaccard_coord(jc_data1, jc_data2)


def process_loo_analysis(slide, model, device, cfg, out_root):
    """
    Calcule la chute de probabilité (LOO) pour un patch spécifique.
    """

    loo_list = []
    h5_path = (
        Path(build_name_h5_path(cfg.h5_dir, cfg.foundation_model_name))
        / slide["filename"]
    )
    
    features, coords, _, _ = load_h5(h5_path)

    attn, probs = get_attention(model, features, device)
    
    features_occ = features.copy()
    
    bg_vector = features.mean(axis=0)

    for patch_idx in range(len(features)):
        features_occ[patch_idx] = bg_vector
        attn_occ, probs_occ = get_attention(model, features_occ, device)
        score_loo = proba_base - proba_occ
        loo_list.apppend(score_loo)

    return spearman_attn(attn, score_loo)


def compute_deletion_insertion_curves(slide, model, device, cfg, out_root, 
                                      step_ratio=0.05, n_steps=20):
    """
    Calcule les courbes de Deletion et d'Insertion pour une lame.
    
    Args:
        features: np.array [N, in_dim] - les embeddings de la lame
        attn_scores: np.array [N] - les scores d'attention
        step_ratio: proportion de patchs à ajouter/retirer à chaque étape (ex: 0.05 = 5%)
        n_steps: nombre d'étapes (paliers) pour la courbe
    """

    deletion_probs = []
    insertion_probs = []

    h5_path = (
        Path(build_name_h5_path(cfg.h5_dir, cfg.foundation_model_name))
        / slide["filename"]
    )
    
    features, coords, _, _ = load_h5(h5_path)

    n_patches = len(features)

    _, probs_base = get_attention(model, features, device)
    pred_class = probs_base.argmax().item()
    proba_base = probs_base[pred_class].item()
    
    features_occ = features.copy()
    
    bg_vector = features.mean(axis=0)
     
    sorted_indices = np.argsort(attn_scores)[::-1] # Du plus important au moins important
    
    # 4. Définir les paliers (ex: 0%, 5%, 10%, ..., 100%)
    steps = np.linspace(0, n_patches, n_steps + 1, dtype=int)
    
    for k in steps:
        # --- DELETION ---
        # On part de l'image complète et on retire les k patchs les plus importants
        features_del = features.copy()
        if k > 0:
            indices_to_remove = sorted_indices[:k]
            features_del[indices_to_remove] = bg_vector
        
        _, probs_del = get_attention(model, features_del, device)
        deletion_probs.append(probs_del[pred_class].item())
        
        # --- INSERTION ---
        # On part d'une image "vide" (que du background) et on ajoute les k patchs les plus importants
        features_ins = np.tile(bg_vector, (n_patches, 1))
        if k > 0:
            indices_to_insert = sorted_indices[:k]
            features_ins[indices_to_insert] = features[indices_to_insert]
        
        _, probs_ins = get_attention(model, features_ins, device)
        insertion_probs.append(probs_ins[pred_class].item())
        
    return deletion_probs, insertion_probs, steps / n_patches

def compute_random_baseline(slide, model, device, cfg, out_root, 
                                       n_steps=20, n_repeats=5):
    """
    Calcule la baseline aléatoire (en moyennant sur plusieurs tirages aléatoires).
    """
    n_patches = len(features)
    bg_vector = features.mean(axis=0)
    
    _, probs_base = get_attention(model, features, device)
    pred_class = probs_base.argmax().item()
    
    steps = np.linspace(0, n_patches, n_steps + 1, dtype=int)
    rand_del_probs = []
    rand_ins_probs = []
    
    for k in steps:
        del_probs_k = []
        ins_probs_k = []
        
        for _ in range(n_repeats):
            # Tirage aléatoire des indices
            random_indices = np.random.permutation(n_patches)
            
            # Deletion aléatoire
            features_del = features.copy()
            if k > 0:
                features_del[random_indices[:k]] = bg_vector
        
            _, probs_del = get_attention(model, features_del, device)
            del_probs_k.append(probs_del[pred_class].item())
            
            # Insertion aléatoire
            features_ins = np.tile(bg_vector, (n_patches, 1))
            if k > 0:
                features_ins[random_indices[:k]] = features[random_indices[:k]]
    
            _, probs_ins = get_attention(model, tensor_ins, device)
            ins_probs_k.append(probs_ins[pred_class].item())
            
        rand_del_probs.append(np.mean(del_probs_k))
        rand_ins_probs.append(np.mean(ins_probs_k))
        
    return rand_del_probs, rand_ins_probs


def curve_plot(x_axis, list_attn, list_rand, xlab, title, out, file):

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')

    # Historigramme épuré (Indigo moderne avec contours blancs)
    ax.plot(
        x_axis, list_attn, label="Attention", color="red", linewidth=2
        )
    
    ax.plot(
        x_axis, list_rand, label="Aléatoire", color="gray", linestyle="--"
        )

    # Textes et labels (Gris ardoise au lieu du noir pur)
    ax.set_xlabel(xlab, fontsize=11, fontweight='medium', color='#334155', labelpad=10)
    ax.set_ylabel('Probability of predicted class', fontsize=11, fontweight='medium', color='#334155', labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', color='#0F172A', pad=15)

    # Grille d'arrière-plan discrète (placée DERRIÈRE les barres)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6, color='#E2E8F0')

    # Supression des bordures supérieures et droites
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')
    ax.spines['bottom'].set_color('#CBD5E1')

    out_path = out / file

    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    return
    
 
    
