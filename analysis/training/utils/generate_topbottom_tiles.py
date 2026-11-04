import shutil
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MIL.model import AttentionMILNecFib, AttentionMIL, EarlyStopping
from MIL.data_loader_bag import SlideBagNecFibDataset, collate_necfib_bags
import random 
from torch import nn
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import glob
import json
import math
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import warnings
import re
warnings.filterwarnings('ignore')

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# PARAMÈTRES
# ============================================================
H5_DIR = "/home/userx/work/20251227_EGFR_necrosis/necrosis_quanti/h5_files"
PATH_CLINICAL_SCORES = "/home/userx/work/20251227_EGFR_necrosis/necrosis_quanti/data/clinical_data/slide_scores_necfib_binary_test.json"
PATH_SAVE_RESULTS = "/home/userx/work/20251227_EGFR_necrosis/necrosis_quanti/MIL/results/training_results_abmil.csv"
OUTPUT_DIR = "/home/userx/work/20251227_EGFR_necrosis/necrosis_quanti/MIL/results/30p/risk/uni/heatmaps"
MODEL_WEIGHTS = "/home/userx/work/20251227_EGFR_necrosis/necrosis_quanti/MIL/model_weights/risks/final_model_stratified_risks_uni_20260728_split2.pth"  # À définir si nécessaire
TASK = "risk"  # Options: "risk" or "necfib"
INPUT_DIM = 1536
N_CLASSES = 2
EPOCHS = 200
BATCH_SIZE = 1
LR = 1e-4
MODEL_NAME = "uni"
SCALE_FACTOR = 16
SCALE = 1.0 / SCALE_FACTOR
N_SPLITS = 5
TEST_SIZE = 0.2
TOP_K = 10
BOTTOM_K = 10
SCORE_THRESHOLD = 0
CMAP = 'hot'
ALPHA = 0.6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Device: {device}")

# ============================================================
# FONCTIONS POUR HEATMAP ET PATCHES
# ============================================================

def load_model(model_class, pth_path, device, task=TASK):
    """Charge un modèle depuis un fichier .pth."""
    if task=="risk":
        model = model_class(num_classes=3, in_dim=1536).to(device)
    elif task=="necfib":
        model = model_class().to(device)
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"❌ Fichier .pth introuvable: {pth_path}")
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Modèle chargé avec succès")
    return model

def load_h5_data(h5_path):
    """Charge les données d'un fichier H5."""
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"❌ Fichier H5 introuvable: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        embeddings = f['embeddings'][:]
        coords = f['coordinates'][:]
        tile_ids = f['tile_ids'][:].astype(str) if 'tile_ids' in f else None
    return embeddings, coords, tile_ids

def sample_patches(embeddings, coords, tile_ids=None, use_all_patches=True, num_patches=None):
    """Échantillonne les patches si nécessaire."""
    if use_all_patches:
        return embeddings, coords, tile_ids
    
    if num_patches is None:
        num_patches = 20000
    
    if len(embeddings) > num_patches:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(embeddings), num_patches, replace=False)
        embeddings_used = embeddings[indices]
        coords_used = coords[indices]
        tile_ids_used = tile_ids[indices] if tile_ids is not None else None
        print(f"✅ Échantillonnage de {num_patches} patches (sur {len(embeddings)})")
        return embeddings_used, coords_used, tile_ids_used
    else:
        print(f"✅ Utilisation des {len(embeddings)} patches disponibles")
        return embeddings, coords, tile_ids

def normalize_scores(scores):
    """Normalise les scores entre 0 et 1."""
    if scores.max() > scores.min():
        return (scores - scores.min()) / (scores.max() - scores.min())
    else:
        return np.ones_like(scores)


def create_heatmap_figure(bg, coords, scores_norm, predictions, attention_scores, 
                         title_prefix, output_path, cmap='hot', alpha=0.6,
                         scale=SCALE, score_threshold=0):
    """Crée et sauvegarde une figure heatmap.
    
    Args:
        bg: Image de fond (PIL Image)
        coords: Coordonnées des patches (N, 2)
        scores_norm: Scores d'attention normalisés (N,)
        predictions: Valeur prédite
        attention_scores: Scores d'attention bruts (N,)
        title_prefix: Préfixe du titre
        output_path: Chemin de sauvegarde
        cmap: Colormap
        alpha: Transparence
        scale: Facteur d'échelle (1/16 par défaut)
        score_threshold: Seuil d'affichage des patches
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(bg)
    
    # Taille des tuiles adaptée avec scale
    tile_size_px = 224 * scale
    half = tile_size_px / 2
    
    norm = plt.Normalize(vmin=scores_norm.min(), vmax=scores_norm.max())
    
    try:
        import matplotlib as mpl
        cmap_obj = mpl.colormaps[cmap]
    except (AttributeError, KeyError):
        cmap_obj = plt.cm.get_cmap(cmap)
    
    n_displayed = 0
    for i, (x, y) in enumerate(coords):
        score = scores_norm[i]
        if score > score_threshold:
            cx = x * scale
            cy = y * scale
            rect = patches.Rectangle(
                (cx - half, cy - half), tile_size_px, tile_size_px,
                linewidth=0, facecolor=cmap_obj(norm(score)), alpha=alpha
            )
            ax.add_patch(rect)
            n_displayed += 1
    
    print(f"📊 {n_displayed}/{len(coords)} patches affichés (seuil={score_threshold})")
    
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("Score d'attention (normalisé)")
    
    title = f'{title_prefix}\n'
    title += f'Prédiction: {predictions} | '
    title += f'Min: {attention_scores.min():.4f}, Max: {attention_scores.max():.4f}, Mean: {attention_scores.mean():.4f}'
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Heatmap sauvegardée : {output_path}")
    return output_path


def find_png_file(tile_name, all_png_files):
    """Trouve le fichier PNG correspondant à un tile_id."""
    match = re.search(r'r(\d+)[-_]c(\d+)', tile_name)
    if not match:
        return None
    r_h5, c_h5 = int(match.group(1)), int(match.group(2))
    
    for filepath in all_png_files:
        basename = os.path.basename(filepath)
        match_file = re.search(r'r(\d+)[-_]c(\d+)', basename)
        if match_file:
            r_file, c_file = int(match_file.group(1)), int(match_file.group(2))
            if r_file == r_h5 and c_file == c_h5:
                return filepath
    return None

def generate_heatmap_risk(slide_num, pth_path, model_class, device,
                         h5_dir="h5_files", img_dir="data/wsi_raw/training_png",
                         output_dir=OUTPUT_DIR, score_threshold=SCORE_THRESHOLD,
                         cmap=CMAP, alpha=ALPHA, scale=SCALE,
                         use_all_patches=True, num_patches=None, model_name=MODEL_NAME):
    """Génère une heatmap pour la tâche Risk (classification binaire)."""
    print(f"\n{'='*60}")
    print(f"Génération de la heatmap pour le slide {slide_num} (Risk)")
    print(f"{'='*60}")
    
    # Chargement du modèle
    model = load_model(model_class, pth_path, device, task=TASK)
    
    # Chargement des données H5
    h5_path = os.path.join(h5_dir, f"EGFR_{slide_num:03d}_{model_name}.h5")
    embeddings, coords, tile_ids = load_h5_data(h5_path)
    print(f"📊 {len(embeddings)} patches trouvés dans le H5")
    
    # Échantillonnage
    embeddings_used, coords_used, _ = sample_patches(embeddings, coords, tile_ids, use_all_patches, num_patches)
    
    # Inférence
    print("🔄 Inférence en cours...")
    embeddings_tensor = torch.from_numpy(embeddings_used).float().to(device)
    with torch.no_grad():
        logits_risk, attn_weights = model([embeddings_tensor])
    
    attention_scores = attn_weights[0].cpu().numpy()
    pred_risk = torch.argmax(logits_risk, dim=1).item()
    
    print(f"📊 Attention - Min: {attention_scores.min():.6f}, Max: {attention_scores.max():.6f}")
    print(f"📊 Prédiction - Risque: {pred_risk}")
    
    # Normalisation
    scores_norm = normalize_scores(attention_scores)
    
    # Chargement de l'image de fond
    bg = load_background_image(slide_num, img_dir)
    
    # Création de la heatmap
    output_path = os.path.join(output_dir, f'heatmap_attention_slide_{slide_num}.svg')
    title_prefix = f'Slide {slide_num} – Heatmap d\'attention MIL (Risk)'
    create_heatmap_figure(
        bg, coords_used, scores_norm, pred_risk, attention_scores,
        title_prefix, output_path, cmap, alpha, scale  # ⚠️ Passage de scale
    )
    
    print(f"{'='*60}\n")
    return output_path, attention_scores, logits_risk

def generate_heatmap_necfib(slide_num, pth_path, model_class, device,
                           h5_dir="h5_files", img_dir="data/wsi_raw/training_png",
                           output_dir=OUTPUT_DIR, score_threshold=SCORE_THRESHOLD,
                           cmap=CMAP, alpha=ALPHA, scale=SCALE,
                           use_all_patches=True, num_patches=None, model_name=MODEL_NAME):
    """Génère une heatmap pour la tâche NecFib (classification multi-tâches)."""
    print(f"\n{'='*60}")
    print(f"Génération de la heatmap pour le slide {slide_num} (NecFib)")
    print(f"{'='*60}")
    
    # Chargement du modèle
    model = load_model(model_class, pth_path, device, task=TASK)
    
    # Chargement des données H5
    h5_path = os.path.join(h5_dir, f"EGFR_{slide_num:03d}_{model_name}.h5")
    embeddings, coords, tile_ids = load_h5_data(h5_path)
    print(f"📊 {len(embeddings)} patches trouvés dans le H5")
    
    # Échantillonnage
    embeddings_used, coords_used, _ = sample_patches(embeddings, coords, tile_ids, use_all_patches, num_patches)
    
    # Inférence
    print("🔄 Inférence en cours...")
    embeddings_tensor = torch.from_numpy(embeddings_used).float().to(device)
    with torch.no_grad():
        logits_nec, logits_fib, attn_weights_nec, attn_weights_fib = model([embeddings_tensor])
    
    attention_scores_nec = attn_weights_nec[0].cpu().numpy()
    attention_scores_fib = attn_weights_fib[0].cpu().numpy()
    
    pred_nec = torch.argmax(logits_nec, dim=1).item()
    pred_fib = torch.argmax(logits_fib, dim=1).item()
    
    print(f"📊 Attention Nécrose - Min: {attention_scores_nec.min():.6f}, Max: {attention_scores_nec.max():.6f}")
    print(f"📊 Attention Fibrose - Min: {attention_scores_fib.min():.6f}, Max: {attention_scores_fib.max():.6f}")
    print(f"📊 Prédictions - Nécrose: {pred_nec}, Fibrose: {pred_fib}")
    
    # Normalisation
    scores_norm_nec = normalize_scores(attention_scores_nec)
    scores_norm_fib = normalize_scores(attention_scores_fib)
    
    # Chargement de l'image de fond
    bg = load_background_image(slide_num, img_dir)
    
    # Heatmap Nécrose
    output_path_nec = os.path.join(output_dir, f'heatmap_attention_slide_{slide_num}_necrosis.svg')
    title_prefix_nec = f'Slide {slide_num} – Heatmap d\'attention MIL (Nécrose)'
    create_heatmap_figure(
        bg, coords_used, scores_norm_nec, pred_nec, attention_scores_nec,
        title_prefix_nec, output_path_nec, cmap, alpha, scale  # ⚠️ Passage de scale
    )
    
    # Heatmap Fibrose
    output_path_fib = os.path.join(output_dir, f'heatmap_attention_slide_{slide_num}_fibrosis.svg')
    title_prefix_fib = f'Slide {slide_num} – Heatmap d\'attention MIL (Fibrose)'
    create_heatmap_figure(
        bg, coords_used, scores_norm_fib, pred_fib, attention_scores_fib,
        title_prefix_fib, output_path_fib, cmap, alpha, scale  # ⚠️ Passage de scale
    ) 
    
    print(f"{'='*60}\n")
    return [output_path_nec, output_path_fib], (attention_scores_nec, attention_scores_fib), (logits_nec, logits_fib)

def load_background_image(slide_num, img_dir):
    """Charge l'image de fond pour un slide donné."""
    png_pattern = os.path.join(img_dir, f"EGFR_{slide_num:03d}*.png")
    png_files = glob.glob(png_pattern)
    if not png_files:
        print(f"⚠️ Aucun PNG trouvé pour slide {slide_num}, utilisation d'un fond blanc")
        return Image.new('RGB', (2000, 2000), 'white')
    else:
        png_path = png_files[0]
        bg = Image.open(png_path).convert('RGB')
        print(f"🖼️ Image fond : {png_path} ({bg.size[0]}x{bg.size[1]})")
        return bg

def extract_top_bottom_patches_risk(slide_num, model, device, 
                                   h5_dir="h5_files", png_dir="data/wsi_raw/tiles_png",
                                   output_dir=OUTPUT_DIR, top_k=TOP_K, bottom_k=BOTTOM_K,
                                   model_name=MODEL_NAME):
    """Extrait et copie les patches les plus/moins importants pour Risk."""
    return _extract_top_bottom_patches_generic(
        slide_num, model, device, h5_dir, png_dir, output_dir,
        top_k, bottom_k, model_name, task_type="risk"
    )

def extract_top_bottom_patches_necfib(slide_num, model, device,
                                     h5_dir="h5_files", png_dir="data/wsi_raw/tiles_png",
                                     output_dir=OUTPUT_DIR, top_k=TOP_K, bottom_k=BOTTOM_K,
                                     model_name=MODEL_NAME):
    """Extrait et copie les patches les plus/moins importants pour NecFib."""
    return _extract_top_bottom_patches_generic(
        slide_num, model, device, h5_dir, png_dir, output_dir,
        top_k, bottom_k, model_name, task_type="necfib"
    )






def _extract_top_bottom_patches_generic(slide_num, model, device,
                                       h5_dir, png_dir, output_dir,
                                       top_k, bottom_k, model_name, task_type):
    """Fonction générique d'extraction des top/bottom patches."""
    # Création des dossiers principaux
    slide_output_dir = os.path.join(output_dir, f"slide_{slide_num:03d}")
    os.makedirs(slide_output_dir, exist_ok=True)
    
    # Chargement du H5
    h5_path = os.path.join(h5_dir, f"EGFR_{slide_num:03d}_{model_name}.h5")
    embeddings, _, tile_ids = load_h5_data(h5_path)
    print(f"🔍 Slide {slide_num}: {len(embeddings)} patches chargés.")
    
    # Calcul des poids d'attention
    model.eval()
    with torch.no_grad():
        embeddings_tensor = torch.from_numpy(embeddings).float().to(device)
        
        if task_type == "risk":
            _, attn_weights = model([embeddings_tensor])
            attention_scores = attn_weights[0].cpu().numpy()
            scores_list = [("", attention_scores)]  # Pas de suffixe pour risk
        else:  # necfib
            _, _, attn_weights_nec, attn_weights_fib = model([embeddings_tensor])
            attention_scores_nec = attn_weights_nec[0].cpu().numpy()
            attention_scores_fib = attn_weights_fib[0].cpu().numpy()
            scores_list = [
                ("_necrosis", attention_scores_nec),
                ("_fibrosis", attention_scores_fib)
            ]
    
    # Chargement des PNG
    slide_png_dir = os.path.join(png_dir, f"{slide_num:03d}")
    if not os.path.isdir(slide_png_dir):
        raise FileNotFoundError(f"Dossier PNG introuvable : {slide_png_dir}")
    
    all_png_files = glob.glob(os.path.join(slide_png_dir, "*.png"))
    if not all_png_files:
        raise FileNotFoundError(f"Aucun PNG trouvé dans {slide_png_dir}")
    
    results = {}
    
    for suffix, scores in scores_list:
        # Création des dossiers avec suffixe
        top_dir = os.path.join(slide_output_dir, f"top{suffix}")
        bottom_dir = os.path.join(slide_output_dir, f"bottom{suffix}")
        os.makedirs(top_dir, exist_ok=True)
        os.makedirs(bottom_dir, exist_ok=True)
        
        # Tri des indices par score décroissant pour les top et croissant pour les bottom
        sorted_indices_desc = np.argsort(scores)[::-1]  # Du plus haut au plus bas
        sorted_indices_asc = np.argsort(scores)  # Du plus bas au plus haut
        
        # Fonction pour copier les patches avec fallback
        def copy_patches_with_fallback(indices_list, target_dir, k, mode="top"):
            """
            Copie les patches correspondant aux indices.
            Si un patch n'est pas trouvé, prend le suivant dans la liste.
            """
            copied_count = 0
            current_pos = 0
            failed_indices = []
            
            while copied_count < k and current_pos < len(indices_list):
                idx = indices_list[current_pos]
                tile_name = tile_ids[idx]
                src = find_png_file(tile_name, all_png_files)
                
                if src:
                    dst = os.path.join(target_dir, os.path.basename(src))
                    shutil.copy2(src, dst)
                    copied_count += 1
                    if current_pos > 0:
                        print(f"   ✅ Patch trouvé après {current_pos} essais: {tile_name}")
                else:
                    failed_indices.append((current_pos, tile_name))
                
                current_pos += 1
            
            # Affichage des échecs
            if failed_indices:
                print(f"⚠️ {len(failed_indices)} patches {mode} non trouvés (sur {k} demandés)")
                for pos, tile_name in failed_indices[:5]:  # Limite à 5 pour la lisibilité
                    print(f"   - Position {pos}: {tile_name}")
                if len(failed_indices) > 5:
                    print(f"   - ... et {len(failed_indices) - 5} autres")
            
            return copied_count
        
        # Copie des TOP patches (scores les plus élevés)
        print(f"📁 Recherche de {top_k} top patches pour {suffix if suffix else 'risk'}...")
        top_copied = copy_patches_with_fallback(
            sorted_indices_desc, top_dir, top_k, mode="top"
        )
        print(f"✅ {top_copied}/{top_k} top patches sauvegardés dans {top_dir}")
        
        # Copie des BOTTOM patches (scores les plus bas)
        print(f"📁 Recherche de {bottom_k} bottom patches pour {suffix if suffix else 'risk'}...")
        bottom_copied = copy_patches_with_fallback(
            sorted_indices_asc, bottom_dir, bottom_k, mode="bottom"
        )
        print(f"✅ {bottom_copied}/{bottom_k} bottom patches sauvegardés dans {bottom_dir}")
        
        # Stockage des résultats (tous les indices, pas seulement ceux copiés)
        top_indices = sorted_indices_desc[:top_k].tolist()
        bottom_indices = sorted_indices_asc[:bottom_k].tolist()
        
        results[f'top_indices{suffix}'] = top_indices
        results[f'bottom_indices{suffix}'] = bottom_indices
        results[f'top_scores{suffix}'] = [float(scores[i]) for i in top_indices]
        results[f'bottom_scores{suffix}'] = [float(scores[i]) for i in bottom_indices]
        results[f'top_copied{suffix}'] = top_copied
        results[f'bottom_copied{suffix}'] = bottom_copied
    
    return results






# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Liste des slides à traiter
    slide_list = [9, 51, 98, 109, 132]  # risk class: High=98, Intermediate=109, Low=132
    
    # Vérification des paramètres
    if TASK == "risk" and not MODEL_WEIGHTS:
        MODEL_WEIGHTS = "MIL/model_weights/final_model_abmil_stratified.pth"
    elif TASK == "necfib" and not MODEL_WEIGHTS:
        MODEL_WEIGHTS = "MIL/model_weights/final_model_abmil_stratified.pth"
    
    for slide_num in slide_list:
        print(f"\n{'#'*60}")
        print(f"📊 Traitement de la slide {slide_num}...")
        print(f"{'#'*60}")
        
        try:
            if TASK == "risk":
                # Génération de la heatmap
                heatmap_out, attn, logits = generate_heatmap_risk(
                    slide_num=slide_num,
                    pth_path=MODEL_WEIGHTS,
                    model_class=AttentionMIL,
                    device=device,
                    h5_dir=H5_DIR,
                    img_dir="data/wsi_raw/training_png",
                    output_dir=OUTPUT_DIR,
                    score_threshold=SCORE_THRESHOLD,
                    cmap=CMAP,
                    alpha=ALPHA,
                    scale=SCALE,
                    use_all_patches=True,
                    model_name=MODEL_NAME
                )
                
                # Extraction des top/bottom patches
                print(f"\n📊 Extraction des top/bottom patches pour slide {slide_num}...")
                model = AttentionMIL(num_classes=3, in_dim=1536).to(device)
                model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
                patches_info = extract_top_bottom_patches_risk(
                    slide_num=slide_num,
                    model=model,
                    device=device,
                    h5_dir=H5_DIR,
                    png_dir="data/wsi_raw/tiles_png",
                    output_dir=OUTPUT_DIR,
                    top_k=TOP_K,
                    bottom_k=BOTTOM_K,
                    model_name=MODEL_NAME
                )
                
            elif TASK == "necfib":
                # Génération de la heatmap
                heatmap_out, attn, logits = generate_heatmap_necfib(
                    slide_num=slide_num,
                    pth_path=MODEL_WEIGHTS,
                    model_class=AttentionMILNecFib,
                    device=device,
                    h5_dir=H5_DIR,
                    img_dir="data/wsi_raw/training_png",
                    output_dir=OUTPUT_DIR,
                    score_threshold=SCORE_THRESHOLD,
                    cmap=CMAP,
                    alpha=ALPHA,
                    scale=SCALE,
                    use_all_patches=True,
                    model_name=MODEL_NAME
                )
                
                # Extraction des top/bottom patches
                print(f"\n📊 Extraction des top/bottom patches pour slide {slide_num}...")
                model = AttentionMILNecFib().to(device)
                model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
                patches_info = extract_top_bottom_patches_necfib(
                    slide_num=slide_num,
                    model=model,
                    device=device,
                    h5_dir=H5_DIR,
                    png_dir="data/wsi_raw/tiles_png",
                    output_dir=OUTPUT_DIR,
                    top_k=TOP_K,
                    bottom_k=BOTTOM_K,
                    model_name=MODEL_NAME
                )
            
            print(f"✅ Slide {slide_num} traitée avec succès !")
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de la slide {slide_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("✅ Terminé !")
    print("="*60)