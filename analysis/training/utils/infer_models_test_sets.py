import torch
import torch.nn as nn
import numpy as np
import itertools
import logging
import json
import math
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, 
    precision_score, recall_score, roc_curve, auc, roc_auc_score
)
from collections import Counter

# Imports de tes modules
from model import AttentionMIL, AttentionMILNecFib_v2, EarlyStopping
from data_loader_bag import SlideBagDatasetOrdinal, SlideBagNecFibDataset, collate_bags_ordinal, collate_necfib_bags

# Import de tes fonctions natives
from training_risks_GLM import evaluate_model_tta, get_metrics, SEED
from training_necfib_GLM import evaluate_model, _compute_binary_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  # Pour PyTorch

# ============================================================
# 1. PARAMÈTRES INTERNES
# ============================================================
# 👉 CHANGE CETTE VARIABLE POUR CHOISIR LA TÂCHE
TASK = "necfib"  # Options: "necfib" ou "risk"

class FakeConfig:
    def __init__(self, task):
        #/home/userx/EGFR_Necrosis_20260825/prepros/data/clinical_data/split_genova.json
        self.global_split_path = "/home/userx/EGFR_Necrosis_20260825/prepros/data/clinical_data/split_genova.json"
        self.h5_dir = "/home/userx/EGFR_Necrosis_20260825/prepros/h5_files"
        self.output_dir = "/home/userx/EGFR_Necrosis_20260825/analysis/results"
        
        self.model_name = "hoptimus1_30p"
        self.input_dim = 1536
        self.label_smoothing = 0.1
        self.batch_size = 1
        
        if task == "necfib":
            self.num_classes = 2
            self.path_clinical_scores = "/home/userx/EGFR_Necrosis_20260825/prepros/data/clinical_data/slide_scores_necfib_binary_all_sets.json"
            self.save_model_path = "/home/userx/EGFR_Necrosis_20260825/analysis/MIL/model_weights/necfib/model_abmil_necfib_hpt1_30p_20260831.pth"
        else:
            self.num_classes = 3
            self.path_clinical_scores = "/home/userx/EGFR_Necrosis_20260825/prepros/data/clinical_data/slide_scores_risks_all_sets.json"
            self.save_model_path = "/home/userx/EGFR_Necrosis_20260825/analysis/MIL/model_weights/risks/model_abmil_risks_hpt1_30p_20260831.pth"
CONFIG = FakeConfig(TASK)
N_DRAWS_TTA = 5

# ============================================================
# 2. FONCTIONS DE VISUALISATION & BOOTSTRAP
# ============================================================
def plot_confusion_matrix(cm, classes, title, save_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close()
    logger.info(f"💾 Image sauvegardée : {save_path}")

def plot_roc_curve(test_labels, test_probs, num_classes, class_names, save_path, set_name="Test"):
    plt.figure(figsize=(8, 6))
    
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(test_labels, test_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC {class_names[1]} (AUC = {roc_auc:0.2f})')
    else:
        y_test = label_binarize(test_labels, classes=list(range(num_classes)))
        colors = ['blue', 'green', 'red']
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_test[:, i], test_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'ROC {class_names[i]} (AUC = {roc_auc:0.2f})')
        
        fpr_micro, tpr_micro, _ = roc_curve(y_test.ravel(), test_probs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', linewidth=4, label=f'Micro-averaged ROC (AUC = {roc_auc_micro:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC curve ({set_name} - {TASK} {CONFIG.model_name})')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close()
    logger.info(f"💾 Image sauvegardée : {save_path}")

def compute_full_bootstrap_ci(y_true, y_pred, y_probs, num_classes, n_iterations=2000):
    """Calcule l'IC 95% pour TOUTES les métriques via Bootstrap."""
    metrics_to_boot = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro', 'auc']
    boot_scores = {m: [] for m in metrics_to_boot}
    n_samples = len(y_true)
    
    if n_samples == 0:
        return {m: (np.nan, np.nan, np.nan) for m in metrics_to_boot}

    for _ in range(n_iterations):
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        y_true_bs = y_true[indices]
        y_pred_bs = y_pred[indices]
        
        if len(np.unique(y_true_bs)) < 2:
            for m in metrics_to_boot:
                boot_scores[m].append(np.nan)
            continue
            
        boot_scores['accuracy'].append(accuracy_score(y_true_bs, y_pred_bs))
        boot_scores['f1_macro'].append(f1_score(y_true_bs, y_pred_bs, average='macro', zero_division=0))
        boot_scores['f1_weighted'].append(f1_score(y_true_bs, y_pred_bs, average='weighted', zero_division=0))
        boot_scores['precision_macro'].append(precision_score(y_true_bs, y_pred_bs, average='macro', zero_division=0))
        boot_scores['recall_macro'].append(recall_score(y_true_bs, y_pred_bs, average='macro', zero_division=0))
        
        if y_probs is not None:
            y_probs_bs = y_probs[indices]
            try:
                if num_classes == 2:
                    boot_scores['auc'].append(roc_auc_score(y_true_bs, y_probs_bs[:, 1]))
                else:
                    y_true_bs_bin = label_binarize(y_true_bs, classes=list(range(num_classes)))
                    boot_scores['auc'].append(roc_auc_score(y_true_bs_bin, y_probs_bs, multi_class='ovr', average='macro'))
            except ValueError:
                boot_scores['auc'].append(np.nan)
        else:
            boot_scores['auc'].append(np.nan)
            
    ci_results = {}
    for m in metrics_to_boot:
        valid_scores = [s for s in boot_scores[m] if not np.isnan(s)]
        if valid_scores:
            mean = np.mean(valid_scores)
            low = np.percentile(valid_scores, 2.5)
            high = np.percentile(valid_scores, 97.5)
            ci_results[m] = (mean, low, high)
        else:
            ci_results[m] = (np.nan, np.nan, np.nan)
            
    return ci_results

def log_metrics(metrics_dict, ci_dict=None):
    """Affiche proprement les métriques et leurs IC dans les logs."""
    metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted', 'auc']
    for name in metric_names:
        val = metrics_dict.get(name, float('nan'))
        if ci_dict and name in ci_dict:
            mean, low, high = ci_dict[name]
            logger.info(f"  {name.replace('_', ' ').capitalize():15} : {val:.4f}  [IC 95%: {low:.4f} - {high:.4f}]")
        else:
            logger.info(f"  {name.replace('_', ' ').capitalize():15} : {val:.4f}")

def save_full_predictions_report(df, task, model_name):
    """Sauvegarde l'ensemble des prédictions (Train/Val et Test) dans un fichier CSV."""
    # On trie d'abord par Split (Train/Val ensemble, Test ensemble), puis par Slide_ID
    df_sorted = df.sort_values(by=['Dataset_Split', 'Slide_ID']).reset_index(drop=True)
    
    csv_path = f"/home/svc_hpc_services/work/20251227_EGFR_necrosis/necrosis_quanti/MIL/output/previsions_completes_{task}_{model_name}.csv"
    df_sorted.to_csv(csv_path, index=False)
    logger.info(f"📊 Tableau complet des prédictions sauvegardé : {csv_path}")
    return csv_path

# 🆕 Évaluation complète d'un groupe de slides : métriques + bootstrap + figures
def evaluate_and_report_group(group_name, y, p, pr, num_classes_eval, classes,
                              task_name, output_dir, do_plots=True):
    if len(y) == 0:
        logger.warning(f"⚠️ Groupe '{group_name}' vide ({task_name}) — ignoré")
        return None
    logger.info(f"\n--- {group_name.upper()} ({len(y)} lames) ---")
    if num_classes_eval == 2:
        metrics_g = _compute_binary_metrics(y, p, pr)
    else:
        metrics_g = get_metrics(y, p, pr, num_classes_eval)
    ci_g = compute_full_bootstrap_ci(y, p, pr, num_classes_eval)
    log_metrics(metrics_g, ci_g)
    if do_plots:
        suffix = f"{group_name}_{task_name}" if task_name else group_name
        plot_confusion_matrix(metrics_g['confusion_matrix'], classes,
                              f'Confusion matrix ({group_name} - {task_name} {CONFIG.model_name})',
                              os.path.join(output_dir, f"cm_{suffix}_{CONFIG.model_name}.svg"))
        plot_roc_curve(y, pr, num_classes_eval, classes,
                       os.path.join(output_dir, f"roc_curve_{suffix}_{CONFIG.model_name}.svg"),
                       set_name=group_name)
    return metrics_g

# ============================================================
# 3. EXÉCUTION PRINCIPALE
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. CHARGEMENT DU SPLIT GLOBAL FIGÉ ---
    try:
        with open(CONFIG.global_split_path, 'r') as f:
            global_split = json.load(f)
        logger.info(f"📂 Split global chargé depuis : {CONFIG.global_split_path}")
    except FileNotFoundError:
        logger.error(f"❌ Fichier de split global introuvable : {CONFIG.global_split_path}. Lancez create_global_split.py d'abord !")
        sys.exit(1)

    # Vérification
    print("Clés disponibles :", list(global_split.keys()))
    print("Contenu complet :", global_split)

    # 🆕 Les 3 sets de test + leur union
    test_set = set(global_split['test'])
    test_genova = set(global_split['test_genova'])
    test_500 = set(global_split['test_500'])
    all_test_ids = test_set | test_genova | test_500

    # 🆕 Vérification des chevauchements entre les 3 sets de test
    for name_a, name_b, set_a, set_b in [
        ("test", "test_genova", test_set, test_genova),
        ("test", "test_500", test_set, test_500),
        ("test_genova", "test_500", test_genova, test_500),
    ]:
        overlap = set_a & set_b
        if overlap:
            logger.warning(f"⚠️ Chevauchement {name_a} ∩ {name_b} : {sorted(overlap)} "
                           f"(ces slides seront comptées dans chaque set ET dans l'union)")

    # --- 2. CHARGEMENT ET NETTOYAGE DES LABELS ---
    with open(CONFIG.path_clinical_scores, 'r') as f:
        slide_labels_raw = json.load(f)
        
    if TASK == "necfib":
        all_slide_labels = {int(k): v for k, v in slide_labels_raw.items() if isinstance(v, (list, tuple)) and len(v) == 2}
    else:
        all_slide_labels = {int(k): int(v[0] if isinstance(v, list) else v) for k, v in slide_labels_raw.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    
    # --- 3. FILTRAGE DES LABELS PAR RAPPORT AU SPLIT FIGÉ ---
    # 🆕 train_val = slides du split SANS les slides des 3 sets de test
    train_val_slides = [sid for sid in global_split['train_val']
                        if sid in all_slide_labels and sid not in all_test_ids]

    # 🆕 Le dataset d'évaluation inclut maintenant les slides des 3 sets de test
    test_slides = [sid for sid in global_split['test'] if sid in all_slide_labels]
    test_genova_slides = [sid for sid in global_split['test_genova'] if sid in all_slide_labels]
    test_500_slides = [sid for sid in global_split['test_500'] if sid in all_slide_labels]

    train_val_labels = {s: all_slide_labels[s] for s in train_val_slides}
    test_labels_dict = {s: all_slide_labels[s]
                        for s in set(test_slides) | set(test_genova_slides) | set(test_500_slides)}

    logger.info(f"📊 Slides avec labels — train_val: {len(train_val_labels)} | "
                f"test: {len(test_slides)} | test_genova: {len(test_genova_slides)} | "
                f"test_500: {len(test_500_slides)} | union: {len(test_labels_dict)}")
    # 🆕 Alerte si des slides d'un set de test n'ont pas de label / pas de h5
    for name, ids in [("test", global_split['test']),
                      ("test_genova", global_split['test_genova']),
                      ("test_500", global_split['test_500'])]:
        missing = [s for s in ids if s not in all_slide_labels]
        if missing:
            logger.warning(f"⚠️ {name} : {len(missing)} slides sans label utilisable : {missing}")

    # --- 4. CRÉATION DU DATASET GLOBAL POUR L'EVALUATION ---
    eval_labels = {**train_val_labels, **test_labels_dict}
    
    logger.info(f"\nCréation du dataset global ({len(eval_labels)} lames)...")
    if TASK == "necfib":
        global_dataset = SlideBagNecFibDataset(CONFIG.h5_dir, eval_labels, model_name=CONFIG.model_name, is_train=False)
        global_loader = DataLoader(global_dataset, batch_size=1, collate_fn=collate_necfib_bags, shuffle=False)
    else:
        global_dataset = SlideBagDatasetOrdinal(CONFIG.h5_dir, eval_labels, model_name=CONFIG.model_name, is_train=False)

    # --- 5. CHARGEMENT DU MODÈLE ---
    if TASK == "necfib":
        model = AttentionMILNecFib_v2().to(device)
    else:
        model = AttentionMIL(num_classes=CONFIG.num_classes, in_dim=CONFIG.input_dim).to(device)
    
    model.load_state_dict(torch.load(CONFIG.save_model_path, map_location=device))
    model.eval()
    logger.info(f"✅ Modèle chargé : {CONFIG.save_model_path}")
    
    # --- 6. ÉVALUATION (1 SEULE FOIS) ---
    logger.info(f"🔄 Évaluation sur TOUT le dataset ({len(global_dataset)} lames)...")
    if TASK == "necfib":
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG.label_smoothing)
        _, _, _, all_labels_nec, all_preds_nec, all_probs_nec, all_labels_fib, all_preds_fib, all_probs_fib = evaluate_model(model, global_loader, criterion, device)
    else:
        _, _, _, all_labels, all_preds, all_probs = evaluate_model_tta(model, global_dataset, device, CONFIG, n_draws=N_DRAWS_TTA)
    
    # --- 7. MASQUES POUR LES GROUPES D'ÉVALUATION ---
    slides_arr = np.array(global_dataset.slides)
    
    # 🆕 Étiquette de split par slide, avec priorité en cas de chevauchement :
    #    test > test_genova > test_500
    split_labels = []
    for s in slides_arr:
        s = int(s)
        if s in test_set:
            split_labels.append("test")
        elif s in test_genova:
            split_labels.append("test_genova")
        elif s in test_500:
            split_labels.append("test_500")
        else:
            split_labels.append("Train/Val")
    split_labels = np.array(split_labels)

    # 🆕 Masques des 4 groupes de test (les 3 sets + le mélange/union)
    EVAL_GROUPS = {
        "test":        test_set,
        "test_genova": test_genova,
        "test_500":    test_500,
        "test_all":    all_test_ids,   # mélange des 3 sets
    }
    group_masks = {name: np.array([int(s) in ids for s in slides_arr])
                   for name, ids in EVAL_GROUPS.items()}
    for name, m in group_masks.items():
        logger.info(f"   Groupe {name}: {m.sum()} lames dans le dataset chargé")

    # Compatibilité avec le reste du script : Test = union des 3 sets
    is_test_mask = split_labels != "Train/Val"
    
    # --- 8. CALCUL DES MÉTRIQUES, BOOTSTRAP ET AFFICHAGE ---
    if TASK == "necfib":
        num_classes_eval = 2
        for task_name, labels, preds, probs in [
            ("Necrosis", all_labels_nec, all_preds_nec, all_probs_nec), 
            ("Fibrosis", all_labels_fib, all_preds_fib, all_probs_fib)
        ]:
            logger.info("\n" + "="*60 + f"\n📈 MÉTRIQUES & BOOTSTRAP - {task_name}\n" + "="*60)
            
            # Train/Val (inchangé, pas de bootstrap ni de figures)
            y_train = labels[~is_test_mask]
            p_train = preds[~is_test_mask]
            pr_train = probs[~is_test_mask]
            
            logger.info(f"\n--- TRAIN+VAL ({len(y_train)} lames) ---")
            metrics_train = _compute_binary_metrics(y_train, p_train, pr_train)
            log_metrics(metrics_train)
            
            # 🆕 Boucle sur les 4 groupes de test (3 sets + union)
            for group_name, mask in group_masks.items():
                evaluate_and_report_group(group_name, labels[mask], preds[mask], probs[mask],
                                          num_classes_eval, ['Classe 0', 'Classe 1'],
                                          task_name, CONFIG.output_dir)

    else: # TASK == "risk"
        num_classes_eval = 3
        logger.info("\n" + "="*60 + "\n📈 MÉTRIQUES & BOOTSTRAP - RISKS\n" + "="*60)
        
        y_train = all_labels[~is_test_mask]
        p_train = all_preds[~is_test_mask]
        pr_train = all_probs[~is_test_mask]
        
        logger.info(f"\n--- TRAIN+VAL ({len(y_train)} lames) ---")
        metrics_train = get_metrics(y_train, p_train, pr_train, num_classes_eval)
        log_metrics(metrics_train)
        
        # 🆕 Boucle sur les 4 groupes de test (3 sets + union)
        for group_name, mask in group_masks.items():
            evaluate_and_report_group(group_name, all_labels[mask], all_preds[mask], all_probs[mask],
                                      num_classes_eval, ['Low(0)', 'Int(1)', 'High(2)'],
                                      "Risks", CONFIG.output_dir)

    # --- 9. SAUVEGARDE DES PRÉVISIONS COMPLÈTES ET DU RAPPORT D'ERREURS ---
    logger.info("📊 Création des rapports globaux...")
        
    # Création du dossier output s'il n'existe pas (sécurité supplémentaire)

    os.makedirs(CONFIG.output_dir, exist_ok=True)
        
    if TASK == "necfib":
        # Sécurisation : on convertit en array numpy
        all_labels_nec = np.array(all_labels_nec)
        all_preds_nec = np.array(all_preds_nec)
        all_probs_nec = np.array(all_probs_nec)
        
        all_labels_fib = np.array(all_labels_fib)
        all_preds_fib = np.array(all_preds_fib)
        all_probs_fib = np.array(all_probs_fib)

        # Gestion du cas où all_probs_nec est à 1 dimension (ex: shape (N,) au lieu de (N, 2))
        if all_probs_nec.ndim == 1:
            P_1_nec = all_probs_nec
            P_0_nec = 1.0 - P_1_nec
        else:
            P_0_nec = all_probs_nec[:, 0]
            P_1_nec = all_probs_nec[:, 1]

        if all_probs_fib.ndim == 1:
            P_1_fib = all_probs_fib
            P_0_fib = 1.0 - P_1_fib
        else:
            P_0_fib = all_probs_fib[:, 0]
            P_1_fib = all_probs_fib[:, 1]

        df_nec = pd.DataFrame({
            'Slide_ID': slides_arr,
            'Dataset_Split': ['Test' if x else 'Train/Val' for x in is_test_mask],
            'Test_Subset': split_labels,          # 🆕 test / test_genova / test_500 / Train/Val
            'True_Label': all_labels_nec,
            'Pred_Label': all_preds_nec,
            'P_0': P_0_nec,
            'P_1': P_1_nec,
            'Type': "Nec"
        })
        df_fib = pd.DataFrame({
            'Slide_ID': slides_arr,
            'Dataset_Split': ['Test' if x else 'Train/Val' for x in is_test_mask],
            'Test_Subset': split_labels,          # 🆕
            'True_Label': all_labels_fib,
            'Pred_Label': all_preds_fib,
            'P_0': P_0_fib,
            'P_1': P_1_fib,
            'Type': "Fib"
        })
        df = pd.concat([df_nec, df_fib], ignore_index=True)
        prob_cols = ['P_0', 'P_1']
        
    else:
        # Sécurisation pour "risk"
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        df = pd.DataFrame({
            'Slide_ID': slides_arr,
            'Dataset_Split': ['Test' if x else 'Train/Val' for x in is_test_mask],
            'Test_Subset': split_labels,          # 🆕 test / test_genova / test_500 / Train/Val
            'True_Label': all_labels,
            'Pred_Label': all_preds,
            'P_0': all_probs[:, 0],
            'P_1': all_probs[:, 1],
            'P_2': all_probs[:, 2],
            'Type': "Risk"
        })
        prob_cols = ['P_0', 'P_1', 'P_2']
    
    df['Is_Error'] = df['True_Label'] != df['Pred_Label']
    df['Confiance_Modele'] = df[prob_cols].max(axis=1)
    
    # ✅ Sauvegarde du tableau complet (Train + Test)
    df_sorted = df.sort_values(by=['Dataset_Split', 'Slide_ID']).reset_index(drop=True)
    full_csv_path = os.path.join(CONFIG.output_dir, f"previsions_completes_{TASK}_{CONFIG.model_name}.csv")
    df_sorted.to_csv(full_csv_path, index=False)
    logger.info(f"📊 Tableau complet des prédictions sauvegardé : {full_csv_path}")
    
    # Sauvegarde du rapport d'erreurs
    errors_only = df[df['Is_Error']].sort_values(by='Confiance_Modele', ascending=False)
    errors_csv_path = os.path.join(CONFIG.output_dir, f"autopsie_erreurs_global_{TASK}_{CONFIG.model_name}.csv")
    errors_only.to_csv(errors_csv_path, index=False)
    
    logger.info(f"✅ Rapport d'erreurs sauvegardé dans : {errors_csv_path}")
    logger.info(f"Nombre total d'erreurs : {len(errors_only)} sur {len(df)} lignes évaluées.")
    
    logger.info("\nTop 15 des erreurs les plus 'confidentes' (à vérifier cliniquement) :")
    for index, row in errors_only.head(15).iterrows():
        probas_str = ", ".join([f"{col}:{row[col]:.2f}" for col in prob_cols])
        logger.info(f"  Slide {int(row['Slide_ID'])} [{row['Test_Subset']}] [{row['Type']}] | Vrai: {int(row['True_Label'])} -> Prédit: {int(row['Pred_Label'])} | Probas: [{probas_str}]")