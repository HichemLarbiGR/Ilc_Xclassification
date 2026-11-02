"""Tableau des prédictions par lame et journal des erreurs.
 
À coller dans utils/utils.py (ou à importer tel quel). Le bloc commenté en bas
du fichier remplace la section 6 de main().
"""
import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
def build_predictions_df(
    splits: Dict[str, Sequence[Tuple]],
    class_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    frames = []
    for split_name, rows in splits.items():
        ids = [r[0] for r in rows]
        y = np.asarray([r[1] for r in rows])
        pred = np.asarray([r[2] for r in rows])
        prob = np.asarray([r[3] for r in rows])

        if not (len(ids) == len(y) == len(pred) == len(prob)):
            raise ValueError(
                f"{split_name} : longueurs incohérentes "
                f"(ids={len(ids)}, y={len(y)}, pred={len(pred)}, prob={len(prob)})"
            )

        frame = pd.DataFrame(
            {"Slide_ID": ids, "Dataset_Split": split_name, "True_Label": y, "Pred_Label": pred}
        )
        prob_cols = [f"P_{k}" for k in range(prob.shape[1])]
        frame[prob_cols] = prob
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    duplicated = df.loc[df["Slide_ID"].duplicated(keep=False), "Slide_ID"].unique()
    if len(duplicated):
        raise ValueError(
            f"{len(duplicated)} Slide_ID en double (fuite possible entre jeux) : {list(duplicated[:10])}"
        )

    prob_cols = [c for c in df.columns if c.startswith("P_")]
    if class_names is not None:
        names = dict(enumerate(class_names))
        df["True_Class"] = df["True_Label"].map(names)
        df["Pred_Class"] = df["Pred_Label"].map(names)
    df["Is_Error"] = df["True_Label"] != df["Pred_Label"]
    df["Confiance_Modele"] = df[prob_cols].max(axis=1)
    return df