"""Dataset utilities for TCGA-BRCA whole-slide image embeddings.

This module loads UNI2 patch embeddings from HDF5 files and returns
one bag of embeddings per slide, together with its label and case ID.
Slides with too few patches are filtered out during batch collation.
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


THRESHOLD_BAG = 5000
MIN_PATCHES = 100
H5_DIR = "UNI2-h_features/TCGA"


class TCGA_BRCA_BagDataset(Dataset):
    """Dataset of TCGA-BRCA slides represented as bags of patch embeddings.

    Each item corresponds to one slide and contains its patch embeddings,
    binary label, and case ID.

    Args:
        filename_map: Mapping from case IDs to HDF5 filenames.
        targets: Mapping from case IDs to integer labels (0 or 1).
        h5_dir: Directory containing the HDF5 files.
        feature_key: Name of the HDF5 dataset containing the embeddings.
            Defaults to "features".
    """

    def __init__(self, slides, h5_dir, is_train):
        self.h5_dir = h5_dir
        self.slides = slides
        self.rng = np.random.default_rng(42)
        self.is_train = is_train
    
    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        """Load a slide's embeddings and label.

        Slides with fewer than MIN_PATCHES patches return None and are
        subsequently removed by the custom collate function. Slides with
        more than THRESHOLD_BAG patches are randomly subsampled.
        """
        slide = self.slides[idx]
        case_id = slide["id"]
        path_h5_file = os.path.join(self.h5_dir, slide["filename"])
        label = int(slide["target"])

        with h5py.File(path_h5_file, "r") as h5_file:
            # Convert embeddings from (1, N, D) to (N, D), if necessary.
            embeddings = h5_file["features"][:]

            if embeddings.ndim == 3:
                embeddings = embeddings[0]

        # Exclude slides containing too few patches.
        if len(embeddings) < MIN_PATCHES:
            return None

        # Randomly subsample slides containing too many patches.
        if self.is_train and len(embeddings) > THRESHOLD_BAG:
            selected_indices = self.rng.choice(
                len(embeddings),
                THRESHOLD_BAG,
                replace=False,
            )
            embeddings = embeddings[selected_indices]

        embeddings = torch.from_numpy(embeddings).float()
        label = torch.tensor(label, dtype=torch.long)

        return embeddings, label, case_id


def collate_bags(batch):
    """Filter invalid slides and collate a batch of variable-sized bags.

    Args:
        batch: List of dataset items. Each valid item contains embeddings,
            a label, and a case ID.

    Returns:
        None if every item in the batch is invalid. Otherwise, returns a
        tuple containing the list of embedding tensors, stacked labels,
        and list of case IDs.
    """
    # Remove slides that do not meet the minimum patch requirement.
    batch = [item for item in batch if item is not None]

    if not batch:
        return None

    # Keep bags as a list because each slide can contain a different
    # number of patches.
    embeddings_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    case_ids = [item[2] for item in batch]

    return embeddings_list, labels, case_ids