"""Attention-based explainability for ILC vs NST whole-slide classification.

For each slide of a given split, this script:

1. loads the pre-extracted patch embeddings (TRIDENT HDF5 files) and runs the
   trained gated-attention MIL model;
2. renders a vector attention heatmap over the 1/``scale_factor`` PNG
   thumbnail of the whole-slide image (WSI);
3. extracts the highest- and lowest-attention patches directly from the
   original WSI with OpenSlide, using the coordinates stored in the HDF5.

Coordinate conventions
----------------------
TRIDENT stores, for each patch, the (x, y) position of its **top-left corner**
in the level-0 reference frame of the WSI. The ``patch_size_level0`` attribute
of the ``coords`` dataset gives the patch edge length at level 0 (e.g. 512 px
for a 256 px patch at 20x on a 40x slide), and ``patch_size`` gives the edge
length of the patch actually fed to the foundation model (256 px).

On a thumbnail downsampled by ``scale_factor``, a patch is therefore a square
of ``patch_size_level0 / scale_factor`` pixels anchored at
``(x / scale_factor, y / scale_factor)``.

Expected directory layout (all overridable from the command line)::

    <h5_dir>/                      HDF5 feature files (searched recursively)
    <h5_dir>/data/wsi/             <stem>.svs
    <h5_dir>/data/png_wsi/         <stem>_1_<scale_factor>.png

Usage
-----
    python xai_ilc.py --config config.yaml
    python xai_ilc.py --config config.yaml --slides TCGA-GM-A2DO TCGA-AR-A1AO
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")  # Headless backend: figures are only written to disk.

import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
import yaml
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image

from models.MIL_models import AttentionMIL

logger = logging.getLogger("xai_ilc")

# Thumbnails of large WSIs can exceed Pillow's decompression-bomb threshold.
# The inputs are trusted local files, so the check is disabled.
Image.MAX_IMAGE_PIXELS = None


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass(frozen=True)
class SlideRecord:
    """Entry of the split JSON file.

    Attributes
    ----------
    slide_id : str
        Case-level identifier (e.g. ``"TCGA-GM-A2DO"``).
    filename : str
        Name of the HDF5 feature file.
    target : int
        Ground-truth class index.
    """

    slide_id: str
    filename: str
    target: int

    @property
    def stem(self) -> str:
        """Full slide name without extension (shared by .h5, .svs and .png)."""
        return Path(self.filename).stem


@dataclass(frozen=True)
class SlideFeatures:
    """Content of a TRIDENT HDF5 file.

    Attributes
    ----------
    features : np.ndarray
        Patch embeddings of shape (n_patches, embedding_dim).
    coords : np.ndarray
        Top-left patch coordinates at level 0, shape (n_patches, 2), as (x, y).
    patch_size_level0 : int
        Patch edge length at level 0, in pixels.
    patch_size : int
        Patch edge length at the target magnification, in pixels.
    """

    features: np.ndarray
    coords: np.ndarray
    patch_size_level0: int
    patch_size: int


@dataclass(frozen=True)
class XAIConfig:
    """Resolved configuration of the explainability run."""

    h5_dir: Path
    wsi_dir: Path
    png_dir: Path
    output_dir: Path
    split_json: Path
    weights: Path
    subset: str
    input_dim: int
    num_classes: int
    class_names: list[str]
    scale_factor: int
    top_k: int
    bottom_k: int
    score_threshold: float
    cmap: str
    alpha: float
    figure_ext: str


# ============================================================
# CONFIGURATION AND INPUT RESOLUTION
# ============================================================


def load_config(config_path: Path, args: argparse.Namespace) -> XAIConfig:
    """Build the run configuration from the YAML file and CLI overrides.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.
    args : argparse.Namespace
        Parsed command-line arguments; non-null values override the YAML.

    Returns
    -------
    XAIConfig
        Fully resolved configuration.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    paths, model, evaluation, xai = (
        raw[key] for key in ("paths", "model", "evaluation", "xai")
    )

    h5_dir = Path(paths["h5_dir"])
    subset = args.subset or evaluation["subset_name"]

    return XAIConfig(
        h5_dir=h5_dir,
        wsi_dir=Path(args.wsi_dir) if args.wsi_dir else h5_dir / "data" / "wsi",
        png_dir=Path(args.png_dir) if args.png_dir else h5_dir / "data" / "png_wsi",
        output_dir=(
            Path(args.output_dir)
            if args.output_dir
            else Path(paths["path_results"]) / "xai" / subset
        ),
        split_json=Path(paths["json_split_wsi_target_path"]),
        weights=Path(args.weights or evaluation["infer_model"]),
        subset=subset,
        input_dim=int(model["input_dim"]),
        num_classes=int(model["num_classes"]),
        class_names=list(model["classes_name"]),
        scale_factor=int(xai["scale_factor"]),
        top_k=int(xai["top_k"]),
        bottom_k=int(xai["bottom_k"]),
        score_threshold=float(xai["score_threshold"]),
        cmap=str(xai["cmap"]),
        alpha=float(xai["alpha"]),
        figure_ext=str(evaluation.get("figures_extension", "svg")).lstrip("."),
    )


def load_slide_records(json_path: Path, subset: str) -> list[SlideRecord]:
    """Read the slide list from the split JSON file.

    The file may either be a flat list of entries or a dictionary mapping
    split names to lists of entries (in which case ``subset`` selects one).

    Parameters
    ----------
    json_path : Path
        Path to the JSON split file.
    subset : str
        Split name, used only if the JSON root is a dictionary.

    Returns
    -------
    list[SlideRecord]
        Slide entries of the requested split.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if subset not in payload:
            raise KeyError(
                f"Subset '{subset}' not found in {json_path}; "
                f"available keys: {list(payload)}."
            )
        entries = payload[subset]
    else:
        logger.info("Split JSON is a flat list: 'subset' is ignored.")
        entries = payload

    return [
        SlideRecord(
            slide_id=entry["id"],
            filename=entry["filename"],
            target=int(entry["target"]),
        )
        for entry in entries
    ]


def select_slides(
    records: list[SlideRecord], requested: list[str] | None
) -> list[SlideRecord]:
    """Filter slide records by identifier, file name or file stem.

    Parameters
    ----------
    records : list[SlideRecord]
        All slides of the split.
    requested : list[str] or None
        Requested identifiers. If empty or None, all records are returned.

    Returns
    -------
    list[SlideRecord]
        Selected slides.

    Raises
    ------
    ValueError
        If a requested identifier matches no slide.
    """
    if not requested:
        return records

    wanted = set(requested)
    selected = [r for r in records if wanted & {r.slide_id, r.filename, r.stem}]

    found = {key for r in selected for key in (r.slide_id, r.filename, r.stem)}
    missing = wanted - found
    if missing:
        raise ValueError(f"Slides not found in the selected subset: {sorted(missing)}")

    return selected


def find_file(root: Path, name: str) -> Path:
    """Locate a file by name, first directly in ``root`` and then recursively.

    Parameters
    ----------
    root : Path
        Directory to search.
    name : str
        File name to look for.

    Returns
    -------
    Path
        Path of the file.

    Raises
    ------
    FileNotFoundError
        If the file cannot be found under ``root``.
    """
    direct = root / name
    if direct.is_file():
        return direct

    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"'{name}' not found under {root}.")
    if len(matches) > 1:
        logger.warning("%d files named '%s' under %s; using %s.", len(matches), name, root, matches[0])

    return matches[0]


# ============================================================
# MODEL AND INFERENCE
# ============================================================


def load_model(
    weights: Path, input_dim: int, num_classes: int, device: torch.device
) -> AttentionMIL:
    """Instantiate the MIL model and load trained weights.

    Parameters
    ----------
    weights : Path
        Path to a ``state_dict`` saved with ``torch.save``.
    input_dim : int
        Patch embedding dimension.
    num_classes : int
        Number of output classes.
    device : torch.device
        Target device.

    Returns
    -------
    AttentionMIL
        Model in evaluation mode.
    """
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    model = AttentionMIL(num_classes=num_classes, in_dim=input_dim).to(device)
    state_dict = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Model loaded from %s.", weights)
    return model


def read_h5(h5_path: Path) -> SlideFeatures:
    """Read embeddings, coordinates and patching metadata from a TRIDENT file.

    The leading batch dimension of the stored arrays, e.g. (1, N, 1536), is
    removed.

    Parameters
    ----------
    h5_path : Path
        Path to the HDF5 file.

    Returns
    -------
    SlideFeatures
        Embeddings, level-0 coordinates and patch sizes.
    """
    with h5py.File(h5_path, "r") as f:
        features = f["features"][()].astype(np.float32, copy=False)

        coords_key = "coords" if "coords" in f else "coords_patching"
        coords = f[coords_key][()]
        attrs = dict(f[coords_key].attrs)

    features = features.reshape(-1, features.shape[-1])
    coords = coords.reshape(-1, 2)

    if len(features) != len(coords):
        raise ValueError(
            f"{h5_path.name}: {len(features)} embeddings but {len(coords)} coordinates."
        )

    missing = {"patch_size_level0", "patch_size"} - attrs.keys()
    if missing:
        raise KeyError(f"{h5_path.name}: missing attributes {sorted(missing)} on '{coords_key}'.")

    return SlideFeatures(
        features=features,
        coords=coords,
        patch_size_level0=int(attrs["patch_size_level0"]),
        patch_size=int(attrs["patch_size"]),
    )


@torch.no_grad()
def predict(
    model: AttentionMIL, features: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on a single slide.

    Parameters
    ----------
    model : AttentionMIL
        Trained model in evaluation mode.
    features : np.ndarray
        Patch embeddings of shape (n_patches, embedding_dim).
    device : torch.device
        Device on which inference is performed.

    Returns
    -------
    probs : np.ndarray
        Class probabilities, shape (num_classes,).
    attention : np.ndarray
        Attention weights summing to 1, shape (n_patches,).
    """
    bag = torch.from_numpy(features).to(device)
    logits, attention_list = model([bag])

    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    attention = attention_list[0].cpu().numpy()

    return probs, attention


# ============================================================
# HEATMAP
# ============================================================


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise scores to [0, 1].

    Parameters
    ----------
    scores : np.ndarray
        Raw scores.

    Returns
    -------
    np.ndarray
        Normalised scores; an array of ones if all scores are identical.
    """
    low, high = float(scores.min()), float(scores.max())
    if high <= low:
        return np.ones_like(scores)

    return (scores - low) / (high - low)


def check_background_geometry(
    background: Image.Image, wsi: openslide.OpenSlide, scale_factor: int, tolerance: int = 2
) -> None:
    """Warn if the thumbnail size is inconsistent with the WSI size.

    Parameters
    ----------
    background : PIL.Image.Image
        Downsampled WSI thumbnail.
    wsi : openslide.OpenSlide
        Opened whole-slide image.
    scale_factor : int
        Expected downsampling factor between level 0 and the thumbnail.
    tolerance : int, default=2
        Accepted deviation in pixels, to account for rounding.
    """
    expected_w = wsi.dimensions[0] / scale_factor
    expected_h = wsi.dimensions[1] / scale_factor

    if abs(background.width - expected_w) > tolerance or abs(background.height - expected_h) > tolerance:
        logger.warning(
            "Thumbnail size %dx%d differs from the expected %.0fx%.0f "
            "(WSI %dx%d / %d): the heatmap may be misaligned.",
            background.width,
            background.height,
            expected_w,
            expected_h,
            wsi.dimensions[0],
            wsi.dimensions[1],
            scale_factor,
        )


def plot_heatmap(
    background: Image.Image,
    coords: np.ndarray,
    attention: np.ndarray,
    patch_size_level0: int,
    cfg: XAIConfig,
    title: str,
    output_path: Path,
) -> None:
    """Overlay the attention scores on the WSI thumbnail and save the figure.

    Each patch is drawn as a square of ``patch_size_level0 / scale_factor``
    pixels anchored at its top-left corner. Patches are rendered as vector
    shapes; only the background thumbnail is raster.

    Parameters
    ----------
    background : PIL.Image.Image
        Downsampled WSI thumbnail.
    coords : np.ndarray
        Level-0 top-left patch coordinates, shape (n_patches, 2).
    attention : np.ndarray
        Raw attention weights, shape (n_patches,).
    patch_size_level0 : int
        Patch edge length at level 0, in pixels.
    cfg : XAIConfig
        Run configuration (scale factor, colormap, opacity, threshold).
    title : str
        First line of the figure title.
    output_path : Path
        Destination file; the extension selects the format.
    """
    scale = cfg.scale_factor
    tile = patch_size_level0 / scale

    scores = normalize_scores(attention)
    keep = scores >= cfg.score_threshold

    rectangles = [Rectangle((x / scale, y / scale), tile, tile) for x, y in coords[keep]]

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(background)

    collection = PatchCollection(
        rectangles,
        cmap=cfg.cmap,
        norm=Normalize(vmin=0.0, vmax=1.0),
        alpha=cfg.alpha,
        edgecolor="none",
        linewidths=0,
    )
    collection.set_array(scores[keep])
    collection.set_antialiased(False)  # Avoids visible seams between adjacent tiles.
    ax.add_collection(collection)

    fig.colorbar(collection, ax=ax, shrink=0.7, label="Normalized attention score")

    ax.set_title(
        f"{title}\n"
        f"Attention - min: {attention.min():.2e}, max: {attention.max():.2e}, "
        f"mean: {attention.mean():.2e}",
        fontsize=12,
    )
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Heatmap saved: %s (%d/%d patches drawn).", output_path, int(keep.sum()), len(coords))


# ============================================================
# PATCH EXTRACTION
# ============================================================


def read_patch(
    wsi: openslide.OpenSlide, x: int, y: int, size_level0: int, out_size: int
) -> Image.Image:
    """Read one patch from the WSI and resize it to the model input resolution.

    Parameters
    ----------
    wsi : openslide.OpenSlide
        Opened whole-slide image.
    x, y : int
        Top-left corner of the patch at level 0.
    size_level0 : int
        Patch edge length at level 0, in pixels.
    out_size : int
        Edge length of the returned patch, in pixels.

    Returns
    -------
    PIL.Image.Image
        RGB patch of size (out_size, out_size).
    """
    patch = wsi.read_region((x, y), 0, (size_level0, size_level0)).convert("RGB")

    if size_level0 != out_size:
        patch = patch.resize((out_size, out_size), Image.Resampling.LANCZOS)

    return patch


def extract_ranked_patches(
    wsi: openslide.OpenSlide,
    data: SlideFeatures,
    attention: np.ndarray,
    indices: np.ndarray,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Extract and save the patches at the given indices.

    File names encode the rank, the raw attention weight and the level-0
    coordinates, e.g. ``rank01_attn3.214e-03_x51200_y49664.png``.

    Parameters
    ----------
    wsi : openslide.OpenSlide
        Opened whole-slide image.
    data : SlideFeatures
        Slide embeddings, coordinates and patch sizes.
    attention : np.ndarray
        Raw attention weights, shape (n_patches,).
    indices : np.ndarray
        Patch indices to extract, in rank order.
    output_dir : Path
        Directory in which patches are written.

    Returns
    -------
    list[dict]
        One entry per patch (rank, index, coordinates, score, file name).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for rank, idx in enumerate(indices, start=1):
        x, y = (int(v) for v in data.coords[idx])
        score = float(attention[idx])
        file_name = f"rank{rank:02d}_attn{score:.3e}_x{x}_y{y}.png"

        patch = read_patch(wsi, x, y, data.patch_size_level0, data.patch_size)
        patch.save(output_dir / file_name)

        entries.append(
            {
                "rank": rank,
                "index": int(idx),
                "x": x,
                "y": y,
                "attention": score,
                "file": file_name,
            }
        )

    return entries


# ============================================================
# PER-SLIDE PIPELINE
# ============================================================


def process_slide(
    record: SlideRecord,
    model: AttentionMIL,
    cfg: XAIConfig,
    device: torch.device,
    run_heatmap: bool = True,
    run_patches: bool = True,
) -> dict[str, Any]:
    """Run inference, heatmap rendering and patch extraction for one slide.

    Parameters
    ----------
    record : SlideRecord
        Slide to process.
    model : AttentionMIL
        Trained model in evaluation mode.
    cfg : XAIConfig
        Run configuration.
    device : torch.device
        Device on which inference is performed.
    run_heatmap : bool, default=True
        Whether to render the attention heatmap.
    run_patches : bool, default=True
        Whether to extract the top/bottom patches.

    Returns
    -------
    dict
        JSON-serialisable summary of the slide (also written to disk as
        ``summary.json`` in the slide output directory).
    """
    data = read_h5(find_file(cfg.h5_dir, record.filename))

    if data.features.shape[1] != cfg.input_dim:
        raise ValueError(
            f"{record.slide_id}: embedding dimension {data.features.shape[1]} "
            f"does not match input_dim={cfg.input_dim}."
        )

    probs, attention = predict(model, data.features, device)
    pred = int(np.argmax(probs))

    true_name = cfg.class_names[record.target]
    pred_name = cfg.class_names[pred]
    logger.info(
        "%s | patches: %d | true: %s | pred: %s (p=%.3f)",
        record.slide_id,
        len(attention),
        true_name,
        pred_name,
        probs[pred],
    )

    slide_dir = cfg.output_dir / record.slide_id
    slide_dir.mkdir(parents=True, exist_ok=True)

    order = np.argsort(attention)
    top_indices = order[::-1][: cfg.top_k]
    bottom_indices = order[: cfg.bottom_k]

    summary: dict[str, Any] = {
        "slide_id": record.slide_id,
        "filename": record.filename,
        "true_label": true_name,
        "predicted_label": pred_name,
        "probabilities": {name: float(p) for name, p in zip(cfg.class_names, probs)},
        "n_patches": int(len(attention)),
        "patch_size_level0": data.patch_size_level0,
        "patch_size": data.patch_size,
        "attention": {
            "min": float(attention.min()),
            "max": float(attention.max()),
            "mean": float(attention.mean()),
        },
    }

    wsi_path = find_file(cfg.wsi_dir, f"{record.stem}.svs")

    with openslide.OpenSlide(str(wsi_path)) as wsi:
        if run_heatmap:
            png_path = find_file(cfg.png_dir, f"{record.stem}_1_{cfg.scale_factor}.png")
            background = Image.open(png_path).convert("RGB")
            check_background_geometry(background, wsi, cfg.scale_factor)

            heatmap_path = slide_dir / f"heatmap.{cfg.figure_ext}"
            title = (
                f"{record.slide_id} | True: {true_name} | "
                f"Pred: {pred_name} (p={probs[pred]:.2f})"
            )
            plot_heatmap(
                background, data.coords, attention, data.patch_size_level0, cfg, title, heatmap_path
            )
            summary["heatmap"] = heatmap_path.name

        if run_patches:
            summary["top_patches"] = extract_ranked_patches(
                wsi, data, attention, top_indices, slide_dir / "top"
            )
            summary["bottom_patches"] = extract_ranked_patches(
                wsi, data, attention, bottom_indices, slide_dir / "bottom"
            )
            logger.info(
                "Patches saved: %d top, %d bottom (%s).",
                len(top_indices),
                len(bottom_indices),
                slide_dir,
            )

    with open(slide_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# ENTRY POINT
# ============================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Attention heatmaps and top/bottom patches for MIL slide classification."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration.")
    parser.add_argument(
        "--slides",
        nargs="*",
        default=None,
        help="Slide ids, file names or stems to process (default: whole subset).",
    )
    parser.add_argument("--subset", default=None, help="Split name (default: evaluation.subset_name).")
    parser.add_argument("--weights", default=None, help="Model weights (default: evaluation.infer_model).")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--wsi-dir", default=None, help="Directory containing the .svs files.")
    parser.add_argument("--png-dir", default=None, help="Directory containing the downsampled PNGs.")
    parser.add_argument("--no-heatmap", action="store_true", help="Skip heatmap rendering.")
    parser.add_argument("--no-patches", action="store_true", help="Skip patch extraction.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the explainability pipeline on the selected slides.

    Returns
    -------
    int
        Exit code: 0 if all slides were processed, 1 otherwise.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args(argv)
    cfg = load_config(args.config, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    records = select_slides(load_slide_records(cfg.split_json, cfg.subset), args.slides)
    logger.info("%d slide(s) to process (subset '%s').", len(records), cfg.subset)

    model = load_model(cfg.weights, cfg.input_dim, cfg.num_classes, device)

    failures: list[str] = []
    for i, record in enumerate(records, start=1):
        logger.info("[%d/%d] %s", i, len(records), record.slide_id)
        try:
            process_slide(
                record,
                model,
                cfg,
                device,
                run_heatmap=not args.no_heatmap,
                run_patches=not args.no_patches,
            )
        except Exception:
            logger.exception("Failed to process slide %s.", record.slide_id)
            failures.append(record.slide_id)

    logger.info(
        "Done: %d/%d slide(s) processed. Outputs in %s.",
        len(records) - len(failures),
        len(records),
        cfg.output_dir,
    )
    if failures:
        logger.error("Failed slides: %s", failures)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())