import h5py
import numpy as np
from pathlib import Path

fpath = "/home/h_larbi/ilc_Xclass_20261026/prepros/UNI2-h_features/TCGA/TCGA-GM-A2DI-01Z-00-DX1.5E9715EA-9D2B-49D0-8F1D-972CF24B1960.h5"


with h5py.File(fpath, "r") as f:
    print("=" * 70)
    print(f"File : {Path(fpath).name}")
    print(f"Size : {Path(fpath).stat().st_size / 1e6:.1f} MB")
    print("=" * 70)

    # --- 1. Root attributes (optional metadata) ---
    print("\n[ Root attributes ]")
    if len(f.attrs) == 0:
        print("  (none)")
    for k, v in f.attrs.items():
        print(f"  {k}: {v}")

    # --- 2. Recursive structure ---
    print("\n[ Structure ]")
    def show(name, obj):
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            print(f"{indent}{name}/  (group)")
    f.visititems(show)

    # --- 3. Dataset details + global stats ---
    print("\n[ Dataset details ]")
    def describe(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        print(f"\n--- {name} ---")
        print(f"  shape      : {obj.shape}")
        print(f"  dtype      : {obj.dtype}")
        print(f"  ndim       : {obj.ndim}")
        print(f"  size       : {obj.nbytes / 1e6:.2f} MB")
        print(f"  compression: {obj.compression}")

        # Dataset attributes (often useful: patch_size, level, etc.)
        if len(obj.attrs) > 0:
            print("  attrs      :")
            for ak, av in obj.attrs.items():
                print(f"    {ak}: {av}")

        # Global stats (only if numeric and not too large)
        if np.issubdtype(obj.dtype, np.number) and obj.size < 5e7:
            arr = obj[()]
            arr_flat = arr.ravel()
            print(f"  min        : {np.nanmin(arr_flat):.4f}")
            print(f"  max        : {np.nanmax(arr_flat):.4f}")
            print(f"  mean       : {np.nanmean(arr_flat):.4f}")
            print(f"  std        : {np.nanstd(arr_flat):.4f}")
            print(f"  NaN count  : {np.isnan(arr_flat).sum() if np.issubdtype(arr.dtype, np.floating) else 0}")

    f.visititems(describe)

    # --- 4. "Head" preview of 2D+ datasets ---
    print("\n[ Head of datasets (first 5 rows, up to 8 columns) ]")
    def head(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
            print(f"\n--- {name} ---")
            slices = tuple(slice(0, min(5, s)) for s in obj.shape)
            if obj.ndim >= 2:
                slices = (slice(0, min(5, obj.shape[0])), slice(0, min(8, obj.shape[1])))
            try:
                print(obj[slices])
            except Exception as e:
                print(f"  could not read: {e}")
    f.visititems(head)