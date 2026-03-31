#!/usr/bin/env python3
"""
Make a 10x10 sagittal collage from T1 + segmentation NIfTI pairs.

Pairing rule (in one folder or subfolders):
- T1: <name>_000.nii.gz
- Seg: <name>.nii.gz

Labels (in seg):
1 = amygdala (pink)
2 = ERC (blue)
3 = TEC (red)
4 = hippocampus anterior (green)
5 = hippocampus posterior (yellow)
"""

import os
import re
import math
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import random
# ---------- Config ----------
LABEL_COLORS = {
    1: (1.00, 0.41, 0.71, 0.40),  # pink  (amygdala)
    2: (0.12, 0.56, 1.00, 0.40),  # blue  (ERC)
    3: (1.00, 0.00, 0.00, 0.40),  # red   (TEC)
    4: (0.20, 0.80, 0.20, 0.40),  # green (Hippo anterior)
    5: (1.00, 0.84, 0.00, 0.40),  # yellow(Hippo posterior)
}
LABELS = [1, 2, 3, 4, 5]


# ---------- Helpers ----------
def brain_mask_from_t1_slice(t1_slice):
    sl = np.asarray(t1_slice, dtype=float)
    if (sl == 0).any() and (sl > 0).any():
        mask = sl > 0
    else:
        p10, p70 = np.percentile(sl, [10, 70])
        thr = 0.5 * (p10 + p70)
        mask = sl > thr
    mask = ndi.binary_closing(mask, structure=np.ones((3,3)))
    mask = ndi.binary_fill_holes(mask)
    lbl, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, lbl, index=range(1, n+1))
        keep = 1 + int(np.argmax(sizes))
        mask = (lbl == keep)
    return mask

def find_center_yx(t1_slice, seg_slice):
    """
    Choose crop center (y, x) for a sagittal slice.
    Prefer labels union; fallback to whole-brain mask; else image center.
    """
    seg_mask = seg_slice > 0
    if seg_mask.any():
        cy, cx = ndi.center_of_mass(seg_mask)
    else:
        bmask = brain_mask_from_t1_slice(t1_slice)
        if bmask.any():
            cy, cx = ndi.center_of_mass(bmask)
        else:
            h, w = t1_slice.shape
            return (h // 2, w // 2)
    # center_of_mass returns floats; guard NaNs
    h, w = t1_slice.shape
    if not np.isfinite(cy) or not np.isfinite(cx):
        return (h // 2, w // 2)
    return (int(round(cy)), int(round(cx)))

def crop_fixed_window(arr, center_yx, size=(240,240), pad_value=0):
    """
    Crop a fixed HxW window around center (y,x); clamp to bounds; pad if needed.
    Returns cropped view and the (y0,y1,x0,x1) box actually used.
    """
    H, W = size
    h, w = arr.shape
    cy, cx = center_yx
    hy, hx = H // 2, W // 2

    y0 = cy - hy
    y1 = y0 + H
    x0 = cx - hx
    x1 = x0 + W

    # Shift window to fit within the image
    if y0 < 0:
        y1 -= y0  # move down
        y0 = 0
    if x0 < 0:
        x1 -= x0  # move right
        x0 = 0
    if y1 > h:
        y0 -= (y1 - h)
        y1 = h
    if x1 > w:
        x0 -= (x1 - w)
        x1 = w

    # After shifting, clamp again
    y0 = max(0, y0); x0 = max(0, x0)
    y1 = min(h, y1); x1 = min(w, x1)

    crop = arr[y0:y1, x0:x1]

    # Pad if the slice is smaller than the target window (rare)
    pad_top    = 0
    pad_bottom = H - crop.shape[0]
    pad_left   = 0
    pad_right  = W - crop.shape[1]
    if pad_bottom < 0 or pad_right < 0:
        # Shouldn’t happen after clamping, but guard anyway
        pad_bottom = max(0, pad_bottom)
        pad_right  = max(0, pad_right)

    if pad_bottom > 0 or pad_right > 0:
        crop = np.pad(
            crop,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=pad_value
        )

    # Ensure exact shape
    crop = crop[:H, :W]
    return crop, (y0, y1, x0, x1)


def as_ras_data(path):
    """Load NIfTI and reorient to RAS; return ndarray data (x,y,z)."""
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    return data

def percentile_normalize(x, lo=1, hi=99):
    a, b = np.percentile(x, [lo, hi])
    if b <= a:
        b = a + 1e-6
    y = (x - a) / (b - a)
    return np.clip(y, 0, 1)

def find_pairs(root):
    """
    Find pairs by base name: T1 '<name>_000.nii.gz' & seg '<name>.nii.gz'.
    Recurses under root.
    Returns list of (t1_path, seg_path, name).
    """
    t1_paths = glob.glob(os.path.join(root, "imagesTr", "*_0000.nii.gz"), recursive=True)
    #print(t1_paths)
    pairs = []
    for t1 in t1_paths:
        base = os.path.basename(t1)
        m = re.match(r"(.+)_0000\.nii\.gz$", base)
        if not m:
            continue
        name = m.group(1)
        seg = os.path.join(os.path.dirname(t1), f"{name}.nii.gz")
        
        if not os.path.exists(seg):
            # try to find seg anywhere under root
            candidates = glob.glob(os.path.join(root, "labelsTr", f"{name}.nii.gz"), recursive=True)
            #print(candidates)
            if candidates:
                seg = candidates[0]
            else:
                continue
        pairs.append((t1, seg, name))
    return pairs

def best_k_sagittal_indices(seg, k=3, min_gap=5):
    """
    Rank sagittal slices by (1) number of distinct labels present (prefer 5),
    then (2) total labeled voxels. Return up to k indices with a min slice-gap.
    """
    seg = np.asarray(seg)
    X = seg.shape[0]
    LABELS = [1, 2, 3, 4, 5]

    # Score each slice
    scores = []
    for x in range(X):
        sl = seg[x, :, :]
        counts = [(sl == L).sum() for L in LABELS]
        num_present = sum(c > 0 for c in counts)
        total = sum(counts)
        scores.append((x, num_present, total))

    # Sort: more labels first, then more voxels
    scores.sort(key=lambda t: (t[1], t[2]), reverse=True)

    # Greedy pick with min_gap to avoid near-duplicate neighboring slices
    chosen = []
    for x, npresent, total in scores:
        if all(abs(x - c) >= min_gap for c in chosen):
            chosen.append(x)
            if len(chosen) >= k:
                break
    return chosen

def make_overlay_rgba(seg_slice, colors_dict, out_shape=None):
    """
    Build an RGBA overlay image from a label slice (2D).
    If out_shape is provided, resize with nearest-neighbor.
    """
    h, w = seg_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for L, color in colors_dict.items():
        mask = (seg_slice == L)
        if mask.any():
            rgba[mask] = color
    if out_shape is not None and (out_shape[0] != h or out_shape[1] != w):
        # nearest neighbor resize for labels/overlay
        y_idx = (np.linspace(0, h - 1, out_shape[0])).round().astype(int)
        x_idx = (np.linspace(0, w - 1, out_shape[1])).round().astype(int)
        rgba = rgba[np.ix_(y_idx, x_idx)]
    return rgba

def prepare_tile(t1, seg, x_idx, rotate_k=1, target_hw=(200, 200), brain_margin=None):
    """
    Extract sagittal slice, compute center, crop a fixed (240,240) window, no resizing.
    """
    t1_slice = t1[x_idx, :, :]
    seg_slice = seg[x_idx, :, :]

    # Optional rotation for display convention
    if rotate_k:
        t1_slice = np.rot90(t1_slice, k=rotate_k)
        seg_slice = np.rot90(seg_slice, k=rotate_k)

    # Find crop center (labels -> brain -> image center)
    cy, cx = find_center_yx(t1_slice, seg_slice)

    # Fixed-window crop for both T1 and seg (no resampling)
    t1_crop, box = crop_fixed_window(t1_slice, (cy, cx), size=target_hw, pad_value=0)
    seg_crop, _  = crop_fixed_window(seg_slice, (cy, cx), size=target_hw, pad_value=0)

    # Normalize T1 for display; build overlay
    t1_norm = percentile_normalize(t1_crop)
    overlay = make_overlay_rgba(seg_crop, LABEL_COLORS)
    return t1_norm, overlay


def plot_collage(tiles, out_png, rows=10, cols=10, figsize=(20, 20), titles=None):
    """
    tiles: list of (base_gray, overlay_rgba). Length <= rows*cols.
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i < len(tiles):
            base, overlay = tiles[i]
            ax.imshow(base, cmap="gray", vmin=0, vmax=1, origin="lower")
            ax.imshow(overlay, origin="lower")
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=8)
    plt.tight_layout(pad=0.05, w_pad=0.0, h_pad=0.0)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------- Main pipeline ----------
def build_collage(input_root, out_png="collage_20x20.png", limit=0,
                  rotate_k=1, seed=21):
    pairs = find_pairs(input_root)
    if not pairs:
        raise RuntimeError("No T1/seg pairs found under: " + input_root)
    all_tiles = []
    titles = []

    rng = random.Random(seed)

    for (t1_path, seg_path, name) in pairs:
        try:
            t1 = as_ras_data(t1_path)
            seg = as_ras_data(seg_path)
            if t1.shape != seg.shape:
                print(f"[skip] shape mismatch for {name}: T1{t1.shape} vs seg{seg.shape}")
                continue

            # Top 3 sagittal slices for this subject
            k_indices = best_k_sagittal_indices(seg, k=2, min_gap=5)

            for x_idx in k_indices:
                #print(x_idx)
                base, overlay = prepare_tile(
                    t1, seg, x_idx, rotate_k=rotate_k,target_hw=TARGET_HW)
                all_tiles.append((base, overlay))
                # Optional title per tile if you want it during debugging:
                titles.append(f"{name} | x={x_idx}")

        except Exception as e:
            print(f"[error] {name}: {e}")

    # Shuffle across ALL tiles from all subjects, then take first 400
    idxs = list(range(len(all_tiles)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    tiles = [all_tiles[i] for i in idxs]
    # titles_sel = [titles[i] for i in idxs]  # if you want titles

    # Plot 20x20 collage
    plot_collage(
        tiles,
        out_png,
        rows=GRID_ROWS,
        cols=GRID_COLS,
        figsize=(24, 24),
        titles=None  # or titles_sel
    )
    print(f"Saved collage: {out_png} with {len(tiles)} tiles (from {len(all_tiles)} candidates).")

if __name__ == "__main__":

    limit=25#36#200
    TARGET_HW = (180, 180)
    GRID_ROWS = int(np.sqrt(limit))
    GRID_COLS = int(np.sqrt(limit))
    MAX_TILES = GRID_ROWS * GRID_COLS  # 400
    '''build_collage(
        '/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset103_ADNIall/',
        f'/cis/home/yxie91/paper1/Figure/ADNI{limit}_{TARGET_HW[0]}.png',
        limit=limit,
        rotate_k=3)
    build_collage(
        '/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset006_axis0/',
        f'/cis/home/yxie91/paper1/Figure/BIOCARD{limit}_{TARGET_HW[0]}.png',
        limit=limit,
        rotate_k=3)'''
    limit=36
    TARGET_HW = (180, 180)
    GRID_ROWS = int(np.sqrt(limit))
    GRID_COLS = int(np.sqrt(limit))
    MAX_TILES = GRID_ROWS * GRID_COLS  # 400
    build_collage(
        '/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset103_ADNIall/',
        f'/cis/home/yxie91/paper1/Figure/ADNI{limit}_{TARGET_HW[0]}.png',
        limit=limit,
        rotate_k=3)
    build_collage(
        '/cis/home/yxie91/nnUNet/nnUNet_raw/Dataset006_axis0/',
        f'/cis/home/yxie91/paper1/Figure/BIOCARD{limit}_{TARGET_HW[0]}.png',
        limit=limit,
        rotate_k=3)


