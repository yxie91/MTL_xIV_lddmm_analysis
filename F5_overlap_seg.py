import numpy as np
import nibabel as nib
from pathlib import Path

t1_path     = "Data/Others/3915530_309_KRIMAR_220111_MPRAGE.nii.gz"
manual_path = "Data/Others/KRIMAR_220111_hand.nii.gz"
pred_path   = "Data/Others/KRIMAR_220111_pred.nii.gz"
manual_img = nib.load(manual_path)
pred_img   = nib.load(pred_path)

manual = manual_img.get_fdata().astype(np.int16)
pred   = pred_img.get_fdata().astype(np.int16)

if manual.shape != pred.shape:
    raise ValueError(f"Shape mismatch: manual {manual.shape}, pred {pred.shape}")

out = np.zeros_like(manual, dtype=np.int16)  # 0 = Clear Label

# Mapping: base label -> (overlap_label, manual_only_label, pred_only_label)
label_map = {
    1: (11, 21, 31),  # Amygdala
    2: (12, 22, 32),  # ERC
    3: (13, 23, 33),  # TEC
    4: (14, 24, 34),  # Hippocampus tail
    5: (15, 25, 35),  # Hippocampus head
}
for base_label, (lbl_overlap, lbl_manual_only, lbl_pred_only) in label_map.items():
    m = (manual == base_label)
    p = (pred   == base_label)

    both      = m & p
    manual_only = m & ~p
    pred_only   = p & ~m

    out[both]        = lbl_overlap
    out[manual_only] = lbl_manual_only
    out[pred_only]   = lbl_pred_only

conflict_mask = (manual > 0) & (pred > 0) & (manual != pred)
out[conflict_mask] = 99  # "conflict"

out_img = nib.Nifti1Image(out, manual_img.affine, manual_img.header)
out_path = Path(manual_path).with_name("KRIMAR_220111_overlap_conflict_labels.nii.gz")
nib.save(out_img, out_path)

print(f"Saved comparison label map to: {out_path}")

for lbl, name in [
    (11, "Overlap amygdala"),
    (12, "Overlap ERC"),
    (13, "Overlap TEC"),
    (14, "Overlap Hippo tail"),
    (15, "Overlap Hippo head"),
    (21, "Manual only amygdala"),
    (22, "Manual only ERC"),
    (23, "Manual only TEC"),
    (24, "Manual only Hippo tail"),
    (25, "Manual only Hippo head"),
    (31, "Pred only amygdala"),
    (32, "Pred only ERC"),
    (33, "Pred only TEC"),
    (34, "Pred only Hippo tail"),
    (35, "Pred only Hippo head"),
    (99, "Conflict (different labels)")
]:
    count = np.count_nonzero(out == lbl)
    print(f"{name:28s}: {count} voxels")
