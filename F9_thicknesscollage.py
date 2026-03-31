#!/usr/bin/env python3
"""
Build 2D collages of surface thickness (stored in VTK files) using PyVista + Matplotlib.

- Recursively finds *.vtk in one or more input directories.
- Loads meshes with PyVista, extracts/triangulates the surface.
- Colors by the magnitude of the 'displacement' / thickness field.
- Projects to 2D and plots each subject as a tile in a rows×cols grid.

This version is headless-safe (uses Matplotlib's Agg backend).
"""

import os
import glob
import random
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
import pyvista as pv

# ----------------------- File discovery ------------------------
def find_vtk_files(root, patterns=("**/*.vtk",)):
    """Recursively find VTK files under a root directory."""
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(root, pat), recursive=True))
    # filter Mac sidecars etc.
    paths = [p for p in paths if os.path.isfile(p) and not os.path.basename(p).startswith("._")]
    return sorted(paths)

# -------------------- Displacement helpers ---------------------
def _find_array_case_insensitive(dattrs, name):
    name_lower = name.lower()
    for k in dattrs.keys():
        if k.lower() == name_lower:
            return k
    return None


def _magnitude_array(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return np.linalg.norm(arr[:, :3], axis=1)
    return np.abs(arr).ravel()


def attach_displacement_magnitude(poly):
    """
    Ensure point_data contains 'displacement_mag' for coloring.

    Priority:
      1) point_data['displacement'] (case-insensitive) -> magnitude
      2) cell_data['displacement']  -> cell->point map -> magnitude
      3) fallback: active scalars (point/cell) -> magnitude
    """
    # 1) point data
    key = _find_array_case_insensitive(poly.point_data, "displacement")
    if key is not None:
        poly.point_data["displacement_mag"] = _magnitude_array(poly.point_data[key])
        return poly, "displacement_mag"

    # 2) cell data, then convert to point data
    key = _find_array_case_insensitive(poly.cell_data, "displacement")
    if key is not None:
        tmp = poly.copy()
        tmp.cell_data["disp_mag_cell"] = _magnitude_array(tmp.cell_data[key])
        tmp = tmp.cell_data_to_point_data()
        tmp.point_data["displacement_mag"] = np.asarray(tmp.point_data["disp_mag_cell"]).ravel()
        del tmp.point_data["disp_mag_cell"]
        return tmp, "displacement_mag"

    # 3) fallback: active scalars
    if poly.active_scalars_name is not None:
        nm = poly.active_scalars_name
        if nm in poly.point_data:
            poly.point_data["displacement_mag"] = _magnitude_array(poly.point_data[nm])
            return poly, "displacement_mag"
        if nm in poly.cell_data:
            tmp = poly.copy()
            tmp.cell_data["tmp_cell"] = _magnitude_array(tmp.cell_data[nm])
            tmp = tmp.cell_data_to_point_data()
            tmp.point_data["displacement_mag"] = np.asarray(tmp.point_data["tmp_cell"]).ravel()
            del tmp.point_data["tmp_cell"]
            return tmp, "displacement_mag"

    raise RuntimeError("No usable 'displacement' scalar found (point/cell).")


# --------------------- Surface & triangles ---------------------


def extract_tris(poly):
    """
    Return (pts, tris, surf) where:
        pts  : (N, 3) array of vertex coordinates
        tris : (M, 3) int array of triangle indices
        surf : PyVista PolyData surface (triangulated)
    """
    try:
        surf = poly.extract_surface()
    except Exception:
        surf = poly
    try:
        surf = surf.triangulate()
    except Exception:
        pass

    faces = np.asarray(surf.faces)
    if faces.size == 0 and surf.n_points > 0:
        # last resort: build a surface with Delaunay
        try:
            vol = surf.delaunay_3d(alpha=0.0)
            surf = vol.extract_surface().triangulate()
            faces = np.asarray(surf.faces)
        except Exception:
            raise RuntimeError("Mesh has no faces and surface reconstruction failed.")

    faces = faces.reshape(-1, 4)  # [n, i0, i1, i2]
    tri_mask = faces[:, 0] == 3
    tris = faces[tri_mask, 1:4].astype(np.int32)
    if tris.size == 0:
        raise RuntimeError("No triangle faces found after triangulate().")

    return np.asarray(surf.points), tris, surf


# ------------------------- 2D projection ------------------------


def project_points(pts, view="yz", mirror=False):
    """
    Map 3D points to 2D for plotting.

    view:
      'yz': (y, z)
      'xz': (x, z)
      'xy': (x, y)

    If mirror=True, reflect the *first* axis of the chosen plane.
    """
    if view == "yz":
        a, b = pts[:, 1], pts[:, 2]
        if mirror:
            a = (a.min() + a.max()) - a
        return a, b

    if view == "xz":
        a, b = pts[:, 0], pts[:, 2]
        if mirror:
            a = (a.min() + a.max()) - a
        return a, b

    if view == "xy":
        a, b = pts[:, 0], pts[:, 1]
        if mirror:
            b = (b.min() + b.max()) - b
        return a, b

    # default fallback = yz
    a, b = pts[:, 1], pts[:, 2]
    if mirror:
        a = (a.min() + a.max()) - a
    return a, b


def rotate_points(x, y, angle_deg=90.0, center="bbox"):
    """
    Rotate (x, y) by angle_deg counterclockwise around the chosen center.

    center: 'bbox' (midpoint of extents), 'mean', or (cx, cy) tuple.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if center == "bbox":
        cx = 0.5 * (x.min() + x.max())
        cy = 0.5 * (y.min() + y.max())
    elif center == "mean":
        cx = float(x.mean())
        cy = float(y.mean())
    else:  # assume tuple
        cx, cy = center

    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    xr = c * (x - cx) - s * (y - cy) + cx
    yr = s * (x - cx) + c * (y - cy) + cy
    return xr, yr


# ----------------------- Collage builder ------------------------


def build_collage_from_vtk(
    inputs,
    out_png,
    limit=36,
    rows=None,
    cols=None,
    cmap="coolwarm",
    clim=(0.0, 5.0),
    center=None,
    view="xy",
    mirror=False,
    rotate=90,
    seed=42,
    figsize=None,
    bg_color=(0.20, 0.22, 0.28),
):
    """
    Build a collage of surfaces colored by |displacement|.

    inputs : str or list[str]
        One or more root directories containing VTK files.
    """

    # Normalize inputs to a list of roots
    if isinstance(inputs, str):
        roots = [inputs]
    else:
        roots = list(inputs)

    # collect files
    files = []
    for root in roots:
        files.extend(find_vtk_files(root))
    if not files:
        raise RuntimeError(f"No .vtk files found under: {roots}")

    rng = random.Random(seed)
    rng.shuffle(files)

    # grid size
    if rows is None or cols is None:
        n = int(np.sqrt(limit))
        rows = rows or n
        cols = cols or n
    need = rows * cols
    files = files[: max(need, min(limit, len(files)))]

    # figure
    if figsize is None:
        figsize = (cols * 1.2, rows * 1.2)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.patch.set_facecolor(bg_color)

    # colormap / normalization
    cmap_obj = plt.get_cmap(cmap)
    if center is None:
        norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    else:
        from matplotlib.colors import TwoSlopeNorm

        norm = TwoSlopeNorm(vmin=clim[0], vcenter=center, vmax=clim[1])
    axes = np.atleast_2d(axes)
    used = 0

    # global extents in centered coordinates
    global_half_width  = 0.0
    global_half_height = 0.0

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")

        if i >= len(files):
            continue

        path = files[i]
        try:
            ds = pv.read(path)
            pts, tris, surf = extract_tris(ds)
            surf2, scalar_name = attach_displacement_magnitude(surf)
            scalars = np.asarray(surf2.point_data[scalar_name]).ravel()

            # project + rotate
            x2d, y2d = project_points(pts, view=view, mirror=mirror)
            if rotate is not None:
                x2d, y2d = rotate_points(x2d, y2d, angle_deg=rotate, center="bbox")

            # ---- RECENTER THIS SUBJECT ----
            cx = 0.5 * (x2d.min() + x2d.max())
            cy = 0.5 * (y2d.min() + y2d.max())
            x2d_c = x2d - cx
            y2d_c = y2d - cy
            # --------------------------------

            # update global half-width/height (for absolute scale)
            global_half_width  = max(global_half_width,  np.max(np.abs(x2d_c)))
            global_half_height = max(global_half_height, np.max(np.abs(y2d_c)))

            tri = mtri.Triangulation(x2d_c, y2d_c, tris)
            ax.tripcolor(tri, scalars, shading="gouraud", cmap=cmap_obj, norm=norm)
            ax.set_aspect("equal", adjustable="box")
            used += 1
        except Exception as e:
            print(f"[skip] {os.path.basename(path)}: {e}")
            continue

    # use same limits for everyone, so mm-scale is shared
    xlim = (-global_half_width,  global_half_width)
    ylim = (-global_half_height, global_half_height)
    for ax in axes.ravel():
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    plt.tight_layout(pad=0.0)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved collage: {out_png} with {used} tiles (grid {rows}x{cols}).")
if __name__ == "__main__":
    limit = 36
    rows = None
    cols = None
    cmap = "coolwarm"
    clim = (0.0, 5.0)
    view = "xy"
    seed = 23

    # BIOCARD: combine wide & narrow folders by passing a list of roots
    biocard_roots = [
        "Data/BIOCARD_Surface/LH_MCI_thickness/wide",
        "Data/BIOCARD_Surface/LH_MCI_thickness/narrow",]
    build_collage_from_vtk(
        inputs=biocard_roots,
        out_png=f"Figure/BIOCARD{limit}_vtk.png",
        limit=limit,
        rows=rows,
        cols=cols,
        cmap=cmap,
        clim=clim,
        view=view,
        mirror=False,
        rotate=90,
        seed=seed,
    )
    seed = 41
    # ADNI
    adni_root = "/cis/home/yxie91/ADNI/ADNI_Surface/thickness"
    build_collage_from_vtk(
        inputs=adni_root,
        out_png=f"Figure/ADNI{limit}_vtk.png",
        limit=limit,
        rows=rows,
        cols=cols,
        cmap=cmap,
        clim=clim,
        view=view,
        mirror=False,
        rotate=90,
        seed=seed,
    )
