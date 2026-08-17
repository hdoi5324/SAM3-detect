from __future__ import annotations

import datetime
import math
import os.path
from pathlib import Path
from typing import Optional, Sequence, Union, Any, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon as MplPolygon
from sam3.visualization_utils import generate_colors, plot_bbox, plot_mask
from sam3_exemplar.file_exemplars import load_exemplar_annotations
from sam3_exemplar.images import load_image


COLORS = generate_colors(n_colors=128, n_samples=5000)


# Assumes your helpers can take an Axes (recommended).
# If they don't, see the note inline for alternatives.
def plot_results_ax(img, results, ax, COLORS):
    """
    Draws SAM3 masks + boxes over the image on the provided Axes.
    """
    # Show base image
    if hasattr(img, "size"):  # PIL.Image
        ax.imshow(np.asarray(img))
        w, h = img.size
    else:  # numpy array
        ax.imshow(img)
        h, w = img.shape[:2][0], img.shape[:2][1]  # careful if grayscale
        # if grayscale, img.shape == (H, W)

    nb_objects = len(results["scores"])
    print(f"found {nb_objects} object(s)")

    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]
        plot_mask(results["masks"][i].squeeze(0).cpu(), color=color, ax=ax)
        w, h = img.size
        prob = results["scores"][i].item()
        plot_bbox(
            h,
            w,
            results["boxes"][i].cpu(),
            text=f"(id={i}, {prob=:.2f})",
            box_format="XYXY",
            color=color,
            relative_coords=False,
            ax=ax,
        )

    ax.set_axis_off()
    ax.set_title("SAM3 overlay")


def show_three_panels(
    img,
    results,
    heatmap,
    heat_mode="max",
    cmap="viridis",
    alpha=0.5,
    figsize=(16, 6),
    *,
    output_dir="./outputs",
    results_filename="three_panels.png",
    dpi=150,
    close_fig=True,
):
    """
    1) Original image
    2) SAM3 overlay (masks + boxes)
    3) Heatmap overlay on the original image, with colorbar

    Saves the figure to `output_dir/results_filename`.
    If `close_fig=True` (default), the figure is closed after saving to avoid memory issues.

    Returns
    -------
    out_path : pathlib.Path
        The path the figure was saved to. If close_fig=False, also returns (fig, axes, out_path).
    """


    # Convert image to array for display
    if hasattr(img, "size"):  # PIL
        im_arr = np.asarray(img)
    else:
        im_arr = img

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # 1) Original
    axes[0].imshow(im_arr)
    axes[0].set_title("Original")
    axes[0].set_axis_off()

    # 2) SAM3 overlay (assumes plot_results_ax and COLORS are defined/importable in your scope)
    plot_results_ax(img, results, ax=axes[1], COLORS=COLORS)

    # 3) Heatmap overlay (same size)
    axes[2].imshow(im_arr)
    hm = axes[2].imshow(heatmap,
                        cmap=cmap,
                        alpha=alpha,
                        vmin=0,  # lock lower bound
                        vmax=1  # lock upper bound
                        )
    axes[2].set_title("Heatmap from mask weighted score from SAM3")
    axes[2].set_axis_off()

    # Colorbar only for the heatmap axis
    cb = fig.colorbar(hm, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])  # optional: tidy ticks
    cb.set_ticklabels(["0", "0.25", "0.5", "0.75", "1"])  # optional: custom labels

    # Save
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / results_filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    # Close or return fig/axes based on preference
    if close_fig:
        plt.close(fig)
        return out_path
    else:
        return fig, axes, out_path

def plot_exemplars_side_by_side(
        image_paths,
        boxes_per_image: Optional[Sequence[Sequence[Sequence[float]]]] = None,
        points_per_image: Optional[Sequence[Sequence[Union[Sequence[float], dict]]]] = None,
        polygons_per_image: Optional[Sequence[Sequence[Optional[Sequence[Sequence[float]]]]]] = None,
        *,
        box_format: str = "xywh",
        titles: Optional[Sequence[Optional[str]]] = None,
        ncols: Optional[int] = None,
        figsize_per_col: float = 4.0,
        figsize_per_row: float = 4.0,
        box_colors: Optional[Union[str, Sequence[str]]] = None,
        point_colors: Optional[Union[str, Sequence[str]]] = None,
        box_linewidth: float = 2.0,
        point_size: float = 30,
        point_edgecolor: str = "k",
        point_alpha: float = 0.9,
        tight: bool = True,
        # --- NEW: saving options ---
        outputs_dir: Union[str, Path] = "./outputs",
        exemplar_img_file: Optional[str] = None,
        dpi: int = 150,
):
    """
    Plot exemplar images side by side with optional boxes and points overlays,
    then save the figure to `outputs_dir`.

    Parameters
    ----------
    image_paths : list
        List of images, each either a PIL.Image.Image or numpy array (H,W[,3]).
    boxes_per_image : list[list] or None
        For each image, a list of boxes. A box is [x1,y1,x2,y2] if box_format='xyxy',
        or [x,y,w,h] if box_format='xywh'. Use empty list for no boxes on that image.
        Default: None (no boxes for all).
    points_per_image : list[list] or None
        For each image, a list of points. A point can be [x,y] or (x,y). You can also pass
        a list of dicts like {'xy': (x,y), 'color': 'r'} if you need per-point color.
        Default: None (no points for all).
    polygons_per_image : list[list] or None
        For each image, a list of polygons. Each polygon is a sequence of relative (x, y)
        vertices. ``None`` entries are skipped. Used when exemplars have segmentation
        masks rather than axis-aligned boxes.
        Default: None (no points for all).
    box_format : {'xyxy', 'xywh'}
        Coordinates format for boxes. 'xyxy' = [x_min, y_min, x_max, y_max].
        'xywh' = [x_min, y_min, width, height].
    titles : list[str] or None
        Optional per-image titles.
    ncols : int or None
        Number of columns in the grid. If None, use min(len(images), 6).
        Rows are computed automatically.
    figsize_per_col : float
        Width (in inches) per column when computing figure size.
    figsize_per_row : float
        Height (in inches) per row when computing figure size.
    box_colors : list or str or None
        If list, will cycle colors across boxes (and across images).
        If str, single color for all boxes. If None, use a default cycle.
    point_colors : list or str or None
        If list, will cycle colors across points (and across images).
        If str, single color for all points. If None, use a default cycle.
    box_linewidth : float
        Line width for rectangle edges.
    point_size : float
        Marker size for points (passed to scatter 's' argument).
    point_edgecolor : str
        Edge color for point markers.
    point_alpha : float
        Alpha for points.
    tight : bool
        If True, use constrained_layout for tighter layout.
    outputs_dir : str or Path
        Directory to save the resulting figure. Will be created if it doesn't exist.
    exemplar_img_file : str or None
        File name to save as (e.g., "exemplars.png"). If None, a timestamped name is generated.
    dpi : int
        Output resolution for the saved figure.

    Returns
    -------
    fig, axes, out_path : (matplotlib.figure.Figure, numpy.ndarray | matplotlib.axes.Axes, pathlib.Path)
        The created figure, the axes array (shape R x C or 1D when N=1), and the saved file path.
    """
    # Defaults / normalization
    N = len(image_paths)
    if boxes_per_image is None:
        boxes_per_image = [[] for _ in range(N)]
    if points_per_image is None:
        points_per_image = [[] for _ in range(N)]
    if polygons_per_image is None:
        polygons_per_image = [[] for _ in range(N)]
    if titles is None:
        titles = [os.path.basename(i) for i in image_paths]

    images = [load_image(img) for img in image_paths]


    if ncols is None:
        ncols = min(N, 4)

    nrows = math.ceil(N / ncols) if N > 0 else 1

    # Figure size
    fig_w = ncols * figsize_per_col
    fig_h = nrows * figsize_per_row

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        constrained_layout=tight
    )
    # axes could be a single Axes if N=1
    if isinstance(axes, np.ndarray):
        axes_flat = axes.ravel()
    else:
        axes_flat = np.array([axes])

    # Color cycles
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])

    def color_for(idx, palette):
        if palette is None:
            return default_cycle[idx % len(default_cycle)]
        if isinstance(palette, str):
            return palette
        return palette[idx % len(palette)]

    # Draw each image with overlays
    for i in range(N):
        ax = axes_flat[i]
        img = images[i]

        # Show image
        if hasattr(img, "size"):  # PIL.Image
            ax.imshow(np.asarray(img))
            W, H = img.size
        else:
            arr = np.asarray(img)
            ax.imshow(arr)
            H, W = arr.shape[:2]

        ax.set_axis_off()
        if titles[i]:
            ax.set_title(titles[i])

        # Draw boxes
        for b_idx, box in enumerate(boxes_per_image[i] or []):
            if box is None:
                continue
            if box_format.lower() == "xyxy":
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = x1*W, y1*H, x2*W, y2*H
                w, h = (x2 - x1), (y2 - y1)
            elif box_format.lower() == "xywh":
                x1, y1, w, h = box
                x1, y1, w, h = x1*W, y1*H, w*W, h*H
            else:
                raise ValueError("box_format must be 'xyxy' or 'xywh'.")

            rect = plt.Rectangle(
                (x1, y1), w, h,
                fill=False,
                edgecolor=color_for(b_idx, box_colors),
                linewidth=box_linewidth
            )
            ax.add_patch(rect)

        # Draw polygons
        for p_idx, poly in enumerate(polygons_per_image[i] or []):
            if not poly:
                continue
            xy = [(float(x) * W, float(y) * H) for x, y in poly]
            patch = MplPolygon(
                xy,
                fill=False,
                edgecolor=color_for(p_idx, box_colors),
                linewidth=box_linewidth,
            )
            ax.add_patch(patch)

        # Draw points
        pts = points_per_image[i] or []
        xs, ys, cs = [], [], []
        for p_idx, p in enumerate(pts):
            if p is None:
                continue
            if isinstance(p, dict):
                (x, y) = p["xy"]
                c = p.get("color", color_for(p_idx, point_colors))
            else:
                x, y = p
                c = color_for(p_idx, point_colors)
            xs.append(x*W);
            ys.append(y*H);
            cs.append(c)
        if xs:
            ax.scatter(xs, ys, s=point_size, c=cs, edgecolors=point_edgecolor, alpha=point_alpha)

    # Hide any unused axes (if grid > N)
    for j in range(N, nrows * ncols):
        axes_flat[j].set_visible(False)

    # --- Saving (Option A style) ---
    out_dir = Path(outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if exemplar_img_file is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        exemplar_img_file = f"exemplars_{ts}.png"

    out_path = out_dir / exemplar_img_file
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path


def save_heatmap_per_image(
    heatmap: np.ndarray,
    image_path: Optional[str | Path] = None,
    cropped_image: np.ndarray = None,
    out_root: str | Path = "./outputs",
    *,
    # color map (matplotlib name or Colormap object)
    cmap: str = "viridis",
    # overlays
    save_overlay: bool = True,
    overlay_alpha: float = 0.5,
    # raw save options
    save_raw_npy: bool = True,
    save_color_png: bool = True,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """
    Save a heatmap as raw .npy and as a color-mapped .png (plus optional overlay).
    Returns: (raw_path, color_png_path, overlay_png_path)
    """
    assert heatmap.ndim == 2, f"heatmap must be HxW; got shape {heatmap.shape}"

    out_root = Path(out_root)
    (out_root / "heatmaps").mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    # ---- 1) Save RAW as .npy (exact values) ----
    raw_path = None
    if save_raw_npy:
        raw_path = out_root / "heatmaps" / f"{stem}.npy"
        # Ensure C-contiguous for speed if needed
        np.save(raw_path, np.ascontiguousarray(heatmap))

    # ---- 2) Colorize for visualization and save .png ----
    color_png_path = None
    overlay_png_path = None

    if save_color_png or (save_overlay and image_path):
        H, W = heatmap.shape

        # Map to RGBA via cmap
        colormap = mpl.colormaps["viridis"] if isinstance(cmap, str) else cmap
        rgba = (colormap(heatmap) * 255).astype(np.uint8)  # HxWx4

        if save_color_png:
            color_png_path = out_root / "heatmaps" / f"{stem}.png"
            Image.fromarray(rgba[..., :3]).save(color_png_path)

        # ---- 3) Optional overlay on the original image ----
        if save_overlay and image_path:
            base_img = cropped_image
            # Resize heatmap to match base image if needed
            if base_img.size != (W, H):
                hm_img = Image.fromarray(rgba[..., :3]).resize(base_img.size, resample=Image.BILINEAR)
            else:
                hm_img = Image.fromarray(rgba[..., :3])

            # Blend base and heatmap
            overlay = Image.blend(base_img, hm_img, alpha=overlay_alpha)
            overlay_png_path = out_root / "heatmaps" / f"{stem}_overlay.png"
            overlay.save(overlay_png_path)

    return raw_path, color_png_path, overlay_png_path