"""Brain-surface figures composed from brainspace renders.

Each hemisphere view is rendered offscreen by brainspace (same camera, colour map
and zoom as ``plot_hemispheres``) and placed in matplotlib at the viewport
brainspace gives it, which adds annotations brainspace cannot draw: circular zoom
insets of regions of interest under the brain row, with a marker circle on the
region and a vertical line to its inset, and a traced region outline. A forest plot
(``plot_forest``) can accompany stacked inset rows, one forest row level with each.
"""

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.font_manager import FontProperties, findfont, get_font
from matplotlib.patches import Circle, Rectangle

from brainspace.plotting import plot_surf

logger = logging.getLogger(__name__)

# plot_hemispheres columns -> (hemisphere, brainspace camera view). brainspace points
# the same camera at both hemispheres, so its 'lateral' camera shows the medial wall
# of the right hemisphere and its 'medial' camera the right lateral surface.
HEMI_VIEWS = {
    "lh_lateral": ("lh", "lateral"),
    "lh_medial": ("lh", "medial"),
    "rh_medial": ("rh", "lateral"),
    "rh_lateral": ("rh", "medial"),
}

# brainspace window layout (plotting.utils._gen_grid): a shared colour bar takes this
# fraction of the window width and the views split the rest equally.
_CBAR_FRAC = 0.11
# brainspace scalar bar inside its column (defaults_plotting.scalarBarActor_kwds):
# x offset, bar width (width x barRatio), and top/bottom of the bar, as column fractions.
_CBAR_BOX = (0.08, 0.8 * 0.27, 0.25, 0.75)

# Shared colour scale of the Figure 1 maps z-scored within the salience network (MPC
# gradient, BigBrain, Bielschowsky, Parvalbumin): symmetric, so one colour means the same
# deviation from the SN mean in every panel; ~96 % of each map's SN vertices fall inside.
SALIENCE_Z_RANGE = (-2.0, 2.0)
SALIENCE_Z_LABELS = ('\u22122', '+2')  # colour-bar ends (lower, upper)

# Display width (mm) of the 1450-pt zoom rows in the Figure 1 and 3 layouts (~0.29x).
ZOOM_ROW_WIDTH_MM = 150
# plot_forest canvas (pt, drawn at its final size), that of figure_1b_forest.svg: figure
# width, axes x-span and bottom margin (x tick labels + label) copy the Figure 1a
# profiles panel as laid out in Figure 1, so the x axes line up when stacked; font size
# and null-bar height.
_FOREST_WIDTH_PT = 177.845669
_FOREST_AXIS_PT = (38.17095, 158.912516)
_FOREST_BOTTOM_PT = 28.346457
_FOREST_FONT_PT = 8.8
_FOREST_BAR_PT = 8.5

# Opacity gain of a traced outline (see _outline_alpha): the darkening peaks just below
# 1 on the outline vertices and falls off across their triangles, so a gain of 1.5
# makes the line's core opaque and its soft edges narrower.
_OUTLINE_GAIN = 1.5

# Figure 1 salience-network zoom regions: (view, Schaefer-400 parcels without the
# leading '7Networks_', label).
SALIENCE_ZOOM_ROIS = [
    ("lh_lateral", [f"LH_SalVentAttn_FrOperIns_{i}" for i in range(1, 10)], "Anterior insula\n(aINS)"),
    ("lh_medial", [f"LH_SalVentAttn_Med_{i}" for i in (1, 2, 4)], "Dorsal anterior\ncingulate cortex\n(dACC)"),
    ("rh_medial", [f"RH_SalVentAttn_Med_{i}" for i in (3, 5, 6)], "Precuneus\n(PCU)"),
    ("rh_lateral", [f"RH_SalVentAttn_TempOccPar_{i}" for i in range(1, 8)], "Inferior parietal\nlobule (IPL)"),
]


def salience_zoom_rois(df_surf: pd.DataFrame, hemisphere: str = 'both', with_labels: bool = True) -> list[dict]:
    """``plot_hemispheres_zoom`` regions for ``SALIENCE_ZOOM_ROIS``.

    Args:
        df_surf: Surface table over both hemispheres with the Schaefer-400 ``label``
            column (e.g. ``df_yeo_surf``).
        hemisphere: ``'both'``, ``'LH'`` or ``'RH'``; regions of the other
            hemisphere are left out.
        with_labels: Attach the region names (drawn under the insets).

    Returns:
        list[dict]: One ``{'view', 'mask', 'label'}`` dict per region.
    """
    labels = df_surf['label'].fillna('')
    return [
        {"view": view, "label": label if with_labels else None,
         "mask": labels.isin([f"7Networks_{p}" for p in parcels]).to_numpy()}
        for view, parcels, label in SALIENCE_ZOOM_ROIS
        if hemisphere == 'both' or view.startswith(hemisphere.lower())
    ]


def mask_outline(surf_lh, surf_rh, mask: np.ndarray) -> np.ndarray:
    """Vertices just outside a region: the ``outline`` of ``plot_hemispheres_zoom``.

    Args:
        surf_lh: Left-hemisphere surface (``BSPolyData``).
        surf_rh: Right-hemisphere surface (``BSPolyData``).
        mask: Bool array over both hemispheres, LH then RH, shape (n_lh + n_rh,),
            e.g. the vertices of a network.

    Returns:
        np.ndarray: Bool array, same shape, True at vertices outside ``mask`` that
        share a triangle with a vertex inside it, so a line traced on them encloses
        the region without covering its values.
    """
    mask = np.asarray(mask, dtype=bool)
    outline = np.zeros_like(mask)
    offset = 0
    for surf in (surf_lh, surf_rh):
        faces = np.asarray(surf.GetCells2D()) + offset
        inside = mask[faces]
        mixed = inside.any(axis=1) & ~inside.all(axis=1)
        outline[faces[mixed][~inside[mixed]]] = True
        offset += surf.n_points
    return outline


def _vtk_matrix(m) -> np.ndarray:
    """Copy a vtkMatrix4x4 into a (4, 4) array."""
    return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])


def _vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals, oriented away from the mesh centre."""
    v0, v1, v2 = (points[faces[:, k]] for k in range(3))
    face_normals = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(points)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    # Triangle winding differs between surface files; orient outward.
    if np.sum(normals * (points - points.mean(axis=0))) < 0:
        normals = -normals
    return normals


def render_surface_view(
    surf,
    values: np.ndarray,
    view: str,
    *,
    size: tuple[int, int] = (400, 300),
    scale: int = 4,
    zoom: float = 1.3,
    cmap: str = "coolwarm",
    color_range: tuple[float, float] | None = None,
    nan_color: tuple = (220, 220, 220, 1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Render one brainspace surface view and project its vertices into the image.

    The view is rendered offscreen exactly as ``plot_hemispheres`` renders one
    panel (orthographic camera, ``zoom``, transparent background) at ``scale``
    times ``size``. The camera and actor matrices of that render map every vertex
    to its pixel position, so annotations can be placed on the image exactly.

    Args:
        surf: Hemisphere surface (``BSPolyData``).
        values: Per-vertex values, shape (n_vertices,); NaN is drawn in ``nan_color``.
        view: brainspace camera view, ``'lateral'``, ``'medial'``, ``'dorsal'`` or
            ``'ventral'`` (see ``HEMI_VIEWS`` for the right-hemisphere convention).
        size: Viewport size (width, height) in pixels before ``scale``.
        scale: Supersampling factor of the returned image.
        zoom: Camera zoom, as in ``plot_hemispheres``.
        cmap: Matplotlib colour map name.
        color_range: (min, max) of the colour map; data range if ``None``.
        nan_color: RGBA colour of NaN vertices (RGB in 0-255, alpha in 0-1).

    Returns:
        tuple:
            - img: RGBA image, shape (size[1]*scale, size[0]*scale, 4), uint8.
            - xy: pixel position (column, row) of every vertex in ``img``,
              shape (n_vertices, 2).
            - visible: vertices whose normal faces the camera, shape (n_vertices,).
            - px_per_mm: image pixels per surface unit (mm).
    """
    name = surf.append_array(np.asarray(values, dtype=float), at="p")
    try:
        p = plot_surf({"s": surf}, [["s"]], array_name=[[name]], view=[[view]],
                      cmap=cmap, color_range=color_range, nan_color=nan_color,
                      zoom=zoom, size=size, return_plotter=True, offscreen=True,
                      suppress_warnings=True)
        img = p.to_numpy(transparent_bg=True, scale=(scale, scale))
        ren = p.renderers[0][0]
        cam = ren.activeCamera.VTKObject
        model = _vtk_matrix(ren.actors.lastActor.VTKObject.GetMatrix())
        proj = _vtk_matrix(cam.GetCompositeProjectionTransformMatrix(size[0] / size[1], -1, 1))
        dop = np.asarray(cam.GetDirectionOfProjection())
        parallel_scale = cam.GetParallelScale()
        p.close()
    finally:
        surf.remove_array(name, at="p")

    points = np.asarray(surf.Points, dtype=float)
    h = (proj @ model @ np.c_[points, np.ones(len(points))].T).T
    ndc = h[:, :2] / h[:, 3:4]
    rows, cols = img.shape[:2]
    xy = np.column_stack([(ndc[:, 0] + 1) / 2 * cols, (1 - ndc[:, 1]) / 2 * rows])

    normals = _vertex_normals(points, np.asarray(surf.GetCells2D())) @ model[:3, :3].T
    visible = normals @ dop < 0
    return img, xy, visible, rows / (2 * parallel_scale)


def _crop(img: np.ndarray, x_lo: int, x_hi: int, y_lo: int, y_hi: int) -> np.ndarray:
    """``img[y_lo:y_hi, x_lo:x_hi]``, padded with transparent pixels outside ``img``."""
    out = np.zeros((y_hi - y_lo, x_hi - x_lo, img.shape[2]), dtype=img.dtype)
    ys, xs = slice(max(y_lo, 0), min(y_hi, img.shape[0])), slice(max(x_lo, 0), min(x_hi, img.shape[1]))
    out[ys.start - y_lo:ys.stop - y_lo, xs.start - x_lo:xs.stop - x_lo] = img[ys, xs]
    return out


def _outline_alpha(surf, outline: np.ndarray, view: str, **render_kwargs) -> np.ndarray:
    """Per-pixel opacity of an outline traced on the surface, from two brainspace renders.

    The outline vertices are rendered black on an otherwise white surface, and the
    same surface is rendered all white, with the camera of ``render_surface_view``
    (``render_kwargs``: ``size``, ``scale``, ``zoom``). VTK resolves occlusion and
    shading in both, so their luminance ratio is the line's darkening alone: 1 on
    the outline, 0 elsewhere and off the surface. The line spans the triangles
    around the outline vertices (the scalars are interpolated across them); the
    opacity is scaled by ``_OUTLINE_GAIN`` so the line has an opaque core and
    narrower soft edges.
    """
    renders = [render_surface_view(surf, vals, view, cmap="binary", color_range=(0.0, 1.0),
                                   nan_color=(255, 255, 255, 1), **render_kwargs)[0]
               for vals in (np.asarray(outline, dtype=float), np.zeros(surf.n_points))]
    lum_mark, lum_ref = (im[..., :3].astype(float) @ np.array([0.2126, 0.7152, 0.0722]) for im in renders)
    alpha = _OUTLINE_GAIN * (1 - lum_mark / np.maximum(lum_ref, 1e-6))
    return np.where(renders[1][..., 3] > 0, np.clip(alpha, 0.0, 1.0), 0.0)


def plot_hemispheres_zoom(
    surf_lh,
    surf_rh,
    values: np.ndarray,
    rois: list[dict],
    *,
    views: tuple[str, ...] = tuple(HEMI_VIEWS),
    size: tuple[int, int] = (1450, 300),
    zoom: float = 1.3,
    color_bar: bool = True,
    draw_color_bar: bool | None = None,
    roi_only: bool = False,
    title: str | None = None,
    filename=None,
    cmap: str = "coolwarm",
    color_range: tuple[float, float] | None = None,
    nan_color: tuple = (220, 220, 220, 1),
    scale: int = 4,
    inset_frac: float = 0.75,
    gap: float = 0.0,
    roi_pad: float = 1.15,
    cbar_labels: tuple[str, str] | None = None,
    cbar_title: str | None = None,
    label_fontsize: float = 18,
    title_fontsize: float = 20,
    inset_facecolor="white",
    line_color="black",
    line_width: float = 1.0,
    outline: np.ndarray | None = None,
    outline_color="black",
    dpi: int = 300,
) -> plt.Figure:
    """``plot_hemispheres`` row with circular zoom insets of regions of interest.

    The brain row reproduces the ``plot_hemispheres`` window of the same ``size``,
    ``zoom`` and ``color_bar``: the figure is ``size[0]`` pt wide, one pt per window
    pixel, and every view sits in the viewport brainspace gives it (equal columns,
    plus an 11 % colour-bar column when ``color_bar``). Its top ``size[1]`` pt
    therefore overlay a ``plot_hemispheres`` SVG export exactly. Under each column
    with a region of interest, ``gap`` pt below the brain row, a circular inset
    shows the region magnified; a circle marks the region on its view and a
    vertical line drops from that circle to the inset. Each inset is a crop of the
    same supersampled render, so it shows exactly the circled area. An optional
    ``outline`` (e.g. the border vertices of a network) is traced on every view and
    therefore magnified in the insets too. With
    ``roi_only`` only the inset row is drawn (no brain row, markers or lines, and no
    colour bar unless ``draw_color_bar``), with the same width and columns, so
    placing it ``size[1]`` pt below a ``plot_hemispheres`` export reproduces the full
    figure. An optional ``title`` adds a band above everything, which shifts the
    rows down by its height.

    Args:
        surf_lh: Left-hemisphere surface (``BSPolyData``).
        surf_rh: Right-hemisphere surface (``BSPolyData``).
        values: Per-vertex values for both hemispheres, LH then RH, shape
            (n_lh + n_rh,), as for ``plot_hemispheres``.
        rois: One dict per region of interest, at most one per view:
            - ``'view'``: key of ``HEMI_VIEWS`` the region is shown on;
            - ``'mask'``: bool array over both hemispheres, shape (n_lh + n_rh,);
            - ``'label'`` (optional): text under the inset (may contain ``\\n``);
            - ``'radius'`` (optional): zoom-window radius in mm; by default the
              smallest circle around the region's camera-facing vertices,
              enlarged by ``roi_pad``.
        views: Columns of the brain row, keys of ``HEMI_VIEWS`` (default: the four
            ``plot_hemispheres`` panels).
        size: ``plot_hemispheres`` window (width, height) in pixels = figure pt.
        zoom: Camera zoom, as in ``plot_hemispheres``.
        color_bar: Reserve the colour-bar column, as
            ``plot_hemispheres(color_bar='right')``.
        draw_color_bar: Draw the colour bar in that column, with the size and
            position brainspace gives it, centred on the brain row (or on the inset
            row when ``roi_only``). Default: drawn unless ``roi_only``.
        roi_only: Draw only the inset row.
        title: Text above the figure, starting at the left edge of the leftmost
            drawn brain or inset.
        filename: Output path; the figure is only returned if ``None``.
        cmap: Matplotlib colour map name.
        color_range: (min, max) shared by all views; finite data range if ``None``.
        nan_color: RGBA colour of NaN vertices (RGB in 0-255, alpha in 0-1).
        scale: Supersampling factor of the renders (inset sharpness).
        inset_frac: Inset diameter as a fraction of the view width.
        gap: Vertical space between the brain row and the insets, in pt.
        roi_pad: Enlargement of the automatic zoom-window radius.
        cbar_labels: Texts below and above the colour bar (e.g. ``('min', 'max')``);
            the ``color_range`` values if ``None``.
        cbar_title: Text written vertically beside the colour bar (e.g. its unit).
        label_fontsize: Font size (pt) of the inset labels and colour-bar texts.
        title_fontsize: Font size (pt) of ``title``.
        inset_facecolor: Fill of the inset circles behind the surface.
        line_color: Colour of the marker circles, inset outlines and lines.
        line_width: Line width (pt) of the marker circles, inset outlines and lines.
        outline: Optional bool array over both hemispheres, shape (n_lh + n_rh,), of
            vertices traced as a line in ``outline_color`` over the map (e.g.
            ``mask_outline`` of a network, which encloses it). It is rendered by
            brainspace on the same camera (two extra renders per view), so hidden
            parts stay hidden.
        outline_color: Colour of the ``outline`` line.
        dpi: Resolution of the embedded raster images when saving.

    Returns:
        matplotlib.figure.Figure: The composed figure. Its ``inset_cy`` attribute is
        the centre of the inset row in pt from the figure top (with its height,
        ``fig.get_size_inches()[1] * 72``, it lets a companion panel align rows).

    Raises:
        ValueError: If a region's view is not in ``views``, a view has more than one
            region, or a region has no vertex facing the camera in its view.
    """
    unknown = [v for v in views if v not in HEMI_VIEWS]
    if unknown:
        raise ValueError(f"Unknown views {unknown}; expected keys of HEMI_VIEWS {list(HEMI_VIEWS)}.")
    roi_by_view = {}
    for roi in rois:
        if roi["view"] not in views:
            raise ValueError(f"ROI view {roi['view']!r} is not among the drawn views {list(views)}.")
        if roi["view"] in roi_by_view:
            raise ValueError(f"More than one ROI on view {roi['view']!r}.")
        roi_by_view[roi["view"]] = roi

    values = np.asarray(values, dtype=float)
    n_lh = surf_lh.n_points
    surfs = {"lh": surf_lh, "rh": surf_rh}
    hemi_slice = {"lh": slice(0, n_lh), "rh": slice(n_lh, n_lh + surf_rh.n_points)}
    if color_range is None:
        color_range = (float(np.nanmin(values)), float(np.nanmax(values)))

    # Canvas geometry in pt (= plot_hemispheres window pixels), y pointing down:
    # [title band] + [brain row, starting at y0] + gap + inset row + label band.
    width, height = size
    view_w = width * (1 - _CBAR_FRAC if color_bar else 1) / len(views)
    view_px = int(round(view_w))  # each view is rendered at this width, centred on its viewport
    radius_in = inset_frac * view_w / 2
    n_lines = max((str(r.get("label", "")).count("\n") + 1 for r in rois if r.get("label")), default=0)
    label_h = 1.4 * n_lines * label_fontsize + 0.3 * label_fontsize if n_lines else 0
    y0 = 1.6 * title_fontsize if title else 0
    top = y0 + (0 if roi_only else height)
    inset_cy = top + gap + radius_in
    fig_h = inset_cy + radius_in + label_h + line_width

    fig = plt.figure(figsize=(width / 72, fig_h / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(fig_h, 0)
    ax.set_axis_off()

    content_x0 = width  # left edge of the leftmost drawn brain or inset (title anchor)
    for i, v in enumerate(views):
        if roi_only and v not in roi_by_view:
            continue
        hemi, cam_view = HEMI_VIEWS[v]
        sl = hemi_slice[hemi]
        img, xy, visible, px_per_mm = render_surface_view(
            surfs[hemi], values[sl], cam_view, size=(view_px, height), scale=scale, zoom=zoom,
            cmap=cmap, color_range=color_range, nan_color=nan_color)
        if outline is not None:
            # Darken the render by the outline's opacity, before any inset is cropped.
            alpha = _outline_alpha(surfs[hemi], np.asarray(outline, dtype=bool)[sl], cam_view,
                                   size=(view_px, height), scale=scale, zoom=zoom)[..., None]
            rgb = img[..., :3].astype(float) * (1 - alpha) + 255 * np.array(to_rgb(outline_color)) * alpha
            img = np.concatenate([np.round(rgb).astype(img.dtype), img[..., 3:]], axis=-1)
        col_cx = (i + 0.5) * view_w
        x_off = col_cx - view_px / 2  # canvas x of the render's left edge
        if not roi_only:
            ax.imshow(img, extent=(x_off, x_off + view_px, y0 + height, y0), aspect="auto",
                      interpolation="antialiased")
            opaque_cols = np.flatnonzero(img[..., 3].max(axis=0) > 0)
            if opaque_cols.size:
                content_x0 = min(content_x0, x_off + opaque_cols[0] / scale)
        if v not in roi_by_view:
            continue

        # Zoom window: circle around the region's camera-facing vertices (canvas units).
        roi = roi_by_view[v]
        name = (roi.get("label") or v).replace("\n", " ")
        sel = np.asarray(roi["mask"], dtype=bool)[sl] & visible
        if not sel.any():
            raise ValueError(f"ROI {name!r} has no vertex facing the camera in view {v!r}.")
        pts = np.column_stack([x_off + xy[sel, 0] / scale, y0 + xy[sel, 1] / scale])
        cx, cy = (pts.min(axis=0) + pts.max(axis=0)) / 2
        if roi.get("radius") is not None:
            r = roi["radius"] * px_per_mm / scale
        else:
            r = roi_pad * np.max(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy))

        # Inset centred under its column; shifted only if the vertical line from the
        # marker circle would otherwise miss it.
        max_dx = 0.85 * radius_in
        inset_cx = float(np.clip(col_cx, cx - max_dx, cx + max_dx))
        if inset_cx != col_cx:
            logger.info(f"[zoom] {v}: inset shifted {inset_cx - col_cx:+.0f} pt so the line meets it")
        logger.info(f"[zoom] {v}: {name} -> {int(sel.sum())} visible vertices, "
                    f"window radius {r * scale / px_per_mm:.1f} mm, magnification x{radius_in / r:.1f}")

        # Inset: crop of the supersampled render around the window, magnified onto
        # the inset circle (crop bounds snapped to whole render pixels).
        m = radius_in / r
        cx_px, cy_px, r_px = (cx - x_off) * scale, (cy - y0) * scale, r * scale
        x_lo, x_hi = int(np.floor(cx_px - r_px)), int(np.ceil(cx_px + r_px))
        y_lo, y_hi = int(np.floor(cy_px - r_px)), int(np.ceil(cy_px + r_px))
        extent = (inset_cx + m * (x_lo - cx_px) / scale, inset_cx + m * (x_hi - cx_px) / scale,
                  inset_cy + m * (y_hi - cy_px) / scale, inset_cy + m * (y_lo - cy_px) / scale)
        ax.add_patch(Circle((inset_cx, inset_cy), radius_in, facecolor=inset_facecolor,
                            edgecolor="none", zorder=1))
        im = ax.imshow(_crop(img, x_lo, x_hi, y_lo, y_hi), extent=extent, aspect="auto",
                       interpolation="antialiased", zorder=2)
        im.set_clip_path(Circle((inset_cx, inset_cy), radius_in, transform=ax.transData))
        ax.add_patch(Circle((inset_cx, inset_cy), radius_in, fill=False, edgecolor=line_color,
                            linewidth=line_width, zorder=3))
        content_x0 = min(content_x0, inset_cx - radius_in - line_width / 2)  # outer edge of outline
        if roi.get("label"):
            ax.text(inset_cx, inset_cy + radius_in + 0.3 * label_fontsize, roi["label"],
                    ha="center", va="top", fontsize=label_fontsize)

        if not roi_only:
            # Marker circle on the brain + vertical line down to the inset outline.
            ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=line_color,
                                linewidth=line_width, zorder=3))
            y_end = inset_cy - np.sqrt(radius_in ** 2 - (cx - inset_cx) ** 2)
            ax.plot([cx, cx], [cy + r, y_end], color=line_color, linewidth=line_width,
                    solid_capstyle="butt", zorder=3)

    if title:
        # Shift by the first glyph's left side bearing so the visible ink, not the text
        # origin, starts at the content edge.
        font = get_font(findfont(FontProperties()))
        font.set_size(title_fontsize, 72)  # 72 dpi: 1 px = 1 pt
        bearing = font.load_char(ord(title.lstrip()[0])).horiBearingX / 64
        ax.text((content_x0 if content_x0 < width else 0) - bearing, y0 - 0.35 * title_fontsize, title,
                ha="left", va="bottom", fontsize=title_fontsize)

    if color_bar and (not roi_only if draw_color_bar is None else draw_color_bar):
        # Same box as brainspace's scalar bar inside its column, centred on the drawn row.
        col_x0, col_w = len(views) * view_w, width - len(views) * view_w
        bx0, bw = col_x0 + _CBAR_BOX[0] * col_w, _CBAR_BOX[1] * col_w
        bar_cy = inset_cy if roi_only else y0 + height / 2
        half_h = (_CBAR_BOX[3] - _CBAR_BOX[2]) * height / 2
        by0, by1 = bar_cy - half_h, bar_cy + half_h
        cax = fig.add_axes([bx0 / width, 1 - by1 / fig_h, bw / width, (by1 - by0) / fig_h])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(*color_range), cmap=cmap), cax=cax, ticks=[])
        cb.outline.set_visible(False)
        low, high = cbar_labels if cbar_labels else [f"{c:g}" for c in color_range]
        pad = 0.25 * label_fontsize
        ax.text(bx0 + bw / 2, by0 - pad, high, ha="center", va="bottom", fontsize=label_fontsize)
        ax.text(bx0 + bw / 2, by1 + pad, low, ha="center", va="top", fontsize=label_fontsize)
        if cbar_title:
            ax.text(bx0 + bw + pad, bar_cy, cbar_title, rotation=90, ha="left", va="center",
                    fontsize=label_fontsize)

    fig.inset_cy = inset_cy  # inset-row centre, pt from the top (for aligning companion panels)
    if filename is not None:
        fig.savefig(filename, dpi=dpi, transparent=True)
    return fig


def format_pvalue(value: float) -> str:
    """p/q value to 3 decimals, with more digits where rounding would cross 0.05.

    E.g. an add-one q of 50/1001 = 0.04995 prints as '0.04995' rather than '0.050',
    so the printed value agrees with a p < 0.05 significance marker.
    """
    text = f"{value:.3f}"
    if (value < 0.05) != (float(text) < 0.05):
        text = f"{value:.4g}"
    return text


def plot_forest(
    stats: pd.DataFrame,
    inset_cy: float,
    row_h: float,
    filename=None,
    *,
    row_width: float = 1450,
    xlim: tuple[float, float] = (-0.4, 0.4),
    xlabel: str = "Spearman r",
) -> plt.Figure:
    """Forest plot whose rows sit level with stacked ``plot_hemispheres_zoom`` inset rows.

    One row per ``stats`` row, top to bottom: the observed correlation (black dot,
    filled if its uncorrected Moran ``p`` < 0.05) over the central 95 % of its null
    distribution (grey bar from ``null_lo`` to ``null_hi``, tick at ``null_med``), with r
    and the uncorrected Moran p (labelled $p_m$) written above; FDR ``q`` values, where
    the table has them, are not drawn. Drawn at its
    final size on the ``figure_1b_forest.svg`` canvas (8.8-pt text). Row k is centred
    level with the insets of the k-th inset row when the rows (each ``row_h`` pt tall,
    inset centre ``inset_cy`` pt from its top, ``row_width`` pt wide) are stacked and
    shown ``ZOOM_ROW_WIDTH_MM`` wide, with the forest's top at the first row's top.

    Args:
        stats: One row per test with columns ``r``, ``p`` (Moran permutation p),
            ``null_lo``, ``null_med`` and ``null_hi``; a row with NaN ``r`` leaves an
            empty slot.
        inset_cy: Inset-row centre (pt from the top) of the rows, the ``inset_cy``
            attribute of a ``plot_hemispheres_zoom`` figure.
        row_h: Height (pt) of each inset row figure.
        filename: Output path; the figure is only returned if ``None``.
        row_width: Width (pt) of the inset rows.
        xlim: Signed correlation axis (ticks at its ends and 0).
        xlabel: Axis label.

    Returns:
        matplotlib.figure.Figure: The forest figure.
    """
    s = ZOOM_ROW_WIDTH_MM / 25.4 * 72 / row_width  # display scale of the inset rows
    axes_h = len(stats) * row_h * s
    fig_w, fig_h = _FOREST_WIDTH_PT, axes_h + _FOREST_BOTTOM_PT
    (x0, x1), half = _FOREST_AXIS_PT, _FOREST_BAR_PT / 2
    with plt.rc_context({'font.size': _FOREST_FONT_PT}):
        fig = plt.figure(figsize=(fig_w / 72, fig_h / 72))
        ax = fig.add_axes([x0 / fig_w, _FOREST_BOTTOM_PT / fig_h, (x1 - x0) / fig_w, axes_h / fig_h])
        ax.set_xlim(*xlim)
        ax.set_ylim(axes_h, 0)  # pt from the top of the first row
        ax.axvline(0, color='#999999', lw=0.8, ls='--', zorder=0)
        for k, row in enumerate(stats.itertuples(index=False)):
            yc = (k * row_h + inset_cy) * s
            if np.isfinite(row.null_lo):
                ax.add_patch(Rectangle((row.null_lo, yc - half), row.null_hi - row.null_lo, 2 * half,
                                       facecolor='#d9d9d9', edgecolor='none', zorder=1))
                ax.plot([row.null_med] * 2, [yc - half, yc + half], color='#8c8c8c', lw=0.8, zorder=2)
            if not np.isfinite(row.r):
                continue
            ax.plot(row.r, yc, 'o', ms=6, mew=1.2, mec='#1a1a1a', mfc='#1a1a1a' if row.p < 0.05 else 'white',
                    zorder=3)
            ax.text(xlim[0], yc - half - 2.5, f"r = {row.r:+.2f}".replace('-', '\u2212')
                    + f"\n$p_{{m}}$ = {format_pvalue(row.p)}", ha='left', va='bottom')
        ax.set_yticks([])
        ax.set_xticks([xlim[0], 0, xlim[1]],
                      [f"{xlim[0]:g}".replace('-', '\u2212'), '0', f"{xlim[1]:g}".replace('-', '\u2212')])
        ax.set_xlabel(xlabel)
        for side in ('left', 'right', 'top'):
            ax.spines[side].set_visible(False)
        if filename is not None:
            fig.savefig(filename, transparent=True)
    return fig
