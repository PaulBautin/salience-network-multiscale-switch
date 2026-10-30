#!/usr/bin/env python
"""Step 2 schematic: a visual key to the Figure 3 projection (Figure 3, panel A).

Shows how the connectivity-weighted projection P is built for each salience-network (SN)
vertex and compared with the SN microstructure gradient, from one representative subject
(the subject whose structural-connectivity r_s is closest to the group median) with the
exact Figure 3 SC weights: SIFT2 streamlines filtered by the Betzel group-consensus mask
and log-scaled (prepare_weights), the FC gradient oriented with the default-mode network
low, and the fsLR-5k MPC gradient of step 1a. Steps, left to right:
  1. the strongest extranetwork structural connections of an example SN vertex a (low
     MPC gradient), over the FC gradient;
  2. the connectivity-weighted projection for a: its grouped targets, the formula and
     its terms;
  3. P on the SN as a circular close-up, as the panel B insets;
  4. P against the MPC gradient across SN vertices (square scatter, fit line, r_s).
The canvas is the Figure 3 page width (210 mm), so the schematic (panel A of
results/figure3.svg) stacks on top of the figure_3a rows; step 1 sits on the first
figure_3a inset column, step 4 is centred on the figure_3a lollipop axis, and the steps
are spaced by three equal gaps, each holding one arrow. Its colour bars follow the
Figure 3 inset rows: numbers at the ends, title apart.

Inputs   step 1a: the fsLR-5k subject list (private) and MPC gradient (source_data); the
         structural connectomes it lists.
Private  <work_dir>/step3_schematic/: the example subject's ID and every subject's r_s, log.
Public   source_data/step3_schematic/: the example subject's projection on the SN
         vertices, the connections of example vertex a, a summary (no subject ID),
         manifest.
Figures  results/figures/figure_3_schematic_<hemisphere>.svg

Run      python pipeline/step3_schematic.py [--stage compute|plot|all]
"""
import json
import logging

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath

import common
from salience_switch import config
from salience_switch.colors import connectivity_cmap
from salience_switch.connectivity import build_consensus_mask, compute_projection_subjects, prepare_weights
from salience_switch.plotting import LOLLI_AXIS_PT, SALIENCE_Z_RANGE, mask_outline, outline_alpha, render_surface_view
from salience_switch.provenance import write_manifest
from salience_switch.surface import compute_network_mask, load_fc_gradient_5k, load_inflated_surfaces, load_yeo_surf_5k

STEP = "step3_schematic"
GRADIENT_STEP = "step1a_mpc_gradient"
logger = logging.getLogger(STEP)

plt.rcParams['svg.fonttype'] = 'none'

NETWORK = "SalVentAttn"

# Figure 3 page geometry (mm) from results/figure3.svg: the panel B inset rows (1450-pt
# windows shown 141.91 mm wide from x = 0.64) split their colour-bar-free 89 % into four
# 31.58-mm view columns, and the panel B lollipop canvas (177.8 pt shown 68.69 mm wide
# from x = 135.82) puts its axis at LOLLI_AXIS_PT.
PAGE_W_MM, PAGE_H_MM = 210.0, 40.0
VIEW_COLUMNS_MM = [(0.642 + 31.575 * k, 0.642 + 31.575 * (k + 1)) for k in range(4)]
LOLLI_AXIS_MM = tuple(135.822 + x * 68.688072 / 177.845669 for x in LOLLI_AXIS_PT)
FONT_PT, HEADER_PT = 7.5, 8.5
CBAR_PT = 8.0  # colour-bar end labels, the size of the Figure 3 inset-row labels as shown
N_EDGES = 40  # strongest connections drawn per example vertex (of those facing the camera)
OUTLINE_SHARPNESS = 3.0  # SN border trace, as the Figure 3 insets (step3_connectivity.ZOOM_KW)
# Render of the example-vertex brain (step 1), shared by the compute and plot stages
RENDER = dict(size=(400, 300), scale=6)


def _table_paths(hemisphere: str) -> dict:
    """Public tables of the schematic, and the private record of the example subject."""
    source = config.source_dir(STEP)
    return {"vertex": source / f"schematic_vertex_{hemisphere}.tsv",
            "edges": source / f"schematic_edges_{hemisphere}.tsv",
            "summary": source / f"schematic_summary_{hemisphere}.tsv"}


def compute_schematic_data(paths: dict, hemisphere: str) -> None:
    """Per-subject SC projection of the SN, and the example subject's SN connections.

    Reproduces the Figure 3 SC inputs (``step3_connectivity.run_compute_stage``): the
    fsLR-5k surface table with the step 1a MPC gradient and the oriented FC gradient,
    the source (SN, finite gradient) and target (cortex outside the SN) masks, the Betzel
    consensus mask over all subjects, and ``compute_projection_subjects``. The example
    subject is the one whose r_s is closest to the median r_s, and the example vertex a
    is picked by `_pick_examples` among the vertices facing the step-1 camera. Writes,
    to source_data, ``schematic_vertex_<hemi>.tsv`` (per SN vertex: MPC gradient, the
    example subject's P and target count, the group-mean P), ``schematic_edges_<hemi>.tsv``
    (every weight of vertex a that enters its P, with the target's FC gradient) and
    ``schematic_summary_<hemi>.tsv`` (vertex a, the example r_s, the median r_s, N); and
    to the work dir the example subject's ID and every subject's r_s.
    """
    df_pni = pd.read_csv(config.work_dir(GRADIENT_STEP, paths) / "subjects_fslr5k.csv")
    df = load_yeo_surf_5k()
    col = f"t1_gradient1_{NETWORK}"
    cache = config.source_dir(GRADIENT_STEP) / f"surface_fslr5k_{hemisphere}.tsv"
    if not cache.exists():
        raise FileNotFoundError(f"{cache} not found. Run step 1a with hemisphere {hemisphere} first.")
    cached = config.read_table(cache)
    if len(cached) != len(df) or col not in cached:
        raise ValueError(f"{cache.name} does not hold the fsLR-5k {col} column.")
    df[col] = cached[col].to_numpy()
    df = load_fc_gradient_5k(df)

    # Source / target masks as step3_connectivity._prepare_network_gradient.
    cortex = df["hemisphere"].notna().to_numpy()
    cortex_idx = np.flatnonzero(cortex)
    grad = df[col].to_numpy(dtype=float)[cortex]
    sn = compute_network_mask(df, NETWORK, hemisphere)[cortex] & np.isfinite(grad)
    other = df["network"].ne(NETWORK).to_numpy()
    if hemisphere in ("LH", "RH"):
        other &= df["hemisphere"].eq(hemisphere).to_numpy()
    other = other[cortex]
    g_fc = df["fc_g1"].to_numpy(dtype=float)[cortex]

    logger.info("Building the Betzel consensus mask and loading every subject's SC...")
    mask_G, sc_subjects = build_consensus_mask(df_pni["path_sc_5k"].tolist(), df_pni["path_sc_dist_5k"].tolist(),
                                               df, nbins=10)
    res = compute_projection_subjects(files=df_pni["path_sc_5k"].tolist(), modality="SC", g_fc_cortex=g_fc,
                                      g_mpc_cortex_at_sn=grad[sn], sn_mask_cortex=sn, other_mask_cortex=other,
                                      df_yeo_surf_5k=df, mask_G=mask_G, preloaded_subjects=sc_subjects)
    r_s = np.asarray(res["r_subjects"], dtype=float)
    s = int(np.nanargmin(np.abs(r_s - np.nanmedian(r_s))))
    subject = df_pni["ID_PNI"].iloc[s]
    logger.info(f"SC r_s across {np.isfinite(r_s).sum()} subjects: median {np.nanmedian(r_s):+.3f}; "
                f"example subject {subject} (r_s = {r_s[s]:+.3f}; private)")

    # The example subject's weights entering P: SN rows, target columns, finite and
    # positive, at targets with a finite FC gradient (as _weighted_mean_projection).
    W = prepare_weights(sc_subjects[s], "SC", df.loc[cortex, "hemisphere"].to_numpy(), sn, mask_G=mask_G)
    del sc_subjects
    sub = W[np.ix_(sn, other)]
    tgt = np.flatnonzero(other)
    rows, cols = np.nonzero(np.isfinite(sub) & (sub > 0) & np.isfinite(g_fc[tgt])[None, :])
    sn_full = cortex_idx[sn]
    edges = pd.DataFrame({"source": sn_full[rows], "target": cortex_idx[tgt[cols]],
                          "w": sub[rows, cols], "g_fc": g_fc[tgt[cols]]})
    P_s = res["P_subjects_sn"][s]
    # Self-check: the weighted mean of the written edges is the subject's P.
    check = edges.assign(wg=edges["w"] * edges["g_fc"]).groupby("source")[["wg", "w"]].sum()
    P_edges = pd.Series(check["wg"] / check["w"]).reindex(sn_full).to_numpy()
    ok = np.isfinite(P_s)
    if not np.allclose(P_edges[ok], P_s[ok], rtol=1e-6, atol=1e-9):
        raise RuntimeError("Edge table does not reproduce the subject's projection P.")

    vertex = pd.DataFrame({"vertex": sn_full, "hemisphere": df.loc[cortex, "hemisphere"].to_numpy()[sn],
                           "g_mpc": grad[sn], "P": P_s, "n_targets": np.bincount(rows, minlength=sn.sum()),
                           "P_mean": res["P_mean"]})
    # Example vertex a, among the left-hemisphere vertices facing the step-1 camera
    # (visibility depends on the surface and camera only, not on the values drawn)
    surf_lh, _ = load_inflated_surfaces("fsLR-5k")
    n_lh = surf_lh.n_points
    _, _, visible, _ = render_surface_view(surf_lh, np.zeros(n_lh), "lateral", cmap=connectivity_cmap.name,
                                           color_range=SALIENCE_Z_RANGE, **RENDER)
    example = _pick_examples(vertex, visible, n_lh)["a"]

    tables = _table_paths(hemisphere)
    config.write_table(vertex, tables["vertex"])
    # Weights at 6 significant digits, as drawn
    config.write_table(edges[edges["source"] == example], tables["edges"], float_format="%.6g")
    config.write_table(pd.DataFrame([{"network": NETWORK, "hemisphere": hemisphere, "measure": "SC",
                                      "example_vertex": example, "r_s": float(r_s[s]),
                                      "r_s_median": float(np.nanmedian(r_s)),
                                      "n_subjects": int(np.isfinite(r_s).sum())}]), tables["summary"])
    private = config.work_dir(STEP, paths) / f"example_subject_{hemisphere}.json"
    private.write_text(json.dumps({"subject": subject, "subjects": df_pni["ID_PNI"].tolist(),
                                   "r_subjects": [float(v) for v in r_s]}, indent=1))
    write_manifest(config.source_dir(STEP) / "manifest.json", STEP, config.PROJECT_ROOT, network=NETWORK,
                   hemisphere=hemisphere, measure="SC", n_subjects=int(np.isfinite(r_s).sum()))
    logger.info(f"Schematic tables written: {', '.join(p.name for p in tables.values())} "
                f"({len(vertex)} SN vertices, {int((edges['source'] == example).sum())} edges of vertex a); "
                f"example subject recorded in {private}")


def _mm_axes(fig: plt.Figure, x0: float, y0: float, w: float, h: float) -> plt.Axes:
    """Axes of w x h mm at (x0, y0) mm from the bottom-left corner of the page."""
    return fig.add_axes([x0 / PAGE_W_MM, y0 / PAGE_H_MM, w / PAGE_W_MM, h / PAGE_H_MM])


def _brain(fig: plt.Figure, surf, values: np.ndarray, column: tuple[float, float], y0: float, *,
           cmap, color_range: tuple[float, float], frame: np.ndarray | None = None,
           height: float | None = None, outline: np.ndarray | None = None,
           width_frac: float = 0.84) -> tuple[plt.Axes, np.ndarray, np.ndarray]:
    """Lateral left-hemisphere render cropped to the brain and centred in a view column.

    With ``frame`` (vertex indices), the view is instead a circular close-up, in the
    style of the ``plot_hemispheres_zoom`` insets, centred on those of the vertices
    facing the camera with a 12 % margin around them. The image is ``width_frac`` of the
    column wide, or ``height`` mm tall if that is narrower. ``outline`` (one bool per vertex,
    e.g. ``mask_outline`` of the SN) is traced in black as on the Figure 3 insets
    (``OUTLINE_SHARPNESS``). Returns the axes (data units = image pixels) and the pixel
    position and camera visibility of every vertex.
    """
    img, xy, visible, _ = render_surface_view(surf, values, "lateral", cmap=cmap, color_range=color_range,
                                              **RENDER)
    if outline is not None:
        alpha = outline_alpha(surf, outline, "lateral", sharpness=OUTLINE_SHARPNESS, **RENDER)[..., None]
        img = img.copy()
        img[..., :3] = (img[..., :3] * (1 - alpha)).astype(np.uint8)
    if frame is None:
        rows = np.flatnonzero((img[..., 3] > 0).any(axis=1))
        cols = np.flatnonzero((img[..., 3] > 0).any(axis=0))
        (r0, r1), (c0, c1) = (rows[0], rows[-1] + 1), (cols[0], cols[-1] + 1)
    else:
        pts = xy[frame[visible[frame]]]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        half = int(0.5 * 1.24 * (hi - lo).max())
        img = np.pad(img, ((half, half), (half, half), (0, 0)))  # transparent margin for the square
        c0, r0 = ((lo + hi) / 2).astype(int)  # square of side 2 * half centred on the region (padded coords)
        r1, c1 = r0 + 2 * half, c0 + 2 * half
        xy = xy + half
    img = img[r0:r1, c0:c1]
    xy = xy - [c0, r0]
    x0, x1 = column
    w = width_frac * (x1 - x0)
    if height is not None:
        w = min(w, height * img.shape[1] / img.shape[0])
    ax = _mm_axes(fig, x0 + (x1 - x0 - w) / 2, y0, w, w * img.shape[0] / img.shape[1])
    im = ax.imshow(img, aspect="auto", interpolation="lanczos")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    if frame is not None:
        r = img.shape[0] / 2
        im.set_clip_path(Circle((r, r), r, transform=ax.transData))
        ax.add_patch(Circle((r, r), r - 1, facecolor="none", edgecolor="#1a1a1a", lw=0.6, zorder=5))
    return ax, xy, visible


def _colorbar(fig: plt.Figure, centre: float, y0: float, cmap, title: str, width: float = 15.8,
              labels: tuple[str, str] = ("\u22122", "+2")) -> None:
    """Horizontal colour bar (``width`` x 1.2 mm) centred at x = ``centre``, in the style of
    the Figure 3 inset rows: ``labels`` (low, high) at its two ends, ``title`` under it."""
    ax = _mm_axes(fig, centre - width / 2, y0, width, 1.2)
    ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, aspect="auto")
    ax.set_axis_off()
    kw = dict(transform=ax.transAxes, va="center", fontsize=CBAR_PT, color="#1a1a1a")
    ax.text(-0.06, 0.5, labels[0], ha="right", **kw)
    ax.text(1.06, 0.5, labels[1], ha="left", **kw)
    ax.text(0.5, -1.1, title, ha="center", **dict(kw, va="top", color="#4d4d4d", fontsize=FONT_PT - 1))


def _arc(ax: plt.Axes, p: np.ndarray, q: np.ndarray, color, lw: float, alpha: float) -> None:
    """Quadratic arc from p to q (image pixels), bowed upwards."""
    d = q - p
    perp = np.array([d[1], -d[0]])
    if perp[1] > 0:  # image rows grow downwards: bow towards the top of the brain
        perp = -perp
    path = MplPath([p, (p + q) / 2 + 0.22 * perp, q], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=lw, alpha=alpha, capstyle="round"))


def _arrow(fig: plt.Figure, x0: float, x1: float, y: float) -> None:
    """Grey arrow between two steps, at height y (mm)."""
    fig.add_artist(FancyArrowPatch((x0 / PAGE_W_MM, y / PAGE_H_MM), (x1 / PAGE_W_MM, y / PAGE_H_MM),
                                   transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
                                   lw=0.8, color="#8c8c8c"))


def _pick_examples(vertex: pd.DataFrame, visible: np.ndarray, n_lh: int) -> dict[str, int]:
    """Example SN vertex a: of the left-hemisphere vertices facing the lateral camera with
    at least 20 targets, the one nearest the 5th percentile of the gradient among them."""
    cand = vertex[(vertex["vertex"] < n_lh) & np.isfinite(vertex["P"]) & (vertex["n_targets"] >= 20)]
    cand = cand[visible[cand["vertex"].to_numpy()]]
    lo = np.percentile(cand["g_mpc"], 5)
    return {"a": int(cand.loc[(cand["g_mpc"] - lo).abs().idxmin(), "vertex"])}


def plot_schematic(hemisphere: str) -> None:
    """Draw ``figure_3_schematic_{hemi}.svg``, a visual key to Figure 3 panel B (210 x 40 mm).

    Steps (left to right): (1) on the fsLR-5k inflated left hemisphere (lateral view) of
    the example subject, the ``N_EDGES`` strongest structural connections, among the
    targets facing the camera, of the example SN vertex a (``_pick_examples``),
    drawn over the whole-brain FC gradient (z-scored over the cortex,
    ``CustomCmap_connectivity`` on the shared +/-2 SD scale; the SN, excluded from the
    targets, in grey); (2) the projection for a: a's targets grouped into four equal
    ranges of the FC gradient, each drawn as a dot coloured by the group's weighted-mean
    position (same scale) and linked to a by a line as wide as the group's total weight,
    a's dot coloured by their weighted average, which is a's P, and the formula of P
    beside them with its terms below; (3) P on the SN (z-scored over the SN, connectivity colour map, +/-2 SD)
    as a circular close-up, as the panel B insets; (4) P against the MPC gradient
    (coolwarm) over all SN vertices with its least-squares line and the subject's
    (Spearman) r_s, as a square centred on the panel B lollipop axis. The steps are spaced
    by three equal gaps (the formula and a's label measured first), each holding one arrow.
    Colour bars carry -2 / +2 at their ends and their title underneath.
    """
    tables = _table_paths(hemisphere)
    vertex = config.read_table(tables["vertex"])
    edges = config.read_table(tables["edges"])
    summary = config.read_table(tables["summary"]).iloc[0]
    df = load_fc_gradient_5k(load_yeo_surf_5k())
    surf, surf_rh = load_inflated_surfaces("fsLR-5k")
    n_lh = surf.n_points
    sn_outline = mask_outline(surf, surf_rh, df["network"].eq(NETWORK).to_numpy())[:n_lh]

    ok = np.isfinite(vertex["P"])
    vertex["g_z"] = (vertex["g_mpc"] - vertex["g_mpc"].mean()) / vertex["g_mpc"].std(ddof=0)
    vertex["P_z"] = (vertex["P"] - vertex.loc[ok, "P"].mean()) / vertex.loc[ok, "P"].std(ddof=0)
    by_vertex = vertex.set_index("vertex")
    mpc_cmap, z_norm = plt.get_cmap("coolwarm"), Normalize(*SALIENCE_Z_RANGE)
    g_fc = df["fc_g1"].to_numpy(dtype=float)
    fc_mean, fc_sd = np.nanmean(g_fc), np.nanstd(g_fc)
    fc_z = (g_fc - fc_mean) / fc_sd  # displayed z-scored over the cortex, on the shared +/-2 SD scale
    sn_lh = vertex.loc[vertex["vertex"] < n_lh, "vertex"].to_numpy()
    grey = "#4d4d4d"
    halo = [patheffects.withStroke(linewidth=2.0, foreground="white")]  # labels over the SN border

    def _surface(column: str) -> np.ndarray:
        values = np.full(n_lh, np.nan)
        values[sn_lh] = by_vertex.loc[sn_lh, column].to_numpy()
        return values

    def _mark(ax: plt.Axes, xy: np.ndarray, colors: dict) -> None:
        for name, v in examples.items():
            ax.plot(*xy[v], "o", ms=5, mfc=colors[name], mec="#1a1a1a", mew=0.9, zorder=4)
            ax.annotate(name, xy[v], xytext=(0, 4.5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=FONT_PT, fontweight="bold", zorder=4, path_effects=halo)

    def _text(x: float, y: float, s: str, **kw):
        kw = dict(dict(ha="center", va="center", fontsize=FONT_PT - 1, color=grey), **kw)
        return fig.text(x / PAGE_W_MM, y / PAGE_H_MM, s, **kw)

    def _dot(x: float, y: float, radius: float, **kw) -> None:
        """Round dot of ``radius`` mm at (x, y) mm on the page."""
        fig.add_artist(Ellipse((x / PAGE_W_MM, y / PAGE_H_MM), 2 * radius / PAGE_W_MM, 2 * radius / PAGE_H_MM,
                               transform=fig.transFigure, zorder=3, **kw))

    with plt.rc_context({"font.size": FONT_PT}):
        fig = plt.figure(figsize=(PAGE_W_MM / 25.4, PAGE_H_MM / 25.4))
        col = VIEW_COLUMNS_MM
        y_title, y_sub = 37.4, 34.4

        # 1. Connections of a to the targets outside the SN, over the FC gradient.
        fc_lh = fc_z[:n_lh].copy()
        fc_lh[df["network"].to_numpy()[:n_lh] == NETWORK] = np.nan
        ax1, xy, visible = _brain(fig, surf, fc_lh, col[0], 11.0, cmap=connectivity_cmap.name,
                                  color_range=SALIENCE_Z_RANGE, outline=sn_outline, width_frac=0.92)
        box = ax1.get_position()
        y_c = (box.y0 + box.height / 2) * PAGE_H_MM  # centre line of every step
        examples = {"a": int(summary["example_vertex"])}
        if examples != _pick_examples(vertex, visible, n_lh):
            raise RuntimeError("The example vertex of the summary table is not the one this view picks.")
        for v in examples.values():
            e = edges[(edges["source"] == v) & (edges["target"] < n_lh)]
            e = e[visible[e["target"].to_numpy()]].nlargest(N_EDGES, "w")
            for t, w in zip(e["target"], e["w"]):
                _arc(ax1, xy[v], xy[t], "#1a1a1a", lw=0.25 + 0.7 * w / e["w"].max(), alpha=0.6)
        _mark(ax1, xy, {name: "white" for name in examples})
        _colorbar(fig, sum(col[0]) / 2, 6.6, connectivity_cmap, r"FC gradient $g^{\mathrm{FC}}_j$ (z)")

        # Horizontal layout: the square scatter centred on the panel B lollipop axis, and the
        # brain, the projection and the map spaced before it by three equal gaps, each
        # holding one arrow with the same clearance at both ends. The formula and a's label
        # are measured first.
        renderer = fig.canvas.get_renderer()

        def _width(artist) -> float:
            return artist.get_window_extent(renderer).width / fig.dpi * 25.4

        formula = _text(0.0, y_c, r"$P_i = \frac{\Sigma_j\, w_{ij}\, g^{\mathrm{FC}}_j}{\Sigma_j\, w_{ij}}$",
                        fontsize=10.5, color="#1a1a1a")
        a_label = _text(0.0, y_c, "a", ha="right", fontsize=FONT_PT + 1.0, fontweight="bold", color="#1a1a1a")
        label_gap, spread, formula_gap = 3.0, 14.5, 10.5  # a's label -> a, a -> targets, targets -> formula
        diameter, side, ylabel_w, clearance = 25.0, 27.0, 4.5, 1.5
        proj_w = _width(a_label) + label_gap + spread + formula_gap + _width(formula)
        brain_right = box.x1 * PAGE_W_MM
        ax_x0 = sum(LOLLI_AXIS_MM) / 2 - side / 2
        ax_x1 = ax_x0 + side
        gap = (ax_x0 - ylabel_w - brain_right - proj_w - diameter) / 3
        proj_x0 = brain_right + gap
        proj_x1 = proj_x0 + proj_w
        src_x = proj_x0 + _width(a_label) + label_gap
        tgt_x = src_x + spread
        a_label.set_x((src_x - label_gap) / PAGE_W_MM)
        formula.set_x((tgt_x + formula_gap + _width(formula) / 2) / PAGE_W_MM)
        map_cx = proj_x1 + gap + diameter / 2

        # 2. The projection for vertex a: a's targets grouped into four equal ranges of the
        # FC gradient (dot = the group's weighted-mean position, line width = its total
        # weight), a's dot coloured by their weighted average, exactly P_a; the formula
        # beside them and its terms below.
        e = edges[edges["source"] == examples["a"]]
        group = np.clip(np.digitize(e["g_fc"], np.linspace(np.nanmin(g_fc), np.nanmax(g_fc), 5)) - 1, 0, 3)
        weight = np.bincount(group, weights=e["w"], minlength=4)
        position = np.bincount(group, weights=e["w"] * e["g_fc"], minlength=4) / np.where(weight > 0, weight, np.nan)
        P_a = float(by_vertex.loc[examples["a"], "P"])
        used = weight > 0
        if not np.isclose(np.sum(weight[used] * position[used]) / np.sum(weight[used]), P_a):
            raise RuntimeError("The grouped targets of vertex a do not average to its P.")
        tgt_y = y_c + np.array([7.8, 2.6, -2.6, -7.8])  # DMN end on top
        for y, w in zip(tgt_y[used], weight[used]):
            fig.add_artist(plt.Line2D([src_x / PAGE_W_MM, tgt_x / PAGE_W_MM], [y_c / PAGE_H_MM, y / PAGE_H_MM],
                                      transform=fig.transFigure, color="#1a1a1a", lw=0.6 + 3.0 * w / weight.max(),
                                      solid_capstyle="round", zorder=2))
        for y, g in zip(tgt_y[used], position[used]):
            _dot(tgt_x, y, 1.7, facecolor=connectivity_cmap(z_norm((g - fc_mean) / fc_sd)), edgecolor="#1a1a1a",
                 lw=0.7)
        _dot(src_x, y_c, 2.3, facecolor=connectivity_cmap(z_norm((P_a - fc_mean) / fc_sd)),
             edgecolor="#1a1a1a", lw=0.9)
        _text((src_x + tgt_x) / 2 - 0.6, y_c + 6.6, "$w_{aj}$", ha="right", fontsize=FONT_PT + 1.0,
              color="#1a1a1a")
        _text(tgt_x + 2.5, tgt_y[0], r"$g^{\mathrm{FC}}_j$", ha="left", fontsize=FONT_PT + 1.0, color="#1a1a1a")
        proj_cx = (proj_x0 + proj_x1) / 2
        _text(proj_cx, y_c - 12.6, r"$w_{ij}$: connection strength of SN vertex $i$ to target $j$")
        _text(proj_cx, y_c - 15.0, r"$g^{\mathrm{FC}}_j$: position of target $j$ on the FC gradient")

        # 3. Projected map on the SN, as the panel B insets.
        ax3, xy3, _ = _brain(fig, surf, _surface("P_z"), (map_cx - 15.8, map_cx + 15.8), y_c - diameter / 2,
                             cmap=connectivity_cmap.name, color_range=SALIENCE_Z_RANGE, frame=sn_lh, height=diameter,
                             outline=sn_outline, width_frac=0.95)
        _mark(ax3, xy3, {name: connectivity_cmap(z_norm(by_vertex.loc[v, "P_z"])) for name, v in examples.items()})
        _colorbar(fig, map_cx, 6.6, connectivity_cmap, "z")

        # 4. Alignment with the microstructure gradient: a square scatter over the lollipops.
        ax4 = _mm_axes(fig, ax_x0, y_c - side / 2, side, side)
        ax4.scatter(vertex.loc[ok, "g_z"], vertex.loc[ok, "P_z"], s=2.6, c=mpc_cmap(z_norm(vertex.loc[ok, "g_z"])),
                    lw=0, alpha=0.7, rasterized=True)
        # Least-squares line through the points, and the subject's (Spearman) r_s.
        slope, intercept = np.polyfit(vertex.loc[ok, "g_z"], vertex.loc[ok, "P_z"], 1)
        g_line = np.array([vertex.loc[ok, "g_z"].min(), vertex.loc[ok, "g_z"].max()])
        ax4.plot(g_line, intercept + slope * g_line, color="#1a1a1a", lw=1.2, zorder=3)
        ax4.text(0.03, 0.97, f"r = {summary['r_s']:+.2f}".replace("-", "−"), transform=ax4.transAxes,
                 ha="left", va="top", fontsize=FONT_PT, color="#1a1a1a", path_effects=halo, zorder=4)
        for name, v in examples.items():
            g, P = by_vertex.loc[v, "g_z"], by_vertex.loc[v, "P_z"]
            ax4.plot(g, P, "o", ms=4.5, mfc=mpc_cmap(z_norm(g)), mec="#1a1a1a", mew=0.9)
            ax4.annotate(name, (g, P), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                         fontsize=FONT_PT - 0.5, fontweight="bold", path_effects=halo, zorder=4)
        ax4.set_xlabel("microstructure gradient", fontsize=FONT_PT - 1, labelpad=1.5, color=grey)
        ax4.set_ylabel("projected map", fontsize=FONT_PT - 1, labelpad=1.5, color=grey)
        ax4.set_xticks([])
        ax4.set_yticks([])
        for spine in ("right", "top"):
            ax4.spines[spine].set_visible(False)

        # Titles, and one arrow per gap.
        headers = [(sum(col[0]) / 2, "Connections", "to targets outside the SN"),
                   (proj_cx, "Connectivity-weighted projection", None),
                   (map_cx, "Projected map", None),
                   ((ax_x0 + ax_x1) / 2, "Alignment", None)]
        for x, title, sub in headers:
            _text(x, y_title, title, fontsize=HEADER_PT, color="#1a1a1a")
            if sub:
                _text(x, y_sub, sub, fontsize=FONT_PT - 0.5)
        for left, right in ((brain_right, proj_x0), (proj_x1, map_cx - diameter / 2),
                            (map_cx + diameter / 2, ax_x0 - ylabel_w)):
            _arrow(fig, left + clearance, right - clearance, y_c)

        out = config.FIGURES_DIR / f"figure_3_schematic_{hemisphere}.svg"
        fig.savefig(out, transparent=True, dpi=600)
        plt.close(fig)
    logger.info(f"Schematic written to {out} (example subject r_s = {summary['r_s']:+.3f}; "
                f"example vertex a = {examples['a']})")


def compute(settings: dict, paths: dict) -> None:
    """Schematic tables of the representative subject."""
    compute_schematic_data(paths, settings["hemisphere"])


def plot(settings: dict) -> None:
    """The schematic, from source_data only."""
    plot_schematic(settings["hemisphere"])


if __name__ == "__main__":
    common.main(STEP, compute, plot, "Step 2 schematic: visual key to the Figure 3 projection.")
