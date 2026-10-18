# `src/surface_plotting`

Brain-surface figures composed from brainspace renders: each view is rendered offscreen by brainspace and laid out in matplotlib, which adds annotations brainspace cannot draw, such as circular zoom insets of regions of interest. No brainspace colour bar is rendered, so the module works with VTK builds whose brainspace colour bar is broken.

### `HEMI_VIEWS`

`dict[str, tuple[str, str]]` — the four `plot_hemispheres` columns mapped to (hemisphere, brainspace camera view).

| Key | Hemisphere | brainspace view |
|-----|------------|-----------------|
| `lh_lateral` | `lh` | `lateral` |
| `lh_medial` | `lh` | `medial` |
| `rh_medial` | `rh` | `lateral` |
| `rh_lateral` | `rh` | `medial` |

brainspace points the same camera at both hemispheres, so its `lateral` camera shows the right hemisphere's medial wall.

---

### `SALIENCE_Z_RANGE`, `SALIENCE_Z_LABELS`

`SALIENCE_Z_RANGE = (-2.0, 2.0)` — the shared, symmetric colour range (in SD) of the Figure 1 maps z-scored within the salience network (MPC gradient, BigBrain, Bielschowsky, Parvalbumin), so one colour means the same deviation from the SN mean in every panel; about 96 % of each map's SN vertices fall inside it. `SALIENCE_Z_LABELS = ('−2', '+2')` are the matching colour-bar end labels.

---

### `ZOOM_ROW_WIDTH_MM`

`ZOOM_ROW_WIDTH_MM = 150` — display width (mm) of the 1450-pt zoom rows in the Figure 1 and Figure 3 layouts (~0.29×); [`plot_forest`](#plot_forest) uses it to place its rows level with stacked inset rows.

---

### `SALIENCE_ZOOM_ROIS`

`list[tuple[str, list[str], str]]` — the Figure 1 zoom regions as (view, Schaefer-400 parcels without the leading `7Networks_`, label).

| View | Parcels | Label |
|------|---------|-------|
| `lh_lateral` | `LH_SalVentAttn_FrOperIns_1`–`9` | Anterior insula (aINS) |
| `lh_medial` | `LH_SalVentAttn_Med_1`, `_2`, `_4` | Dorsal anterior cingulate cortex (dACC), on three lines |
| `rh_medial` | `RH_SalVentAttn_Med_3`, `_5`, `_6` | Precuneus (PCU) |
| `rh_lateral` | `RH_SalVentAttn_TempOccPar_1`–`7` | Inferior parietal lobule (IPL) |

---

### `salience_zoom_rois`

```python
salience_zoom_rois(df_surf: pd.DataFrame, hemisphere: str = 'both', with_labels: bool = True) -> list[dict]
```

Build the [`plot_hemispheres_zoom`](#plot_hemispheres_zoom) region dicts for `SALIENCE_ZOOM_ROIS` from a surface table's Schaefer labels.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `df_surf` | `pd.DataFrame` | Surface table over both hemispheres with the Schaefer-400 `label` column (e.g. `df_yeo_surf`). |
| `hemisphere` | `str` | `'both'`, `'LH'` or `'RH'`; regions of the other hemisphere are left out. |
| `with_labels` | `bool` | Attach the region names (drawn under the insets). |

**Returns** `list[dict]` — one `{'view', 'mask', 'label'}` dict per region.

---

### `mask_outline`

```python
mask_outline(surf_lh, surf_rh, mask: np.ndarray) -> np.ndarray
```

Vertices just outside a region, for the `outline` of [`plot_hemispheres_zoom`](#plot_hemispheres_zoom): those outside `mask` that share a triangle with a vertex inside it, so a line traced on them encloses the region without covering its values. For the salience network it equals the outer side of the `salience_border` of `load_yeo_atlas`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `surf_lh` | `BSPolyData` | Left-hemisphere surface. |
| `surf_rh` | `BSPolyData` | Right-hemisphere surface. |
| `mask` | `np.ndarray` | Bool array over both hemispheres, LH then RH, shape `(n_lh + n_rh,)` (e.g. the vertices of a network). |

**Returns** `np.ndarray` — bool array of the same shape, True on the outline vertices.

---

### `render_surface_view`

```python
render_surface_view(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]
```

Render one surface view offscreen as `plot_hemispheres` renders one panel (orthographic camera, transparent background), supersampled by `scale`, and project every vertex into the image through the render's camera and actor matrices.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `surf` | `BSPolyData` | Hemisphere surface. |
| `values` | `np.ndarray` | Per-vertex values, shape `(n_vertices,)`; NaN drawn in `nan_color`. |
| `view` | `str` | brainspace camera view: `'lateral'`, `'medial'`, `'dorsal'` or `'ventral'`. |
| `size` | `tuple[int, int]` | Viewport (width, height) in pixels before `scale`. |
| `scale` | `int` | Supersampling factor of the returned image. |
| `zoom` | `float` | Camera zoom, as in `plot_hemispheres`. |
| `cmap` | `str` | Matplotlib colour map name. |
| `color_range` | `tuple[float, float] \| None` | Colour-map (min, max); data range if `None`. |
| `nan_color` | `tuple` | RGBA of NaN vertices (RGB 0–255, alpha 0–1). |

**Returns** `(img, xy, visible, px_per_mm)` — the RGBA `uint8` image of shape `(size[1]*scale, size[0]*scale, 4)`; each vertex's pixel position (column, row) in `img`, shape `(n_vertices, 2)`; a boolean mask of vertices whose normal faces the camera; and image pixels per surface millimetre.

---

### `plot_hemispheres_zoom`

```python
plot_hemispheres_zoom(
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
    outline_sharpness: float = 1.0,
    dpi: int = 300,
) -> plt.Figure
```

`plot_hemispheres` row with a circular, magnified inset of a region of interest under each view. The brain row reproduces the `plot_hemispheres` window of the same `size`, `zoom` and `color_bar`: the figure is `size[0]` pt wide (one pt per window pixel) and each view sits in the viewport brainspace gives it (equal columns, plus an 11 % colour-bar column), so its top `size[1]` pt overlay a `plot_hemispheres` SVG export exactly. A circle marks each region on its view and a vertical line drops from it to the inset, centred under the column `gap` pt below the brain row (shifted sideways only if the line would miss it). Each inset is a crop of the same supersampled render, so it shows exactly the circled area. An optional `outline` (e.g. [`mask_outline`](#mask_outline) of a network) is traced over every view, and so magnified in the insets too: brainspace renders the outline vertices black on a white surface and an unmarked white surface with the same camera, and the luminance ratio of the two (×1.5, clipped to 1) is the line's opacity, so occlusion and shading follow the brain render. With `roi_only=True` only the inset row is drawn (no colour bar unless `draw_color_bar=True`), with the same width and columns; placed `size[1]` pt below a `plot_hemispheres` export, it reproduces the full figure. An optional `title` adds a band above everything, shifting the rows down by its height.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `surf_lh`, `surf_rh` | `BSPolyData` | Left and right hemisphere surfaces. |
| `values` | `np.ndarray` | Per-vertex values, LH then RH, shape `(n_lh + n_rh,)`. |
| `rois` | `list[dict]` | One dict per region, at most one per view: `'view'` (key of `HEMI_VIEWS`), `'mask'` (bool, shape `(n_lh + n_rh,)`), optional `'label'` (text under the inset, may contain `\n`) and optional `'radius'` (zoom-window radius in mm; default: smallest circle around the region's camera-facing vertices × `roi_pad`). |
| `views` | `tuple[str, ...]` | Columns of the brain row, keys of `HEMI_VIEWS` (default: the four `plot_hemispheres` panels). |
| `size` | `tuple[int, int]` | `plot_hemispheres` window (width, height) in pixels = figure pt. |
| `zoom` | `float` | Camera zoom, as in `plot_hemispheres`. |
| `color_bar` | `bool` | Reserve the colour-bar column, as `plot_hemispheres(color_bar='right')`. |
| `draw_color_bar` | `bool \| None` | Draw the colour bar in that column, with brainspace's size and position, centred on the brain row (or on the inset row when `roi_only`). Default: drawn unless `roi_only`. |
| `roi_only` | `bool` | Draw only the inset row (no brain row, markers or lines). |
| `title` | `str \| None` | Text above the figure, starting at the left edge of the leftmost drawn brain or inset. |
| `filename` | path or `None` | Output file; the figure is only returned if `None`. |
| `cmap` | `str` | Matplotlib colour map name. |
| `color_range` | `tuple[float, float] \| None` | (min, max) shared by all views; finite data range if `None`. |
| `nan_color` | `tuple` | RGBA of NaN vertices (RGB 0–255, alpha 0–1). |
| `scale` | `int` | Supersampling factor of the renders (inset sharpness). |
| `inset_frac` | `float` | Inset diameter as a fraction of the view width. |
| `gap` | `float` | Vertical space between the brain row and the insets, in pt. |
| `roi_pad` | `float` | Enlargement of the automatic zoom-window radius. |
| `cbar_labels` | `tuple[str, str] \| None` | Texts below and above the colour bar (e.g. `('min', 'max')`); `color_range` values if `None`. |
| `cbar_title` | `str \| None` | Text written vertically beside the colour bar (e.g. its unit). |
| `label_fontsize` | `float` | Font size (pt) of inset labels and colour-bar texts. |
| `title_fontsize` | `float` | Font size (pt) of `title`. |
| `inset_facecolor` | colour | Fill of the inset circles behind the surface. |
| `line_color` | colour | Colour of marker circles, inset outlines and lines. |
| `line_width` | `float` | Line width (pt) of marker circles, inset outlines and lines. |
| `outline` | `np.ndarray \| None` | Bool array over both hemispheres, shape `(n_lh + n_rh,)`, of vertices traced as a line over the map (two extra renders per view). |
| `outline_color` | color | Colour of the `outline` line. Default `'black'`. |
| `outline_sharpness` | `float` | Falloff power of the traced `outline`: `> 1` concentrates the darkening toward the boundary for a crisper, narrower line, useful on a coarse mesh (fsLR-5k) where the one-triangle band is physically wide. Default `1.0` (the Figure 1/3 soft falloff). |
| `dpi` | `int` | Resolution of the embedded raster images when saving. |

**Returns** `matplotlib.figure.Figure`. Its `inset_cy` attribute is the centre of the inset row in pt from the figure top; with the figure height (`fig.get_size_inches()[1] * 72`) it lets a companion panel align its rows with the insets.

**Raises** `ValueError` if a region's view is not in `views`, a view has more than one region, or a region has no camera-facing vertex in its view.

**Example**

```python
labels = df_yeo_surf['label'].fillna('')
rois = [{"view": "lh_lateral", "label": "Anterior insula\n(aINS)",
         "mask": labels.str.startswith("7Networks_LH_SalVentAttn_FrOperIns").to_numpy()}]
values = df_yeo_surf['t1_gradient1_SalVentAttn'].to_numpy()
# brain row + insets; its top 300 pt overlay plot_hemispheres(size=(1450, 300), zoom=1.3, color_bar='right')
plot_hemispheres_zoom(surf_lh, surf_rh, values, rois, color_range=(-3, 3),
                      cbar_labels=('min', 'max'), filename="brain_zoom.svg")
# inset row only, to stack 300 pt below the plot_hemispheres export
plot_hemispheres_zoom(surf_lh, surf_rh, values, rois, color_range=(-3, 3),
                      roi_only=True, filename="brain_zoom_roi.svg")
```

---

### `format_pvalue`

```python
format_pvalue(value: float) -> str
```

Format a p or q value to 3 decimals, with more digits where rounding would cross 0.05 (e.g. an add-one q of 50/1001 = 0.04995 prints as `'0.04995'`, not `'0.050'`), so the printed value agrees with a p < 0.05 significance marker.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `value` | `float` | p or q value. |

**Returns** `str` — the formatted value.

---

### `plot_forest`

```python
plot_forest(
    stats: pd.DataFrame,
    inset_cy: float,
    row_h: float,
    filename=None,
    *,
    row_width: float = 1450,
    xlim: tuple[float, float] = (-0.4, 0.4),
    xlabel: str = "Spearman r",
) -> plt.Figure
```

Forest plot whose rows sit level with stacked [`plot_hemispheres_zoom`](#plot_hemispheres_zoom) inset rows (Figures 1b, 3b and 3c). One row per test, top to bottom: the observed correlation (black dot, filled if its uncorrected Moran `p` < 0.05) over the central 95 % of its null distribution (grey bar from `null_lo` to `null_hi`, tick at `null_med`), with r and the uncorrected Moran p (labelled $p_m$) written above (FDR `q` values, where the table has them, are not drawn), on a signed correlation axis with a dashed zero line. Drawn at its final size (177.8 pt wide, 8.8-pt text; width and x-axis span those of the Figure 1a profiles panel as laid out in Figure 1). Row *k* is centred level with the insets of the *k*-th row when the rows (each `row_h` pt tall, inset centre `inset_cy` pt from its top) are stacked and shown [`ZOOM_ROW_WIDTH_MM`](#zoom_row_width_mm) wide, with the forest's top at the first row's top.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `stats` | `pd.DataFrame` | One row per test with columns `r`, `p` (Moran permutation p), `null_lo`, `null_med`, `null_hi`; a row with NaN `r` leaves an empty slot. |
| `inset_cy` | `float` | Inset-row centre (pt from the top), the `inset_cy` attribute of a `plot_hemispheres_zoom` figure. |
| `row_h` | `float` | Height (pt) of each inset-row figure. |
| `filename` | path or `None` | Output path; the figure is only returned if `None`. |
| `row_width` | `float` | Width (pt) of the inset rows. Default `1450`. |
| `xlim` | `tuple[float, float]` | Signed correlation axis, ticks at its ends and 0. Default `(-0.4, 0.4)`. |
| `xlabel` | `str` | Axis label. Default `'Spearman r'`. |

**Returns** `matplotlib.figure.Figure` — the forest figure.
