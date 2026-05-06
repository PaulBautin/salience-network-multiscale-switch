# `src/plot_colors`

Module-level color definitions for all network and cortical-type visualizations. Importing this module automatically registers the custom colormaps with matplotlib so they can be referenced by name in any plotting call.

## Yeo 7-network colors

### `yeo7_rgb`

`np.ndarray`, shape `(8, 3)` — RGB triplets (float, range 0–1) for the 7 Yeo networks plus a background entry.

| Index | Network |
|-------|---------|
| 0 | Frontoparietal (Cont) |
| 1 | Default Mode |
| 2 | Dorsal Attention |
| 3 | Limbic |
| 4 | Ventral Attention (SalVentAttn) |
| 5 | Somatomotor |
| 6 | Visual |
| 7 | Background / unlabeled |

### `yeo7_rgba`

`np.ndarray`, shape `(8, 4)` — `yeo7_rgb` with a column of ones appended (full opacity).

### `yeo7_cmap`

`matplotlib.colors.ListedColormap` registered as `"CustomCmap_yeo"`.

---

## Von Economo cortical-type colors

### `cmap_types_rgba`

`np.ndarray`, shape `(7, 4)` — RGBA colors for the 7 cortical-type levels (index 0 = background/unlabeled, indices 1–6 = koniocortex → agranular). The last two classes are slightly desaturated.

### `cmap_types`

`matplotlib.colors.ListedColormap` registered as `"CustomCmap_type"`.

### `cmap_types_rgba_mw`

`np.ndarray`, shape `(7, 4)` — Variant without a background entry; index 6 is reserved for the medial wall (light gray). Used when medial-wall vertices are explicitly included in the array.

### `cmap_types_mw`

`matplotlib.colors.ListedColormap` registered as `"CustomCmap_type_mw"`.

---

## Baillarger band colors

### `baillarger_rgba`

`np.ndarray`, shape `(5, 4)` — RGBA colors for Baillarger band types (index 0 = background, indices 1–4 = band classes).

### `baillarger_cmap`

`matplotlib.colors.ListedColormap` registered as `"CustomCmap_baillarger"`.

---

## Intrusion-type colors

### `intrusion_rgba`

`np.ndarray`, shape `(4, 4)` — RGBA colors for Intrusion type classes (index 0 = background, indices 1–3 = intrusion classes).

### `intrusion_cmap`

`matplotlib.colors.ListedColormap` registered as `"CustomCmap_intrusion"`.
