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

---

## Histological map colors (Figure 1b)

### `histology_cmap`

`matplotlib.colors.ListedColormap` registered as `"CustomCmap_histology"` — a diverging map in stain colours, cresyl violet (below the SN mean, LCh hue 300°) → light neutral grey → DAB ochre (above, hue 75°), the companion of the MPC gradient's `coolwarm` on the shared ±2 SD scale (`SALIENCE_Z_RANGE`). Built in CIELAB/LCh with coolwarm's lightness range (L* 38 at both ends, 89 at the midpoint, linear in between) and chroma rising to 45, so a given z reads with the same weight as in the gradient, the arms are symmetric, and the ends stay well apart under deuteranopia, while the hues mark the histological maps as a different quantity.

---

### `ieeg_cmap`

`matplotlib.colors.ListedColormap` registered as `"CustomCmap_ieeg"` — the diverging map of the Figure 3b iEEG spectral maps (aperiodic exponent, oscillatory band power, spectral-similarity projection) z-scored within the SN: green (below the SN mean, LCh hue 135°) → light neutral grey → plum (above, hue 330°), on the shared ±2 SD scale (`SALIENCE_Z_RANGE`). Same lightness and chroma profile as `histology_cmap` (L* 38 → 89 → 38, chroma up to 45), so a given z reads with the same weight as in the gradient and histology maps, with hues distinct from both. The hue pair maximises the smallest end-to-end CIELAB distance under simulated deuteranopia, protanopia and tritanopia (Machado et al., 2009) among in-gamut pairs (ΔE ≥ 48).

---

### `connectivity_cmap`

`matplotlib.colors.ListedColormap` registered as `"CustomCmap_connectivity"` — the diverging map of the Figure 2a connectivity-weighted projection maps z-scored within the source network: azure (below the network mean, LCh hue 215°) → light neutral grey → orange (above, hue 35°), on the shared ±2 SD scale (`SALIENCE_Z_RANGE`). Same lightness and chroma profile as `histology_cmap` and `ieeg_cmap` (L* 38 → 89 → 38, chroma up to 45), so a given z reads with the same weight across Figures 1–3, with a hue pair distinct from the violet–ochre (histology) and green–plum (iEEG) families. The cool-to-warm sense follows the FC-gradient reading the projection carries (low $P$ = default-mode end, high $P$ = task-positive end).
