"""Colours and registered matplotlib colormaps used by every figure.

Importing the module registers the colormaps with matplotlib, so they can also be passed
by name (e.g. ``cmap="CustomCmap_ieeg"``).

- ``yeo7_rgb``, ``yeo7_rgba`` (rows in alphabetical network order, then the background
  colour), ``yeo7_abbrev`` (network abbreviations) and ``yeo7_cmap``
  (``CustomCmap_yeo``): the Yeo et al. (2011) 7 networks.
- ``cmap_types`` (``CustomCmap_type``) and ``cmap_types_mw`` (``CustomCmap_type_mw``,
  with the medial wall): the von Economo cortical types.
- ``histology_cmap`` (``CustomCmap_histology``, violet to ochre), ``ieeg_cmap``
  (``CustomCmap_ieeg``, green to plum) and ``connectivity_cmap``
  (``CustomCmap_connectivity``, azure to orange): diverging maps for values z-scored
  within the salience network, drawn on the shared +/-2 SD scale. They share the
  lightness profile of matplotlib's coolwarm (used for the MPC gradient), so a given z
  reads with the same weight in every figure.
"""
import matplotlib as mpl
import numpy as np


# Yeo et al. (2011) – 7-network parcellation (+ background)
yeo7_rgb = np.array(
    [
        [255, 180, 80],    # Frontoparietal
        [230, 90, 100],    # Default Mode
        [0, 170, 50],      # Dorsal Attention
        [182, 176, 102],   # Limbic
        [210, 100, 255],   # Ventral Attention
        [100, 160, 220],   # Somatomotor
        [170, 70, 200],    # Visual
        [255, 255, 255],   # Background / unlabeled
    ],
    dtype=float,
) / 255.0


yeo7_alpha = np.ones((yeo7_rgb.shape[0], 1))


yeo7_rgba = np.hstack((yeo7_rgb, yeo7_alpha))


yeo7_cmap = mpl.colors.ListedColormap(yeo7_rgba, name="CustomCmap_yeo")


mpl.colormaps.register(yeo7_cmap)


yeo7_abbrev = {
    "Cont":       "FPN",
    "Default":    "DMN",
    "DorsAttn":   "DAN",
    "Limbic":     "Lim",
    "SalVentAttn": "Sal",
    "SomMot":     "SM",
    "Vis":        "Vis",
}


# Von Economo – cortical types
cmap_types_rgb = np.array(
    [
        [255, 255, 255],  # Background / unlabeled
        [127, 140, 172],  # Desaturated blue-gray
        [139, 167, 176],  # Desaturated cyan-gray
        [171, 186, 162],  # Muted green
        [218, 198, 153],  # Dull yellow
        [253, 211, 200],  # Pale coral
        [252, 229, 252],  # Pale magenta
    ],
    dtype=float,
) / 255.0


# Slightly darken the last two classes (RGB x 0.8)
cmap_types_rgb[-2:, :] *= 0.80


cmap_types_alpha = np.ones((cmap_types_rgb.shape[0], 1))


cmap_types_rgba = np.hstack((cmap_types_rgb, cmap_types_alpha))


cmap_types = mpl.colors.ListedColormap(cmap_types_rgba, name="CustomCmap_type")


mpl.colormaps.register(cmap_types)


# Von Economo – cortical types with medial wall
cmap_types_rgb_mw = np.array(
    [
        [127, 140, 172],  # Desaturated blue-gray
        [139, 167, 176],  # Desaturated cyan-gray
        [171, 186, 162],  # Muted green
        [218, 198, 153],  # Dull yellow
        [253, 211, 200],  # Pale coral
        [252, 229, 252],  # Pale magenta
        [220, 220, 220],  # Medial wall
    ],
    dtype=float,
) / 255.0


cmap_types_alpha_mw = np.ones((cmap_types_rgb_mw.shape[0], 1))


cmap_types_rgba_mw = np.hstack((cmap_types_rgb_mw, cmap_types_alpha_mw))


cmap_types_mw = mpl.colors.ListedColormap(cmap_types_rgba_mw, name="CustomCmap_type_mw")


mpl.colormaps.register(cmap_types_mw)


# Figure 1b histological maps – diverging companion to the coolwarm MPC gradient on the
# shared +/-2 SD scale, in stain colours: cresyl violet (below the SN mean) -> light
# neutral grey -> DAB ochre (above). Built in CIELAB/LCh with coolwarm's lightness range
# (L* 38 at both ends, 89 at the midpoint, linear in between) and moderate chroma, so a
# given z reads with the same weight as in the gradient, the arms are symmetric, and the
# ends stay well apart under deuteranopia, while the hues mark a different quantity.
def _lch_diverging(name: str, hue_low: float, hue_high: float, chroma_end: float,
                   l_end: float = 38.0, l_mid: float = 89.0, n: int = 256) -> mpl.colors.ListedColormap:
    t = np.linspace(-1, 1, n)
    L, C = l_mid + (l_end - l_mid) * np.abs(t), chroma_end * np.abs(t)
    h = np.radians(np.where(t < 0, hue_low, hue_high))
    fy = (L + 16) / 116
    f = np.stack([fy + C * np.cos(h) / 500, fy, fy - C * np.sin(h) / 200], -1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * np.array([0.95047, 1.0, 1.08883])
    lin = np.clip(xyz @ np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]).T, 0, 1)
    rgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return mpl.colors.ListedColormap(rgb, name=name)


histology_cmap = _lch_diverging("CustomCmap_histology", hue_low=300, hue_high=75, chroma_end=45)


mpl.colormaps.register(histology_cmap)


# Figure 2b iEEG maps – third companion on the shared +/-2 SD scale, for spectral measures
# z-scored within the SN: green (below the SN mean) -> light neutral grey -> plum (above).
# Same lightness and chroma profile as CustomCmap_histology, so a given z reads with the
# same weight as in the gradient and histology maps, with hues distinct from both. The
# hues maximise the smallest end-to-end separation under simulated deuteranopia,
# protanopia and tritanopia (Machado et al., 2009) among in-gamut pairs (CIELAB dE >= 48).
ieeg_cmap = _lch_diverging("CustomCmap_ieeg", hue_low=135, hue_high=330, chroma_end=45)


mpl.colormaps.register(ieeg_cmap)


# Figure 3 connectivity-projection maps – fourth companion on the shared +/-2 SD scale,
# for the gradient-weighted projection P z-scored within the source network: azure (below
# the network mean) -> light neutral grey -> orange (above). Same lightness and chroma
# profile as CustomCmap_histology / CustomCmap_ieeg, so a given z reads with the same
# weight across figures, with a hue pair distinct from the violet-ochre (histology) and
# green-plum (iEEG) families. The cool-to-warm sense follows the FC-gradient reading the
# projection carries (low P = default-mode end, high P = task-positive end).
connectivity_cmap = _lch_diverging("CustomCmap_connectivity", hue_low=215, hue_high=35, chroma_end=45)


mpl.colormaps.register(connectivity_cmap)
