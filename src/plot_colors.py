import numpy as np
import matplotlib as mpl


# Yeo et al. (2011) – 7-network parcellation (+ background)
yeo7_rgb = np.array(
    [
        [255, 180, 80],    # Frontoparietal
        [230, 90, 100],    # Default Mode
        [0, 170, 50],      # Dorsal Attention
        [240, 255, 200],   # Limbic
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

# Slight desaturation of the last two classes
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


# Baillarger Bands
baillarger_rgb = np.array(
    [
        [255, 255, 255],  # Background / unlabeled
        [102, 194, 165],
        [252, 141, 98],
        [141, 160, 203],
        [231, 138, 195],
    ],
    dtype=float,
) / 255.0

baillarger_alpha = np.ones((baillarger_rgb.shape[0], 1))
baillarger_rgba = np.hstack((baillarger_rgb, baillarger_alpha))
baillarger_cmap = mpl.colors.ListedColormap(baillarger_rgba, name="CustomCmap_baillarger")
mpl.colormaps.register(baillarger_cmap)


# Intrusion classes
intrusion_rgb = np.array(
    [
        [255, 255, 255],  # Background / unlabeled
        [102, 194, 165],
        [252, 141, 98],
        [141, 160, 203],
    ],
    dtype=float,
) / 255.0

intrusion_alpha = np.ones((intrusion_rgb.shape[0], 1))
intrusion_rgba = np.hstack((intrusion_rgb, intrusion_alpha))
intrusion_cmap = mpl.colors.ListedColormap(intrusion_rgba, name="CustomCmap_intrusion")
mpl.colormaps.register(intrusion_cmap)