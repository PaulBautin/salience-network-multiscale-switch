# Figure 3 — iEEG frequency-band mapping

**Scripts:** `scripts/figure_3_ieeg_mni.py` · `scripts/figure_3_ieeg_mica.py`

Figure 3 maps intracranial EEG (iEEG) spectral power onto the cortical surface
using two independent datasets: the MNI open iEEG atlas (panel a) and the MICA
iEEG dataset (panel b).

---

## iEEG signal preprocessing

Raw iEEG signals were preprocessed with a common pipeline applied to both datasets.

1. **Band-pass filter** — 4th-order zero-phase Butterworth, 0.5–80 Hz.
2. **Downsample** — to 200 Hz.
3. **Demean** — subtract the temporal mean of each channel.

---

## Power spectral density

PSD was estimated using Welch's method:

- Window: Hamming, 2-second segments, 1-second overlap.
- Frequency resolution: 0.5 Hz.

Relative band power for band $b$ with frequency range $[f_1, f_2]$:

$$P_b = \log_{10}\!\left(\frac{\int_{f_1}^{f_2} S(f)\,df}{\int_{0.5}^{80} S(f)\,df} + \epsilon\right), \quad \epsilon = 10^{-12}$$

where $S(f)$ is the PSD and the integral is approximated with Simpson's rule.

Frequency bands:

| Band | Range |
|------|-------|
| Delta | 0.5–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–80 Hz |

---

## Surface mapping

### MNI open iEEG atlas (Figure 3a)

Channel coordinates are provided in MNI152 stereotactic space. Each channel is
mapped to the nearest fsLR-32k surface vertex. Band power values are averaged
across channels assigned to the same vertex.

### MICA iEEG dataset (Figure 3b)

Subject-specific electrode sensitivity maps (leadfield-derived, from the
electroMICA pipeline) provide a per-contact spatial weighting vector on the
fsLR-32k surface (32,492 vertices per hemisphere). For each contact, the
sensitivity map is rectified (absolute value), thresholded at 0.001, and
aggregated across hemispheres. Band power is projected onto the surface by
weighting each contact's contribution by its sensitivity map.
