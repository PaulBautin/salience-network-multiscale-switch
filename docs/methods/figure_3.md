# Figure 3 — iEEG frequency-band mapping

**Scripts:** `scripts/figure_3_ieeg_mni.py` · `scripts/figure_3_ieeg_mica.py`

Figure 3 maps the spectral content of resting-state intracranial EEG (iEEG) onto
the cortical surface in two independent cohorts: the MNI open iEEG atlas (panel a)
and the MICA iEEG dataset (panel b). Cohort composition and recording context are
described in [Data Acquisition](datasets.md#mni-open-ieeg-atlas). Both datasets
were processed with the common signal pipeline below and projected onto the
fsLR-32k surface (32,492 vertices per hemisphere).

## Signal preprocessing

Each channel time series was band-pass filtered between 0.5 and 80 Hz with a
fourth-order zero-phase Butterworth filter, resampled to 200 Hz, and demeaned by
subtracting its temporal mean. For the MNI atlas, whose signals are distributed in
an already-preprocessed form, the filtering and resampling steps were omitted and
the power spectrum was computed directly.

## Power spectral density and band power

Power spectral density (PSD) was estimated by Welch's method using a 2-second
Hamming window with 1-second overlap, yielding a frequency resolution of 0.5 Hz.
Relative band power for a band $b$ spanning $[f_1, f_2]$ was computed by
integrating the PSD over the band with Simpson's rule, normalising by total power
across the analysis range, and log-transforming:

$$P_b = \log_{10}\!\left(\frac{\int_{f_1}^{f_2} S(f)\,df}{\int_{0.5}^{80} S(f)\,df} + \epsilon\right), \qquad \epsilon = 10^{-12},$$

where $S(f)$ is the channel PSD. Power was quantified in the five canonical
frequency bands:

| Band | Range |
|------|-------|
| Delta | 0.5–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–80 Hz |

## Surface mapping

### MNI open iEEG atlas (Figure 3a)

Channel coordinates are provided in MNI152 stereotactic space. Each channel was
assigned to its nearest vertex on the fsLR-32k surface, and band-power values were
averaged across channels mapped to the same vertex. The network used to stratify
channels along the microstructural gradient is configurable through the
`-network` flag, which accepts any of the seven Yeo networks (default
`SalVentAttn`), allowing whole-brain or network-specific analyses without changes
to the code. Within the target network, channels falling in the lowest and highest
quartiles of the qT1 gradient were identified, and the electrophysiological
similarity difference between these two groups (ES$_\text{top}$ − ES$_\text{bottom}$)
was correlated with the BigBrain G2 histological gradient across all channels
outside the target network.

### MICA iEEG dataset (Figure 3b)

For the MICA cohort, each electrode contact carries a subject-specific sensitivity
map derived from a leadfield model in the electroMICA pipeline, defined on the
fsLR-32k surface. Each contact's sensitivity map was rectified (absolute value),
thresholded at 0.001, and aggregated across hemispheres to give a per-contact
spatial weighting vector. Band power was projected onto the surface by weighting
each contact's contribution by its sensitivity map, so that contacts influence
surface vertices in proportion to their modelled spatial sensitivity rather than
through a single nearest-vertex assignment.
