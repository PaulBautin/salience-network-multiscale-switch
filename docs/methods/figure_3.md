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
spatial weighting vector. Spectral quantities were projected onto the surface by
weighting each contact's contribution by its sensitivity map, so that contacts
influence surface vertices in proportion to their modelled spatial sensitivity
rather than through a single nearest-vertex assignment. The analyses below were
carried out within one hemisphere at a time.

**Band power along the microstructural gradient.** Sensitivity-weighted relative
band power was mapped to the surface for each canonical frequency band, and within
the target network its value at each covered vertex was correlated (Spearman) with
the within-network MPC gradient. Significance was assessed against a within-network
Moran spectral-randomisation null (see [Shared Methods](shared.md#moran-spectral-randomisation-within-network)),
with the two-tailed empirical $p$-value computed from the add-one estimator
$p = (1 + k)/(1 + n_\text{perm})$.

**Electrophysiological-similarity projection.** The spectral content of the iEEG
recordings was treated as a connectivity measure, by analogy with functional
connectivity, and entered into the same gradient-weighted projection used in
[Figure 2](figure_2.md#the-projection-statistic). For every covered surface vertex
a power spectral density fingerprint was obtained by sensitivity-weighted averaging
of channel spectra and $z$-scored across frequencies. The electrophysiological
similarity between two vertices was defined as the positive part of the Pearson
correlation between their spectral fingerprints, giving a vertex-by-vertex
non-negative similarity that plays the role of a connection weight. For each
source-network vertex $i$ the projection score is the similarity-weighted mean of
the functional-connectivity (FC) gradient across its targets,

$$P_i = \frac{\sum_{j} \text{ES}^{+}_{ij}\, g^{\mathrm{FC}}_j}{\sum_{j} \text{ES}^{+}_{ij}}, \qquad j \in \{\text{cortical vertices outside the source network}\},$$

where $\text{ES}^{+}_{ij}$ is the non-negative spectral similarity and
$g^{\mathrm{FC}}_j$ the FC gradient at target $j$. The FC gradient polarity is
arbitrary, so it was oriented by anatomy — flipped where necessary so the
default-mode network occupied the low end and task-positive systems the high end —
and the chosen orientation was logged. Vertices with fewer than ten contributing
targets were left undefined.

Because the iEEG spectral fingerprints are aggregated across the cohort, inference
was performed at the group level: the projection was computed once and summarised
by a single Spearman correlation between the within-network MPC gradient and the
projection score across source-network vertices. Spatial significance was the
within-network Moran spectral-randomisation null
(see [Shared Methods](shared.md#moran-spectral-randomisation-within-network)),
with surrogates of the MPC gradient generated within the analysed hemisphere and
the two-tailed empirical $p$-value taken from the add-one estimator
$p = (1 + k)/(1 + n_\text{perm})$. For display, the projection was additionally
summarised as its mean restricted to each target network, giving a per-target-network
readout of the spectral-similarity-weighted FC-gradient position.
