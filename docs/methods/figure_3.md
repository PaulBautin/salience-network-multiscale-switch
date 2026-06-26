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

**Spectral measures along the microstructural gradient.** Two electrophysiological
measures were derived from a single per-channel spectral parameterisation
(`specparam`/FOOOF; Donoghue et al., 2020), mapped to the surface by
sensitivity-weighted averaging, and, within the target network, correlated
(Spearman) with the within-network MPC gradient at every vertex carrying both iEEG
coverage and a defined gradient value. Each channel spectrum was fitted once in knee
mode over 1–80 Hz with explicit Gaussian peak modelling. The primary measure was the
aperiodic (1/f) exponent, a theoretically motivated electrophysiological index of
cortical hierarchy and microstructural differentiation (Gao et al., 2020). The
secondary measure was the oscillatory peak power in the five canonical bands — the
power of the strongest periodic peak above the aperiodic fit, which is orthogonal to
the exponent and therefore does not re-encode the same 1/f change (a vertex with no
detected peak carries zero oscillatory power) — with the five band correlations
corrected for multiple comparisons by the Benjamini–Hochberg procedure. For every
measure, significance
was assessed against a within-network Moran spatial null
(see [Shared Methods](shared.md#moran-spectral-randomisation-within-network)), the
spatial graph fitted once on the set of covered, gradient-defined vertices, with the
two-tailed empirical $p$-value from the add-one estimator
$p = (1 + k)/(1 + n_\text{perm})$.

**Spectral-similarity projection.** The similarity of regional power spectra — not
the temporal coupling between signals — was used as the weight in the
gradient-weighted projection of
[Figure 2](figure_2.md#the-projection-statistic); it is accordingly a measure of
spectral similarity rather than of functional connectivity and is interpreted as
such. For every covered surface vertex a power spectral density fingerprint was
obtained by sensitivity-weighted averaging of channel spectra and $z$-scored across
frequencies, and the spectral similarity between two vertices was the positive part
of the Pearson correlation between their fingerprints. For each source-network
vertex $i$ the projection score is the similarity-weighted mean of the
functional-connectivity (FC) gradient across its targets,

$$P_i = \frac{\sum_{j} \text{SS}^{+}_{ij}\, g^{\mathrm{FC}}_j}{\sum_{j} \text{SS}^{+}_{ij}}, \qquad j \in \{\text{cortical vertices outside the source network}\},$$

where $\text{SS}^{+}_{ij}$ is the non-negative spectral similarity and
$g^{\mathrm{FC}}_j$ the FC gradient at target $j$ — the same cohort principal FC
gradient used in [Figure 2](figure_2.md), evaluated on the fsLR-32k surface the
sensitivity maps require and oriented by anatomy so the default-mode network occupies
the low end. Because two vertices sampled by overlapping leadfields share a
sensitivity-averaged spectrum and are therefore similar for instrumental rather than
neural reasons, source–target pairs whose sensitivity profiles had a cosine
similarity above $0.1$ were excluded from the projection as leakage. To establish
that the spectral similarity contributed information beyond spatial geometry, the
projection was recomputed over the same target set with uniform weights and with
inverse-Euclidean-distance weights, and the three correlations were reported
together. Vertices with fewer than ten contributing targets were left undefined.

Because the iEEG spectra are aggregated across the cohort, inference was performed
at the group level: a single Spearman correlation between the within-network MPC
gradient and the projection across source-network vertices, with spatial
significance from the within-network Moran null
(see [Shared Methods](shared.md#moran-spectral-randomisation-within-network);
surrogates of the MPC gradient generated within the analysed hemisphere, add-one
empirical $p$). Both source and target vertices were restricted to the locations of
iEEG coverage, a non-uniform convenience sample of cortex; the projection therefore
probes the FC gradient only where electrodes were implanted, and the group statistic
does not model between-subject variability. For display, the projection was
additionally summarised as its mean restricted to each target network.
