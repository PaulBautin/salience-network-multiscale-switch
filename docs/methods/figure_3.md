# Figure 3 — iEEG frequency-band mapping

**Scripts:** `scripts/figure_3_ieeg_mni.py` · `scripts/figure_3_ieeg_mica.py` · `scripts/figure_3c_ieeg_states.py`

Figure 3 maps the spectral content of resting-state intracranial EEG (iEEG) onto
the cortical surface in two independent cohorts: the MNI open iEEG atlas (panel a)
and the MICA iEEG dataset (panel b), in which the dependence of the aperiodic
exponent on vigilance state was also examined (panel c). Cohort composition and recording context are
described in [Data Acquisition](datasets.md#mni-open-ieeg-atlas). Both datasets
were processed with the common signal pipeline below and projected onto the
fsLR-32k surface (32,492 vertices per hemisphere).

## Signal preprocessing

For the MICA cohort, all recordings were first truncated to the length of the
shortest recording (360,136 samples, approximately 180 s at the native 2000 Hz), so
that every channel spectrum was estimated from the same amount of data. Each channel
time series was band-pass filtered between 0.5 and 80 Hz with a
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
assigned to its nearest vertex on the fsLR-32k surface. Because fsLR-32k is a
bilaterally symmetric template and the gradient is evaluated one hemisphere at a
time, every contact was projected onto the analysis hemisphere (default right) at its
homologous vertex and read against that hemisphere's qT1 gradient; this bilateral
homology assumption mirrors the hemispheric fold used for the MICA cohort
and maximises channel coverage of the gradient. Band-power values were averaged
across channels mapped to the same vertex. The network used to stratify channels along
the microstructural gradient is configurable through the `-network` flag, which accepts
any of the seven Yeo networks (default `SalVentAttn`), allowing whole-brain or
network-specific analyses without changes to the code. Within the target network,
channels falling in the lowest and highest quartiles of the qT1 gradient were
identified, and the electrophysiological similarity difference between these two groups
(ES$_\text{top}$ − ES$_\text{bottom}$) — defined from the absolute power-spectrum
correlation, in contrast to the positive-part spectral similarity used for the MICA
cohort — was correlated with the BigBrain G2 histological gradient across all channels
outside the target network (Spearman). Significance was assessed against a Moran
spectral randomisation null built on the channel layout: because the channels are
scattered points rather than a contiguous surface patch, the spatial weights were the
inverse Euclidean distances between the channels' fsLR-32k midthickness positions
(floored at 1 mm for channels mapped to the same vertex), the BigBrain gradient was
randomised (1,000 surrogates) with the channel values held fixed, and the two-tailed
add-one empirical $p$ was reported. The per-network bars are descriptive means without
an associated significance test.

### MICA iEEG dataset (Figure 3b)

For the MICA cohort, each electrode contact carries a subject-specific signed
leadfield (sensitivity map) from the electroMICA pipeline (von Ellenrieder et al.,
2021), defined separately for each hemisphere on the fsLR-32k midthickness surface.
Following electroMICA, the sensitivity of a bipolar channel was computed as the
difference between the signed leadfields of its two constituent contacts, evaluated
independently within each hemisphere; the sign of the leadfield is retained through
this subtraction, as the recorded channel is the potential difference of its
contacts. Two thresholds were then applied to each channel's area-normalised
sensitivity density (sensitivity divided by the per-vertex surface area): a common
absolute noise floor of $0.001~\text{Vm/A}$ and a channel-specific relative floor at
$5\%$ of the channel's second-largest density; vertices below either floor were set
to zero. The two hemispheres were finally folded onto a single fsLR-32k template by
summing the rectified per-hemisphere sensitivities, so that contacts on either
hemisphere inform the homologous template vertex and channel coverage is maximised.
Spectral quantities were projected onto the surface by a sensitivity-weighted average
across channels (the electroMICA smoothed feature map), so that contacts influence
surface vertices in proportion to their modelled spatial sensitivity rather than
through a single nearest-vertex assignment. Contacts without a leadfield model were
assigned zero sensitivity; channels left without any suprathreshold sensitivity
therefore carried no weight in the projection, and surface vertices covered by no
channel were left undefined. Because the fold makes each template vertex stand for a
pair of homologous vertices, the surface maps were displayed on both hemispheres,
whereas all statistics were computed within the left hemisphere, as for the other
analyses, against the salience-network MPC gradient of that hemisphere
([Figure 1](figure_1.md)).

**Spectral measures along the microstructural gradient.** Two electrophysiological
measures were derived from a single per-channel spectral parameterisation
(`specparam`/FOOOF; Donoghue et al., 2020), mapped to the surface by
sensitivity-weighted averaging, and, within the target network, correlated
(Spearman) with the within-network MPC gradient at every vertex carrying both iEEG
coverage and a defined gradient value. Each channel spectrum was fitted once with
`specparam` 2.0.0rc7 (algorithm `spectral_fit`), with Gaussian peaks and a knee-mode
aperiodic component,

$$L(f) = b - \log_{10}\!\left(\kappa + f^{\chi}\right),$$

where $L(f)$ is the aperiodic log10 power, $b$ the offset, $\kappa$ the knee and $\chi$ the
exponent. The knee was constrained to $\kappa \ge 0$ (the `ap_bounds` setting): a
negative knee has no physical interpretation, and with $\kappa = 0$ the model reduces to
the fixed power law, the appropriate model for a spectrum without a knee in the fit
range. The fit range was 1–45 Hz. Following the `specparam` recommendation not to fit
frequencies that have been filtered out, it stopped below both the narrow attenuation
at the 60 Hz line frequency present in the recordings and the transition band of the
80 Hz low-pass filter of the preprocessing (attenuation below 0.01 log10 units at
45 Hz). Peaks were constrained to widths of 1–12 Hz (the lower limit twice the
0.5 Hz frequency resolution), at most six peaks were fitted per spectrum, and a peak
had to exceed both an absolute height of 0.05 log10 units above the aperiodic fit and a
relative threshold of 2 standard deviations of the flattened spectrum. The same model
and settings were applied to every channel. For each fit, the goodness of fit ($R^2$)
and the mean absolute error were computed, and the fits with the lowest, median and
highest $R^2$ were inspected visually for under- and over-fitting; no fit was excluded
on these metrics, and only fits that failed to converge were discarded. The knee
frequency, $f_\kappa = \kappa^{1/\chi}$ (Gao et al., 2020), was computed to locate the
knee relative to the fit range. The primary measure was the
aperiodic (1/f) exponent, a theoretically motivated electrophysiological index of
cortical hierarchy and microstructural differentiation (Gao et al., 2020). The
secondary measure was the oscillatory peak power in the five canonical bands, each
restricted to the fit range (delta 1–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz,
gamma 30–45 Hz) — the
power of the strongest periodic peak above the aperiodic fit, which is orthogonal to
the exponent and therefore does not re-encode the same 1/f change (a vertex with no
detected peak carries zero oscillatory power) — with the five band correlations
corrected for multiple comparisons by the Benjamini–Hochberg procedure. For every
measure, significance
was assessed against a within-network Moran spatial null
(see [Shared Methods](shared.md#moran-spectral-randomisation-within-network)), the
spatial graph fitted once on the set of covered, gradient-defined vertices, with the
two-tailed empirical $p$-value from the add-one estimator
$p = (1 + k)/(1 + n_\text{perm})$, and the central 95% (2.5th–97.5th percentiles) of
the null correlations was reported alongside each observed correlation. For display,
each measure was $z$-scored across the target-network vertices of both hemispheres and
shown on a fixed range of ±2 SD, the scale of the Figure 1 maps.

To illustrate the spectra underlying these measures, the covered, gradient-defined
target-network vertices of the analysed hemisphere were divided into five equal-count
bins at the quintiles of the MPC gradient. Within each bin, the sensitivity-weighted
power spectrum over the fit range was averaged, together with the flattened spectrum: the log10 power
remaining after subtraction of each channel's fitted aperiodic component $L(f)$,
projected onto the surface by the same sensitivity-weighted average.

#### Supplementary analysis: spectral-similarity projection

The similarity of regional power spectra — not
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
additionally summarised as its mean restricted to each target network. As a
supplementary display of the spectral-similarity measure itself, channel spectra were
$z$-scored per frequency across channels, correlated pairwise, and ordered by the Yeo
network of each channel's peak-sensitivity vertex; channels without suprathreshold
sensitivity, which have no peak vertex, were excluded from this display.

### State dependence (Figure 3c)

The same bipolar channels of the MICA cohort were recorded in wakefulness and in N2
and N3 sleep. To estimate every spectrum from the same amount of data, the first
180 s of each recording (the length of the shortest wakefulness recording) were
analysed in all three states; channels were matched across states by subject and
bipolar channel name, and only channels recorded in all three states were retained.
Each recording was preprocessed and its power spectrum estimated as above, and each
channel spectrum of each state was parameterised with the same `specparam` model and
settings as in Figure 3b (knee-mode aperiodic component with $\kappa \ge 0$, 1–45 Hz),
with the same fit checks in every state; channels whose fit failed in
any state were excluded, so that all states shared one channel set. The aperiodic
exponent of each state, and its within-channel change from wakefulness to N3,

$$\Delta\chi = \chi_{\mathrm{N3}} - \chi_{\mathrm{W}},$$

were projected onto the surface by the sensitivity-weighted average described above.
Within the target network of the left hemisphere, the exponent of each state was
correlated (Spearman) with the MPC gradient at every vertex with iEEG coverage and a
defined gradient value, with significance from the within-network Moran null fitted
once on this vertex set
(see [Shared Methods](shared.md#moran-spectral-randomisation-within-network); add-one
empirical $p$), the central 95% of the null correlations reported alongside each
observed correlation, and the three per-state correlations corrected for multiple
comparisons by the Benjamini–Hochberg procedure. To compare the coupling between
states, the difference between the correlations of two states,
$\Delta r = r_a - r_b$, was tested for every pair of states: the same Moran surrogates
of the gradient were correlated with both states' maps, which preserves the
dependence between the two maps as well as the spatial autocorrelation of the gradient,
and the observed difference was compared with the resulting distribution of
differences (two-tailed add-one empirical $p$; Benjamini–Hochberg correction across the
three pairs). The within-channel change was tested against the gradient in the same
way; because it is computed within each channel, contact-specific factors that are
constant across states — leadfield gain, distance and orientation relative to the
sources, and the bipolar montage — cancel. For display, the covered, gradient-defined
vertices were divided into five equal-count bins at the quintiles of the MPC gradient
and the mean exponent of each bin was computed in each state.

As a supplementary analysis, the oscillatory peak power of each canonical band was
examined in the same way: for every band and state, the peak power above the aperiodic
fit from the same `specparam` fit (zero where no peak was detected, as in Figure 3b) was
projected onto the surface and correlated with the MPC gradient against the
within-network Moran null, with Benjamini–Hochberg correction across the five bands
within each state, and the correlations of every pair of states were compared with
paired Moran surrogates as above (Benjamini–Hochberg correction across the three pairs
within each band).
