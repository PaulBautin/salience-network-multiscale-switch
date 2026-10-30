# Figure 2 — Intracranial EEG along the microstructural gradient

**Steps:** `pipeline/step2a_ieeg_mni.py` · `pipeline/step2b_ieeg_mica.py` · `pipeline/step2c_ieeg_states.py`
**Modules:** [`salience_switch.ieeg`](../api/ieeg.md) · [`salience_switch.stats`](../api/stats.md)

The spectral content of intracranial EEG (iEEG) was mapped onto the cortical surface
in two independent cohorts, the MNI Open iEEG Atlas (Fig 2A) and the MICA iEEG dataset
(Fig 2B). In the MICA dataset, the dependence of the aperiodic exponent on vigilance
state was also examined (Fig 2C). Both cohorts are described in
[Data Acquisition](datasets.md#mni-open-ieeg-atlas). Both datasets were processed with
the same signal pipeline and projected onto the fsLR-32k surface (32,492 vertices per
hemisphere).

## Signal preprocessing and spectral estimation

For the wakefulness recordings of the MICA dataset (Fig 2B), all recordings were
truncated to the length of the shortest recording (360,136 samples, approximately
180 s at the native sampling rate of 2,000 Hz), so that every spectrum was estimated
from the same amount of data. For the comparison across vigilance states (Fig 2C), the
first 180 s (360,000 samples) of every recording in every state were analyzed. Each
channel was band-pass filtered between 0.5 and 80 Hz with a fourth-order Butterworth
filter applied forward and backward (zero phase), resampled to 200 Hz by polyphase
filtering, and demeaned. The MNI atlas recordings, distributed at 200 Hz, were filtered
and demeaned in the same way without resampling.

Power spectral density (PSD) was estimated with Welch's method (Welch, 1967), using
2-s Hamming windows with 1-s overlap (frequency resolution, 0.5 Hz). Each spectrum was
restricted to 0.5–80 Hz and normalized to unit sum over this range. This normalization
shifts only the aperiodic offset of the spectral parameterization described below and
leaves the exponent and oscillatory peak power unchanged.

## Surface mapping of the MNI atlas (Fig 2A)

Channel coordinates are provided in MNI152 space together with the atlas cortical
surface and its registration to the fsLR template. Each channel was assigned to the
nearest vertex of the atlas surface (both hemispheres pooled), transferred to the
corresponding vertex of the registered surface, and assigned to the nearest vertex of
the fsLR-32k midthickness surface. Because fsLR-32k is a bilaterally symmetric
template, every channel was then placed at the homologous vertex of the right
hemisphere. This assumption of bilateral homology parallels the combination of
hemispheres used for the MICA cohort and maximizes cortical coverage. Each channel was
assigned the value of the bilateral SN gradient of Fig 1 at its vertex. SN channels
whose gradient value lay at or below the 25th percentile, or at or above the 75th
percentile, of the gradient across all SN vertices formed the bottom and top groups,
respectively.

Electrophysiological similarity (ES) was defined from the correlation between power
spectra. Each channel's PSD was z-scored across channels at every frequency, and the
spectra of all pairs of channels were correlated (Pearson). For every channel outside
the SN and the medial wall, $\mathrm{ES}_\text{top}$ and $\mathrm{ES}_\text{bottom}$
were its mean correlations with the top and bottom groups, respectively. The measure of
interest was the difference of their absolute values,
$|\mathrm{ES}_\text{top}| - |\mathrm{ES}_\text{bottom}|$, z-scored across these
channels. It was correlated across channels, using Spearman's rank correlation, with
the second histological gradient of BigBrain (G2; Paquola et al., 2021). Significance
was assessed against a Moran null model defined on the channel layout: because the
channels form scattered points rather than a contiguous surface patch, the spatial
weights were the inverse Euclidean distances between channel positions on the fsLR-32k
midthickness surface, with a minimum distance of 1 mm for channels assigned to the same
vertex. The BigBrain gradient was randomized (1,000 surrogates, seed 0) while the
channel values were held fixed, and a two-sided permutation p-value was computed. Mean
values per network are presented descriptively, without statistical testing.

## Surface mapping of the MICA recordings (Fig 2B)

In the MICA cohort, each electrode contact has a subject-specific signed leadfield
(sensitivity map) computed with the electroMICA pipeline (von Ellenrieder et al.,
2026), defined separately for each hemisphere on the fsLR-32k midthickness surface.
Following electroMICA, the sensitivity of a bipolar channel was computed as the
difference between the signed leadfields of its two contacts, separately within each
hemisphere; the sign is retained in this subtraction because the recorded signal is
the potential difference between the contacts. Following electroMICA, two thresholds
were then applied to the sensitivity density of each channel, defined as the
sensitivity divided by the surface area of each vertex (one third of the area of its
adjacent triangles): a threshold common to all channels reflecting the noise level
($0.001~\text{V}\cdot\text{m/A}$), and a channel-dependent relative threshold of 5% of
the channel's maximum density, taken as its second-largest vertex value as in the
electroMICA implementation (von Ellenrieder et al., 2026). Vertices below either
threshold were set to zero. The rectified sensitivities of the two hemispheres were then summed
onto a single fsLR-32k template, so that contacts in either hemisphere contribute to the
homologous template vertex, which maximizes coverage. Channel measures were projected
onto the surface as a sensitivity-weighted average across channels, so that each
channel contributed to surface vertices in proportion to its modeled spatial
sensitivity rather than through a single nearest-vertex assignment. Contacts without a
leadfield model were assigned zero sensitivity, channels without suprathreshold
sensitivity carried no weight, and vertices covered by no channel were left undefined.
Because each template vertex represents a pair of homologous vertices, maps were
displayed on both hemispheres. All statistics, however, were computed in the left
hemisphere, using the left-hemisphere values of the bilateral SN gradient of Fig 1
([Shared Methods](shared.md#microstructural-profile-covariance-gradients)).

## Spectral parameterization

Each channel spectrum was parameterized once with specparam 2.0.0rc7 (Donoghue et al.,
2020), using its standard spectral fitting algorithm, Gaussian peaks, and an aperiodic
component with a knee,

$$L(f) = b - \log_{10}\!\left(\kappa + f^{\chi}\right),$$

where $L(f)$ is the aperiodic component in log10 power, $b$ the offset, $\kappa$ the
knee parameter and $\chi$ the exponent. The knee parameter was constrained to
$\kappa \ge 0$: a negative knee has no physical interpretation, and with $\kappa = 0$
the model reduces to a power law, appropriate for spectra without a knee in the fitted
range. Spectra were fitted between 1 and 45 Hz. In line with the recommendation not to
fit frequencies attenuated by filtering, this range excludes both the line-noise notch
at 60 Hz present in the recordings and the transition band of the 80 Hz low-pass
filter (attenuation below 0.01 log10 units at 45 Hz). Peak widths were limited to
1–12 Hz (the lower limit being twice the frequency resolution), and at most six peaks
were fitted per spectrum. Peaks had to exceed both an absolute height of 0.05 log10
units above the aperiodic fit and a relative threshold of 2 SD of the flattened
spectrum. The same model and settings were applied to every channel. Goodness of fit
($R^2$) and mean absolute error were computed for each fit, and the fits with the
lowest, median and highest $R^2$ were inspected visually for under- and overfitting. No
fit was excluded on these criteria; only fits that failed to converge were discarded.
The knee frequency, $f_\kappa = \kappa^{1/\chi}$ (Gao et al., 2020), was computed to
locate the knee relative to the fitted range.

## Spectral measures along the microstructural gradient (Fig 2B)

Two measures were derived from the parameterization. The primary measure was the
aperiodic exponent, which has been related to the balance of synaptic excitation and
inhibition (Gao et al., 2017). The secondary measure was the oscillatory peak power in
five canonical frequency bands, each restricted to the fitted range:

| Band | Range |
|------|-------|
| Delta | 1–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–45 Hz |

The peak power of a band was the power of its strongest peak above the aperiodic
component. Because it is measured relative to the aperiodic fit, it is not confounded
by differences in the aperiodic exponent (Donoghue et al., 2020). Channels without a
detected peak in a band were assigned zero power in that band.

Both measures were projected onto the surface and correlated with the MPC gradient,
using Spearman's rank correlation, across the left-hemisphere SN vertices with iEEG
coverage and a defined gradient value. Significance was assessed against the
within-network Moran null model ([Shared Methods](shared.md#moran-spectral-randomization)),
fitted once on this set of vertices with spatial weights given by the inverse 1-ring
edge lengths of the fsLR-32k midthickness surface. The MPC gradient was randomized
(1,000 surrogates, seed 0) while the spectral map was held fixed, and the central 95%
of the null correlations is reported alongside each observed correlation. The five
band correlations formed one family for the correction for multiple comparisons
([Shared Methods](shared.md#spatial-null-models)). For display, each measure was z-scored across the SN vertices of both
hemispheres and shown on a fixed range of ±2 SD, as in Fig 1.

To illustrate the spectra underlying these measures, the covered SN vertices of the
left hemisphere were divided into five equal-sized bins at the quintiles of the MPC
gradient. Within each bin, the sensitivity-weighted power spectrum over the fitted
range was averaged, together with the flattened spectrum, that is, the log10 power
remaining after subtraction of each channel's aperiodic component $L(f)$, projected
onto the surface by the same sensitivity-weighted average.

## Spectral-similarity projection (supporting analysis)

As a supporting analysis, the similarity between regional power spectra, rather than
the temporal coupling between signals, was used as the weight of the projection
defined for Fig 3 ([Figure 3](figure_3.md#connectivity-weighted-projection)). This
measure reflects spectral similarity rather than functional connectivity and was
interpreted accordingly. For every covered vertex of the left hemisphere, a spectral
profile was obtained by sensitivity-weighted averaging of the channel spectra
(0.5–80 Hz) and z-scored across frequencies. The spectral similarity between two
vertices was the positive part of the Pearson correlation between their profiles. For
each SN vertex $i$, the projection score was the similarity-weighted mean of the FC
gradient across its targets,

$$P_i = \frac{\sum_{j} \text{SS}^{+}_{ij}\, g^{\mathrm{FC}}_j}{\sum_{j} \text{SS}^{+}_{ij}}, \qquad j \in \{\text{covered cortical vertices outside the SN}\},$$

where $\text{SS}^{+}_{ij}$ is the non-negative spectral similarity and $g^{\mathrm{FC}}_j$
the FC gradient at target $j$. The FC gradient was that of Fig 3, used at its native
fsLR-32k resolution and oriented by the same rule, with the default-mode network at its
low end. Vertices sampled by overlapping leadfields share a sensitivity-weighted
spectrum and therefore appear similar for methodological rather than neural reasons;
source–target pairs whose sensitivity profiles had a cosine similarity above 0.1 were
therefore excluded. To assess whether spectral similarity contributed information
beyond spatial proximity, the projection was recomputed over the same targets with
uniform weights and with inverse Euclidean distance weights, and the three correlations
are reported together. Vertices with fewer than ten contributing targets were left
undefined.

Because the spectra were pooled across patients, inference was performed at the group
level, as a single Spearman correlation between the MPC gradient and the projection
across SN vertices. Significance was assessed against the within-network Moran null
model, with surrogates of the MPC gradient generated on the SN vertices of the left
hemisphere (1,000 surrogates, seed 0). Both source and target vertices were restricted
to locations with iEEG coverage, a non-uniform sample of the cortex; the projection
therefore samples the FC gradient only where electrodes were implanted, and the group
statistic does not model variability between patients. For display, the projection was
also summarized as its mean within each target network, and the channel-level spectral
similarity matrix (channel spectra z-scored across channels at each frequency and
correlated pairwise) was ordered by the network of each channel's vertex of peak
sensitivity; channels without suprathreshold sensitivity were omitted from this
display.

## Dependence on vigilance state (Fig 2C)

The same bipolar channels of the MICA cohort were recorded during wakefulness and
during N2 and N3 sleep. To estimate all spectra from the same amount of data, the first
180 s of each recording were analyzed in all three states (the shortest wakefulness
recording lasted 180.07 s). Channels were matched across states by patient and channel
name, and only channels recorded in all three states were retained. Each recording was
preprocessed and its spectrum estimated as described above, and each spectrum was
parameterized with the same model and settings as for Fig 2B, with the same quality
checks in every state. Channels whose fit failed in any state were excluded, so that
all states shared the same channels. The aperiodic exponent of each state and its
within-channel change from wakefulness to N3,

$$\Delta\chi = \chi_{\mathrm{N3}} - \chi_{\mathrm{W}},$$

were projected onto the surface as described above. Within the left-hemisphere SN, the
exponent of each state was correlated with the MPC gradient (Spearman) across vertices
with iEEG coverage and a defined gradient value. Significance was assessed against the
within-network Moran null model fitted on this set of vertices
([Shared Methods](shared.md#moran-spectral-randomization); 1,000 surrogates, seed 0),
the central 95% of the null correlations is reported alongside each observed
correlation, and the three per-state correlations formed one family for the correction
for multiple comparisons ([Shared Methods](shared.md#spatial-null-models)). To compare
states, the difference
between the correlations of two states, $\Delta r = r_a - r_b$, was tested for each
pair of states: the same Moran surrogates of the gradient were correlated with the
maps of both states, which preserves the dependence between the two maps as well as
the spatial autocorrelation of the gradient, and the observed difference was compared
with the resulting null distribution (two-sided permutation p-value; the three pairs
formed one family). The within-channel change was tested against the
gradient in the same way; because it is computed within each channel, factors that are
constant across states for a given contact, such as leadfield gain, the distance and
orientation of the contacts relative to the sources, and the bipolar montage, cancel
out. For display, the covered vertices were divided into five equal-sized bins at the
quintiles of the MPC gradient, and the mean exponent of each bin was computed in each
state.

As a supporting analysis, the oscillatory peak power of each frequency band was
examined in the same way. For every band and state, the peak power above the aperiodic
component (zero where no peak was detected, as for Fig 2B) was projected onto the
surface and correlated with the MPC gradient against the within-network Moran null
model, the five bands within each state forming one family. The correlations of each
pair of states were compared with paired Moran surrogates as above, the three pairs
within each band forming one family.
