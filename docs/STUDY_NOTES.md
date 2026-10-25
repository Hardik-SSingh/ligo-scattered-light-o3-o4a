# Study notes: scattered-light transients in LIGO O3 and O4a

*Self-contained notes for understanding — and defending — this project.
Written to be read start to finish once, then used as a reference. Every
citation key in square brackets matches an entry in `paper/references.bib`,
so anything you say from here can be backed by the paper.*

Repository: [`Hardik-SSingh/ligo-scattered-light-o3-o4a`](https://github.com/Hardik-SSingh/ligo-scattered-light-o3-o4a).
State as of these notes: pipeline built, validated on injections, exercised on
real data at every stage but one; production run not yet executed.

---

## 0. The sixty-second version

Memorise this. It is the answer to "tell me about this project".

> Scattered light is one of the commonest transient noise sources in LIGO. A
> little of the laser beam hits a moving surface and rejoins the main beam,
> phase-modulated by 4π*x*/λ. When that phase exceeds a radian the noise shows
> up as a train of *arches* in a time–frequency plot, at frequency
> *f* = 2*n*|*v*|/λ. Because it depends on |*v*|, the arches recur twice per
> cycle of whatever is shaking the optic — so the **arch repetition period is
> a direct measurement of the driving motion, from strain alone**.
>
> I built a pipeline that measures that morphology for tens of thousands of
> events across LIGO's third and fourth observing runs, using only public data,
> and ties it to ground motion through the auxiliary seismometer channels. Two
> things I found about the public archive shaped the design: there is no
> Gravity Spy label catalogue for O4a, and the O4a auxiliary release has no
> motion witness at all. So O4a has to be done from strain, with a classifier
> trained on O3, and the O3 auxiliary analysis is what licenses reading the
> O4a arch periods physically.
>
> The pipeline is validated against injections with known parameters — arch
> period recovered to under 1%, a blind witness calibration to within 9% — and
> a 47-hour trial on real data reproduces the published slow-scattering
> picture: microseism correlates, anthropogenic motion doesn't. The production
> run is the remaining step; it's wall-clock, not method.

---

## 1. The physics, derived from nothing

You should be able to reproduce every equation here on a whiteboard.

### 1.1 Where the phase comes from

A gravitational-wave interferometer measures the phase of light. Suppose a
tiny fraction of the circulating beam leaves the main path, hits a surface at
displacement *x*(*t*) along the beam, and comes back. It has travelled an extra
distance 2*x* (out and back), so its phase relative to the main beam is

    φ(t) = (2π / λ) · 2x(t) = 4π x(t) / λ                          (1)

with λ = 1064 nm, the Advanced LIGO Nd:YAG wavelength [AdvancedLIGO2015].

**Check the size.** For φ = 1 rad you need *x* = λ/4π ≈ 85 nm. Suspended
optics and baffles move by microns. So the modulation is *strongly*
non-linear in practice — that is the whole reason scattered light looks the
way it does.

### 1.2 Why it makes arches

The scattered field adds to the main beam a term proportional to sin φ(*t*).
For a signal whose phase varies in time, the instantaneous frequency is
(1/2π) dφ/d*t*. From (1):

    f₁(t) = (1/2π) |dφ/dt| = (2/λ) |v(t)|,   v = dx/dt                (2)

Numbers: 2/λ = 2 / (1.064 × 10⁻⁶ m) = 1.88 × 10⁶ Hz per (m/s). So

> **1 μm/s of surface velocity ⇒ 1.88 Hz of noise frequency.**

If the surface moves sinusoidally, *x* = *X*₀ sin(2π*f*ₘ*t*), then
*v* = 2π*f*ₘ*X*₀ cos(2π*f*ₘ*t*), and *f*₁(*t*) rises from zero to a maximum and
back — an arch. The apex is at

    f₁ᵐᵃˣ = (2/λ) · 2π fₘ X₀ = 4π fₘ X₀ / λ                          (3)

### 1.3 Two arches per cycle — the key fact

Equation (2) has |*v*|, not *v*. Velocity is positive on the way out and
negative on the way back; the frequency doesn't care. So every cycle of the
motion produces **two** arches, one per half-cycle, and

    T_arch = 1 / (2 fₘ)      ⇔      f_drive = 1 / (2 T_arch)          (4)

This is the single most important line in the project. It means that
**measuring the arch repetition period in a spectrogram tells you the
frequency of the mechanical disturbance**, without any auxiliary channel.
A 3.3 s arch period means a 0.15 Hz driver — the ocean microseism. A 0.25 s
arch period means 2 Hz — human activity, machinery, control loops.

That is what lets the O4a half of the study exist despite the O4a auxiliary
release containing no motion witnesses.

### 1.4 Where the harmonics come from (and a trap)

Real Q-scans show several arches stacked: *f*, 2*f*, 3*f*… Where do they come
from?

**The right answer:** light can bounce *n* times between the scattering
surface and the main optic before rejoining. Each round trip adds another
4π*x*/λ of phase, so the *n*-th bounce order has phase *n*φ and instantaneous
frequency

    fₙ(t) = n (2/λ) |v(t)|                                             (5)

with successively weaker amplitude. That is the comb.

**The trap (I fell into it, see D-09):** you might think a single
sin(*β* sin θ) term with large *β* = 4π*X*₀/λ produces harmonics by itself,
via the Jacobi–Anger expansion. It does — but they are harmonics of the
*motion* frequency *f*ₘ, and they terminate at *k* ≈ *β*. Since *β·f*ₘ is
exactly *f*₁ᵐᵃˣ, a one-bounce model produces **one arch**, however large *β*
is. Making the displacement bigger makes that single arch taller, not more
numerous. The simulator sums explicitly over bounce orders for this reason.

### 1.5 Two populations

Soni et al. [Soni2021Scattering] identified two distinct scattering
populations in O3:

| | Driver | Band | Arch period | Arches reach |
|---|---|---|---|---|
| **Slow** | ocean microseism | 0.1–0.3 Hz | 1.7–5 s | ~10–60 Hz |
| **Fast** | anthropogenic / control-loop | 1–6 Hz | 0.08–0.5 s | ~10–120 Hz |

The project's split is at *T*_arch = 1.5 s (`slow_fast_split_period` in
`config/analysis.yaml`). At L1 in O3 the dominant slow path was relative
motion of a few microns between the main suspension chain and its reaction
chain [Soni2021Scattering]; at L1 in O4 both microseismic and vertical
ground-motion paths were found and mitigated [Nandi2026].

### 1.6 Numbers to carry in your head

| Quantity | Value | Source |
|---|---|---|
| λ | 1064 nm | [AdvancedLIGO2015] |
| 2/λ | 1.88 MHz per m/s = 1.88 Hz per μm/s | eq. (2) |
| 1 rad of phase | 85 nm displacement | eq. (1) |
| microseism band | 0.10–0.30 Hz | config |
| anthropogenic band | 1–6 Hz | config |
| earthquake band | 0.03–0.10 Hz | config |

---

## 2. The data — what exists, what doesn't

All verified against the live services on 2026-08-04 by
`scripts/00_check_environment.py`. Details in `docs/DATA_AVAILABILITY.md`.

### 2.1 Observing runs

| Epoch | GPS start | GPS end | UTC | Strain public? |
|---|---|---|---|---|
| O3a | 1238166018 | 1253977218 | 2019-04-01 → 2019-10-01 | yes [GWOSC2023] |
| O3b | 1256655618 | 1269363618 | 2019-11-01 → 2020-03-27 | yes [GWOSC2023] |
| O4a | 1368975618 | 1389456018 | 2023-05-24 → 2024-01-16 | yes, since 2025-08-26 [GWOSCO4aStrain] |

GPS time = seconds since 1980-01-06 with no leap seconds; through O3 and O4
the UTC offset is a constant 18 s (`gwglitch.utils`).

### 2.2 Labels: Gravity Spy — O3 only

Gravity Spy [Zevin2017] classifies Omicron [Robinet2020] triggers (SNR > 7.5,
peak frequency 10–2048 Hz) into 22 morphological classes using a CNN plus
citizen-science labelling. The machine-learning classification release is
Zenodo record 5649212 [GravitySpyData2021; Glanzer2023], one CSV per detector
per epoch, covering **O1, O2, O3a, O3b**.

**There is no O4a catalogue.** The O4a Gravity Spy model [Wu2025] was
published without one. The volunteer release extending to 2024 is keyed by
Zooniverse subject ID with no GPS times. Independent confirmation: the O4a
morphology study of [Cirfeta2026] works unlabelled from raw strain and uses
the O3b Gravity Spy set as its reference — exactly the workaround you'd skip
if O4a labels existed.

Cuts applied (identical for target and control classes): SNR ≥ 7.5, peak
frequency 10–2048 Hz, ML confidence ≥ 0.90, inside the epoch.

What the real catalogue contains after cuts:

| Detector | Epoch | Triggers | Scattered light | Fraction |
|---|---|---|---|---|
| H1 | O3a | 65,457 | 9,436 | 14.4 % |
| H1 | O3b | 82,324 | 58,739 | **71.4 %** |
| L1 | O3a | 87,082 | 6,305 | 7.2 % |
| L1 | O3b | 137,938 | 44,682 | 32.4 % |
| **total** | | **372,801** | **119,162** | |

H1 in O3b was dominated by scattering. That is a real result of the data,
not something I imposed.

### 2.3 Auxiliary channels: rich for O3, useless for O4a

**O3** [GWOSCO3Aux]: 86 full-rate channels (40 H1, 46 L1) plus 402 one-hertz
"second-trend" channels (190 H1, 212 L1). The ones that matter:

| Channel | Rate | Role |
|---|---|---|
| `L1:SUS-OM3_M1_DAMP_L_IN1_DQ` | 256 Hz | OM3 top-stage **length** damping error signal — a genuine optic-motion witness, the only full-rate one in either release. Used for the per-trigger predictor. |
| `[HL]1:ISI-GND_STS_*_BLRMS_100M_300M.rms` | 1 Hz | ground seismometer, **microseism band**, at ETMX/ETMY/ITMX/ITMY/HAM5, x/y/z |
| `[HL]1:ISI-GND_STS_*_BLRMS_{1_3,3_10}.rms` | 1 Hz | **anthropogenic** bands |
| `[HL]1:ISI-GND_STS_ITMY_Z_BLRMS_30M_100M` | 16 Hz | earthquake band |

H1 has **no** full-rate suspension witness in the public release, so the
per-trigger predictor is L1-only.

**O4a** [GWOSCO4Aux]: 25 channels — 14 H1, 11 L1. Every one is a photon
calibrator readout, an input-mode-cleaner wavefront sensor, a length-sensing
readout, a mains monitor, an electrostatic-drive monitor, a calibration-line
drive, or a parametric-instability RMS monitor. **No seismometer,
accelerometer, microphone or suspension channel.** A test
(`test_o4_auxiliary_release_has_no_motion_witness`) asserts this so a future
release is detected.

### 2.4 Three archive properties learned by running

1. **Strain is served in 4096 s HDF5 files.** Any sub-interval request
   downloads the whole file: 2–4 minutes on a domestic connection, regardless
   of how much you asked for. The unit of cost is the file.
2. **Not every file exists.** In the first trial, 12 of 40 events failed, all
   inside one block GWOSC does not serve.
3. **NDS auxiliary requests are capped at ~4 hours** (4 h works, 8 h fails).
   Nine channels in one request cost 11.3 s; one at a time, 40.8 s. When
   probed uniformly in analysis-ready livetime, 18/18 windows had trend data —
   the apparent gaps at fixed calendar offsets were non-observing time.

---

## 3. The pipeline, stage by stage

Eight standalone scripts in `scripts/`, each resumable and writing a JSON
provenance sidecar. Library code in `src/gwglitch/`.

```
00 check_environment      packages, LAPACK, config, live services
01 fetch_gravityspy       download, cut, cluster, block-sample       → catalog.py
02 run_qscans             Q-scan every event, extract 20 features    → qscan.py, morphology.py
03 aux_correlation        rate vs ground motion; witness predictor   → auxcorr.py, scattering.py
04 train_identifier       O3-trained classifier                       → identify.py
05 search_o4a             blind strain-only search (O4a and O3b)     → identify.py
06 population_comparison  rates, slow/fast, feature comparison;      → population.py, latex.py
                          writes paper/numbers.tex + generated tables
07 make_figures           every figure in the paper                   → plotting.py
```

### 3.1 Stage 01 — from catalogue to sample (`catalog.py`)

Downloads the four O3 CSVs (~560 MB), applies the cuts, then does two things
that the real data forced:

**Clustering triggers into events** (`cluster_triggers`). The release lists
Omicron *triggers*. 12.3% of rows share an exact `(event_time, detector)` —
the same trigger under several Gravity Spy subject IDs, one per Q-scan
duration view. About half of all triggers fall within 8 s of another, i.e.
inside each other's ±8 s Q-scan window, and therefore produce *identical*
feature vectors. Left alone, the morphology distributions would be weighted by
how many triggers each arch train happened to generate. Clustering at 16 s
(the full analysed window, so no two retained events share a Q-scan) gives:

- scattered light: 119,162 triggers → **48,824 events** (2.44 per event)
- controls: 157,376 → 116,338 (1.35 per event)

That class-dependence means scattering would have been over-represented by
~80% relative to the controls — a bias on exactly the axis being measured.

**Block-first sampling** (`block_stratified_sample`, `sample_within_blocks`).
Because the download unit is the 4096 s file, sample *blocks* first (100 per
detector-epoch, stratified across the run), then up to 12 events within each.
Result: 400 blocks → 2,256 scattered-light events. Controls are drawn from the
*same* blocks (free in downloads; removes an observing-conditions confound),
round-robin across classes so one class can't swamp the sample — the first
attempt was 41/46 `Low_Frequency_Burst`. Final control sample: 3,335 events
across 11 classes.

The honest cost: events in one block share ground-motion conditions, so the
**block count, not the event count, carries the statistical weight.**

### 3.2 Stage 02 — Q-scans and features (`qscan.py`, `morphology.py`)

For each event: fetch ±32 s of 4 kHz strain (through the block cache),
whiten, Q-transform, and analyse only the central **±8 s** — the 24 s of
padding either side is where the whitening and Q-filter transients live, and
it is discarded. Threshold, trace ridges, compute 20 features. See §4 and §5.

Processes in `(detector, time)` order so consecutive events share a cached
block. Measured: ~10 events per download.

### 3.3 Stage 03 — auxiliary correlation (`auxcorr.py`, `scattering.py`)

Two levels. **Rate level:** bin the *unclustered* trigger list into hours,
divide by analysis-ready livetime, rank-correlate against each seismometer
band. **Trigger level (L1 only):** fit the predictor of eq. (5) to the OM3
witness. See §6.

### 3.4 Stage 04 — the identifier (`identify.py`)

Gradient-boosted trees [Friedman2001; Pedregosa2011] on the 20 features,
scattered light vs the 11 control classes, 5-fold stratified CV, operating
point at FPR = 0.01. See §7.

### 3.5 Stage 05 — the O4a search (`identify.py`)

No O4a trigger list exists, so: draw 400 windows of 512 s per detector
uniformly in livetime (~2.4 days each, ~1.4% of the run), find excess-power
candidates with a robust z-score on whitened 10–120 Hz strain, cluster at the
same 16 s, Q-scan, extract the same features, score with the O3 model. Run on
O3b as well, so the comparison is *like-for-like*: same trigger finder, same
classifier, only the detector differs.

### 3.6 Stage 06 — comparison (`population.py`)

Poisson rates per hour of analysed livetime; slow/fast decomposition; feature
by feature KS + Cliff's δ with Benjamini–Hochberg correction; the slow-fraction
change with a Newcombe interval. Writes the numbers straight into the paper.

---

## 4. The Q-transform, properly

The Q-transform [Chatterji2004] tiles the time–frequency plane with
constant-*Q* windows — *Q* = *f*/Δ*f*, so tiles at higher frequency are
shorter in time and wider in frequency. For each tile it reports the
**normalised energy**

    Z = |X|² / ⟨|X|²⟩                                                  (6)

For stationary Gaussian noise *Z* ~ Exp(1), and √(2*Z*) is the single-tile
SNR. The multi-Q version tries several *Q* planes and keeps the loudest.
Implementation: `gwpy.timeseries.TimeSeries.q_transform` [Macleod2021GWpy] —
the same algorithm as LIGO's `gwdetchar` omega scans, which can't be installed
on Windows (D-04).

Configuration, and why each value is what it is:

| Parameter | Value | Reason |
|---|---|---|
| fetched window | ±32 s | whitening and Q filters ring; padding absorbs it |
| analysed window | ±8 s | 16 s holds 3–9 slow arches |
| frequency range | 8–256 Hz | scattering lives 10–120 Hz |
| high-pass | 8 Hz | inside the whitening filter, one transient not two |
| **Q range** | **[4, 16]** | see below |
| **ridge threshold** | **Z = 20** | see below |
| island filter | 40 pixels | speck removal on the interpolated grid |
| tres / fres | 10 ms / 0.25 Hz | output grid |

**Why cap Q at 16 (D-05).** A tile of quality *Q* at frequency *f* lasts
*Q*/(2π*f*). At *Q* = 45 and 45 Hz that is 0.16 s — comparable to the 0.25 s
arch period of fast scattering, so the arches smear into an unresolved line
forest. The multi-Q search *prefers* that plane because it integrates the
quasi-periodic structure coherently. On an injected 0.250 s arch period the
uncapped transform returned 4.99 s — wrong by twenty. Cap at 16: 0.250 s.

**Why Z = 20, not 5.5 (D-06).** A threshold must sit above the *maximum*
energy a noise-only map produces, not the mean. A 16 s map holds ~10⁴
independent tiles; the expected maximum of 10⁴ Exp(1) draws is ln(10⁴) ≈ 9,
and after the multi-Q search it measures ~16. At Z = 5.5 a pure-noise map
produced hundreds of supra-threshold bins and `f_max` saturated at the top of
the band. Z = 20 (single-tile SNR 6.3) clears it; the null test asserts zero
detections on noise.

---

## 5. The twenty features (`morphology.py`)

Full definitions in the paper's Appendix A. Grouped:

**Extent:** `snr_qscan` (√2*Z*max), `energy_total`, `f_peak`, `f_max` (99th
percentile of the envelope), `f_min`, `bandwidth`, `duration`.

**Arch structure:** `n_arches` (prominent envelope maxima), `arch_period`
(**from the autocorrelation of the envelope**, see below),
`arch_period_scatter`.

**Harmonic structure:** `n_harmonics`, `harmonic_spacing` — peaks in a
frequency slice through the apex. The spacing *equals f*₁ᵐᵃˣ, so it measures
peak surface velocity directly.

**Shape:** `ridge_slope_rms`, `ridge_curvature_rms`, `upward_downward_ratio`
(≈1 for symmetric arches; asymmetric for whistles and blips).

**Energy distribution:** `time_symmetry`, `frac_energy_below_40hz`,
`frac_energy_above_100hz`, `spectral_centroid`, `band_concentration`.

Three things to be able to explain:

**Arch period from autocorrelation, not peak spacing (D-07).** When the top
harmonic dips below threshold the envelope drops to the next harmonic down,
creating a spurious local maximum that splits one arch into two. Peak spacing
gave 2.80 s on an injected 3.33 s (16% low). The autocorrelation is immune —
the periodicity is present in every harmonic simultaneously — and gave 3.36 s.

**`f_max` is biased ~15% high and stays that way.** A Q tile has bandwidth
*f*/*Q*, so the supra-threshold region extends above the true apex by roughly
half of that. Measured on injections, asserted in tests, quoted in the paper.
Not tuned away: it applies identically to both runs and cancels in the
comparison, which is the only place `f_max` is used.

**Missing features are informative.** `arch_period` is NaN when fewer than
two arches exist — a blip has no arch period because it has no arches. The
classifier imputes the median *and* carries a missingness indicator.

---

## 6. The auxiliary analysis, statistic by statistic

### 6.1 Rate level

Hourly trigger rate = triggers in the hour ÷ analysis-ready livetime in the
hour. Hours with zero livetime are dropped; hours with zero triggers are
kept as genuine zeros. (Conflating them manufactures a correlation with
anything that tracks duty cycle.)

**Spearman ρ** — rank correlation, robust to the heavy tails of both series.

**Moving-block bootstrap** [Kunsch1989] — hourly rates are strongly
autocorrelated (the microseism varies over days), so the naive standard error
on ρ is far too small. Resample contiguous 6-hour blocks, 5000 times, take the
2.5/97.5 percentiles.

**Quartile rate ratio** — mean rate in the top quarter of ground motion
divided by the bottom quarter. States the effect in units a commissioner can
act on.

**Lag scan** — ρ as a function of lag over ±12 h. Positive lag = rate responds
*after* the ground motion. A mechanical coupling must peak at zero; a peak at
large lag means both series follow a slow common trend (diurnal, weather).

**Partial correlation** — ρ of rate with the anthropogenic band *after
controlling for* the microseism. Tests whether the anthropogenic band carries
independent information.

**What the 47-hour trial gave (L1 O3b, days 12–15):**

| Band | ρ | 95% CI | Significant |
|---|---|---|---|
| microseism 0.1–0.3 Hz, five channels | 0.61–0.76 | excludes zero | yes |
| earthquake 0.03–0.10 Hz | 0.53 | [0.21, 0.81] | yes |
| anthropogenic 1–3, 3–10 Hz | 0.10–0.29 | includes zero | **no** |

Quartile ratio **136**. Partial correlation with anthropogenic controlling for
microseism: ρ = 0.25, *p* = 0.09 — not significant. Lag peaks at +3 h, which
47 hours cannot resolve well. This is the slow-scattering picture of
[Soni2021Scattering] for L1. Say "47-hour trial" if you quote it.

### 6.2 Trigger level — the predictor (`scattering.py`)

Take the OM3 witness, band-pass 0.03–10 Hz (the low corner kills the DC drift
that differentiation would turn into a ramp), differentiate to velocity, apply
eq. (5) for *n* = 1…8, and ask what fraction of the measured arch envelope
lies within max(1.5 Hz, 15%) of some harmonic.

**The one free parameter.** GWOSC publishes auxiliary channels in raw counts
with no calibration. Rather than invent one, a single scale factor α
(metres per count) is fitted on a *training* subset by maximising the mean
explained fraction — logarithmic grid over eight decades, then golden-section
refinement — then **frozen** and applied to a *disjoint validation* subset.
The width of the range over which the objective stays within 1% of maximum is
reported: narrow means the witness genuinely constrains the prediction.

On injections: a witness generated at 10⁹ counts/m came back as
1.07 × 10⁻⁹ m/count (slow) and 1.09 × 10⁻⁹ (fast) — within 9%, blind. A
predictor wrong by three decades gives less than half the agreement, which is
the negative control.

---

## 7. The classifier and the O4a transfer

**Model:** `GradientBoostingClassifier` (400 trees, depth 3, learning rate
0.05) after median imputation with missingness indicators and standard
scaling. Stratified 5-fold CV. Operating point: the score threshold giving
FPR = 0.01 against the control classes on held-out folds, median across folds.

**Result on real O3 features (126 signal / 66 background):**
AUC = 0.93 ± 0.08. Top features: `f_max` (0.23), `ridge_curvature_rms`
(0.17), `ridge_slope_rms` (0.10), `snr_qscan` (0.08).

**The efficiency was meaningless, and why (D-18).** At FPR = 0.01 the
efficiency came out 0.37 ± 0.33 — folds gave 0.00 (at an *infinite*
threshold), 0.16, 0.68, 0.76, 0.24. Arithmetic: with *n* background events per
fold the threshold rests on *p·n* expected false positives; 66 controls over
5 folds is 13 per fold, so *p·n* = 0.13 — the threshold is decided by whether
zero or one event lands above it. The code now warns below *p·n* = 5 and flags
the saved model. The production sample (~3,300 controls) gives *p·n* ≈ 7. The
AUC is threshold-free and unaffected.

**What the transfer assumes.** Every O4a number inherits an O3-derived
definition of the class. That is untestable without O4a labels. The
cross-validated efficiency and FPR are an upper bound on how well the
transfer holds across runs, not an estimate of it. A genuinely novel O4a
morphology resembling scattering would be absorbed silently. Say this
unprompted; it is the honest limitation.

**Two further caveats.** Folds are not grouped in time, so triggers minutes
apart can land in different folds and the AUC is mildly optimistic. And the
trigger finder is not Omicron — which is why the comparison is run
like-for-like, with the same finder on both runs, and only the *ratio*
between runs is meant to be compared with anything external.

---

## 8. Validation — what was checked and what it found

### 8.1 Injections with known parameters (`simulate.py`)

    x(t) = X₀ sin(2π fₘ t + φ)
    h(t) = Σₙ K rⁿ⁻¹ sin(n · 4π x(t)/λ)      n = 1…N bounces

Two canonical events: **slow** (*X*₀ = 8 μm, *f*ₘ = 0.15 Hz, 4 bounces →
apex 14.2 Hz, top 57 Hz, *T* = 3.33 s) and **fast** (*X*₀ = 0.64 μm,
*f*ₘ = 2 Hz, 3 bounces → apex 15.1 Hz, top 45 Hz, *T* = 0.25 s). The
injection occupies a Tukey-windowed 16 s (or 8 s) inside a 64 s stretch —
because a signal filling the whole window is whitened away entirely (the ASD
estimate treats it as the noise floor).

| Quantity | Slow: in → out | Fast: in → out |
|---|---|---|
| arch period | 3.33 → 3.36 s (0.8%) | 0.250 → 0.250 s |
| harmonic spacing | 14.2 → 13.9 Hz | 15.1 → 14.0 Hz |
| arch count | ~5 → 6 | ~32 → 31 |
| f_max | 56.7 → 65.8 Hz (+16%) | 45.4 → 62.8 Hz (+38%) |
| witness calibration | 1.00 → 1.07 ×10⁻⁹ | 1.00 → 1.09 ×10⁻⁹ |
| envelope explained | 95% | 84% |

### 8.2 The null test

Pure noise at Z = 20: zero active bins, zero duration, NaN extents, peak noise
Z = 14.6. At Z = 5.5 the same map gave hundreds of bins.

### 8.3 The closure test on real data

Injections are circular — they test the extractor against the model it
assumes. A non-circular check: the features are over-determined, so ask
whether they agree. On 126 real H1 O3a events the medians were
*T*_arch = 0.21 s, spacing = 15.0 Hz, *f*_max = 41.5 Hz, measured by three
independent routes. Together they imply

- *f*_drive = 1/(2 × 0.21) = **2.4 Hz** — anthropogenic band
- *v*_peak = λ·15.0/2 = **8 μm/s**
- *X*₀ = λ·15.0/(4π·2.4) = **530 nm**
- *f*_max/*f*₁ = **2.8 harmonics**

Half a micron at a few hertz through two or three bounces is an unremarkable
fast-scattering event — which is the point. An extractor measuring its own
thresholds would have no reason to produce three numbers that combine into a
physically sensible displacement. Completeness: arch period measurable for
56% of events, harmonic spacing for 29%.

### 8.4 What only real data revealed

Six defects survived the entire injection suite and appeared within minutes
of live data. Know these — they are the best interview material in the
project because they show judgement, not just execution.

| # | Symptom | Cause | Fix | Decision |
|---|---|---|---|---|
| 1 | First real fetch died on rename | `np.savez_compressed` appends `.npz` to a temp name lacking it | write through a file handle | D-14 |
| 2 | 4800 events = 3 days | GWOSC serves 4096 s files; every event was a whole download | block cache + block-first sampling; 1.4 → ~10 events/download | D-12 |
| 3 | Distributions would be trigger-weighted | catalogue lists triggers; 2.44 vs 1.35 per event by class | cluster at 16 s | D-13 |
| 4 | 12/40 events failed, 8 retries each | one block GWOSC doesn't serve | negative cache for dead blocks | D-15 |
| 5 | every auxiliary request failed | NDS caps at ~4 h; I used 24 h | 4 h batched chunks, livetime-restricted, canary fallback | D-17 |
| 6 | efficiency 0.37 ± 0.33 | *p·n* = 0.13 expected false positives | resolvability guard | D-18 |

Plus one in the paper itself: `\input` cannot be used inside a `tabular`
(its look-ahead breaks the alignment), and a stale PDF hid the failure for
several builds. Generated tables are now whole environments in their own
files, with a column-count check.

---

## 9. Engineering decisions worth being able to defend

All 18 are in `docs/DECISIONS.md` with dates and reversal conditions. The ones
an interviewer might probe:

**D-01 Why scattered light?** Known closed-form coupling mechanism, so the
correlation analysis has a guaranteed non-null result; benchmarkable against
[Soni2021Scattering]; exercises instrument-facing skills the other portfolio
projects don't.

**D-02 Why not run Omicron on O4a?** No Windows build, and 200 days of strain
is not a laptop job. Building triggers from strain with an O3-trained
classifier keeps the comparison possible at the cost of an inherited class
definition — stated, and bounded by the CV numbers.

**D-03 Why keep "auxiliary correlation across O3/O4a" in the title when only
O3 is possible?** Because the O3 auxiliary work is what *licenses* the O4a
interpretation: it establishes, where labels and witnesses both exist, that
arch period tracks ground motion. Without it the O4a arch-period distribution
would be a description of pictures. The absence of O4a witnesses is itself
reported as a finding.

**D-08 OpenBLAS.** MKL 2026.1.0 on Windows kills the interpreter on every
LAPACK call with `STATUS_STACK_BUFFER_OVERRUN` (0xC0000409) and no
traceback. `lstsq`, `solve`, `detrend`, every sklearn fit. Pinned OpenBLAS;
the environment check exercises the backend directly.

**D-10 / D-11 Why livetime-limited, and why two comparison modes?** A full
O4a search is out of reach; rates are per hour of *analysed* livetime with
coverage recorded. Like-for-like (same search on O3b) controls for the trigger
generator; catalogue mode has better labels. Agreement = robust; disagreement
= a selection effect, reported as such.

**D-16 Why round-robin controls?** Shared blocks are scattering-active, so
co-occurring classes dominate — 41/46 `Low_Frequency_Burst` first time. A
classifier trained on that separates scattering from one class and little
else, while the O4a search meets every class.

---

## 10. Honest status

**Done and verified on real data:** stages 01, 02 (partial sample), 03
(47-hour trial), 04, 07. Stage 06 verified end to end in an isolated copy with
synthetic inputs. Stage 05 verified component-wise (trigger finder and
`search_segment` ran on real strain).

**Not done:** the production run. Stage 02 over 400 blocks is 15–40 hours;
stage 05 on O4a and O3b is 4–10 hours each. All resumable. Until then the
paper's results section shows red placeholders — deliberately, so no unfilled
number can pass as a measurement.

**What you may not claim:** any O3-vs-O4a comparison as a result; a
slow-fraction change; "first public O4a scattered-light catalogue". The
honest verb is *built and validated*, not *measured*.

**Scale:** 8,812 lines of Python across 15 modules; 114 tests, no network
needed, ~3 minutes; 36-page LaTeX write-up with 31 references; 18 logged
decisions.

---

## 11. Interview question bank

Answers are what *I* would say. Make them yours.

**Q: Explain scattered light to me like I'm a first-year.**
Some laser light strays, hits something that's moving, and comes back. Its
phase is shifted by 4π*x*/λ. Since λ is a micron and things move by microns,
that phase winds through many radians, and the noise frequency is the rate the
phase winds: 2|*v*|/λ. It sweeps up and down as the surface speeds and slows —
an arch. Twice per cycle, because speed doesn't care about direction.

**Q: Why is the arch period useful?**
It's 1/(2*f*_drive). Measuring it tells me what frequency is shaking the
optic without touching any auxiliary channel. That's how the O4a half of the
project works despite the O4a release having no seismometers.

**Q: Where do the harmonics come from?**
Multiple round trips. *n* bounces means *n* times the phase, so *n* times the
frequency. Not from the Jacobi–Anger expansion — those harmonics are of the
motion frequency and cap at the single arch's apex. I got that wrong first
and the simulator produced one arch no matter what.

**Q: What's a Q-transform and why not a plain spectrogram?**
Constant-Q tiling: tiles at high frequency are short and wide, at low
frequency long and narrow, matching how transient bursts actually look. The
output is normalised energy, Exp(1) under Gaussian noise, so thresholds have a
meaning. I use gwpy's implementation, same as LIGO's omega scans.

**Q: Why cap Q at 16?**
Tile duration is *Q*/(2π*f*). At *Q* = 45 that's longer than a fast-scattering
arch period, so the arches smear — and the multi-Q search *likes* that plane
because it integrates coherently. Measured: 4.99 s for an injected 0.25 s.

**Q: How did you choose the ridge threshold?**
From the noise maximum of a whole map, not the mean. 10⁴ independent tiles
→ expected max ~ln(10⁴) ≈ 9, measured ~16 with the multi-Q search. Z = 20
sits above it. I picked 5.5 first, looked sensible, and it filled `f_max`
with noise.

**Q: What does the classifier actually tell you about O4a?**
That the O3 definition of "scattered light" can be applied to O4a with a
known efficiency and false-positive rate *against O3 control classes*. It
cannot tell me whether the class boundary moved between runs; that needs O4a
labels, which don't exist. I say that unprompted.

**Q: Your efficiency had a ±0.33 error bar. What happened?**
The operating point sits on *p·n* expected false positives. With 66 controls
over five folds that's 0.13 — the threshold was decided by whether zero or one
event fell above it. I added a guard; the production sample gives ~7. The AUC
is threshold-free and was stable at 0.93 throughout.

**Q: Why cluster triggers?**
Because the catalogue lists Omicron triggers, not events. 12% exact
duplicates, half within one Q-scan window of another — identical feature
vectors. And it's class-dependent: 2.44 per event for scattering, 1.35 for
controls, so scattering was over-represented by 80% relative to its own
background. That's a bias on the axis I'm measuring.

**Q: Why are controls drawn from the same blocks as the signal?**
Free in downloads, and it removes a confound — the classifier can't separate
classes by learning what a noisy night looks like. But those blocks are
scattering-active, so co-occurring classes dominate; round-robin sampling
fixes that.

**Q: What's the one free parameter in the auxiliary analysis?**
The counts-to-metres calibration of the witness — GWOSC doesn't publish one.
I fit it on a training subset over eight decades, freeze it, validate on a
disjoint subset, and report how tightly it's constrained. On injections it
comes back within 9%.

**Q: Why a block bootstrap?**
Hourly rates are autocorrelated over days because the microseism is. Naive
standard errors on ρ would be far too small. Six-hour blocks, resampled 5000
times.

**Q: What's the lag scan for?**
A mechanical coupling responds instantly, so ρ must peak at zero lag. A peak
at ±6 h says both series follow the weather, not each other.

**Q: What did you get wrong?**
Six things, all only visible on real data: a cache that never wrote, a
download cost I mis-estimated by 7×, trigger double-counting, dead blocks
burning retries, a server request cap I didn't know about, and an
unresolvable operating point. Each is logged with its symptom, cause, fix and
a test.

**Q: What would you do with more time / compute?**
Run it. Then time-grouped CV folds, full O4a coverage, and a direct
comparison against [Nandi2026] on overlapping times — that would validate the
strain-only classification against an independent, internally-derived
identification.

**Q: Why does this matter to your PhD group?**
It's instrument-facing detector characterisation: auxiliary channels,
environmental coupling, data quality, on public data with everything
checkable. It's the piece of my portfolio that isn't astrophysical search or
inference.

**Q: What's f = 2|v|/λ in numbers?**
1.88 Hz per micron per second. A 40 Hz arch apex is 21 μm/s.

---

## 12. References, matched to `paper/references.bib`

**Instrument and runs**
- [AdvancedLIGO2015] Aasi et al., *Advanced LIGO*, CQG 32 074001 (2015). The detector; λ = 1064 nm.
- [Capote2025] Capote et al., *Advanced LIGO detector performance in the fourth observing run*, PRD 111 062002 (2025). O4 instrument.
- [Davis2021] Davis et al., *LIGO detector characterization in the second and third observing runs*, CQG 38 135014 (2021). The detchar context; what glitches cost.
- [Soni2024O4] Soni et al., *LIGO detector characterization in the first half of the fourth observing run*, arXiv:2409.02831. O4a detchar.
- [Nguyen2021] Nguyen et al., *Environmental noise in Advanced LIGO detectors*, CQG 38 145001 (2021). Coupling paths generally.

**Scattered light** — the core
- [Soni2021Scattering] Soni et al., *Reducing scattered light in LIGO's third observing run*, CQG 38 025016 (2021). **The benchmark.** Two populations; reaction-chain relative motion at L1; the witness-channel predictor.
- [Nandi2026] Nandi et al., *Scattered light noise at LIGO Livingston Observatory during O4*, arXiv:2605.14143. Two populations again in O4; the direct comparison target.
- [Was2021] Was et al., CQG 38 075020 (2021) and [Accadia2010] Accadia et al., CQG 27 194011 (2010). Scattering at Virgo.

**Glitch classification**
- [Zevin2017] Zevin et al., *Gravity Spy*, CQG 34 064003 (2017). The project.
- [Glanzer2023] Glanzer et al., *Data quality up to the third observing run … Gravity Spy glitch classifications*, CQG 40 065004 (2023). The O1–O3b release paper.
- [GravitySpyData2021] Zenodo 10.5281/zenodo.5649212. The catalogue itself.
- [Soni2021GravitySpy] Soni et al., CQG 38 195016 (2021). Gravity Spy features in O3.
- [Wu2025] Wu et al., CQG (2025), arXiv:2401.12913. The O4 model — no catalogue released with it.
- [Cirfeta2026] Cirfeta et al., arXiv:2605.28572. O4a morphology *without labels*; confirms none exist.

**Algorithms and statistics**
- [Chatterji2004] Chatterji et al., CQG 21 S1809 (2004). The Q-transform.
- [Robinet2020] Robinet et al., SoftwareX 12 100620 (2020). Omicron, the trigger generator Gravity Spy uses.
- [Friedman2001] Gradient boosting. [Kunsch1989] Block bootstrap. [Cliff1993] Cliff's δ. [BenjaminiHochberg1995] FDR correction. [Newcombe1998] Proportion-difference intervals.

**Software and data**
- [Macleod2021GWpy] GWpy — the Q-transform, fetching, whitening.
- [Harris2020NumPy], [Virtanen2020SciPy], [Pedregosa2011], [Hunter2007].
- [GWOSC2023] Abbott et al., ApJS 267 29 (2023). O3 open data.
- [GWOSCO3Aux] doi 10.7935/3bh6-0k98. [GWOSCO4Aux], [GWOSCO4aStrain]: gwosc.org.

---

## 13. Glossary

**Arch** — one rise-and-fall of scattering frequency; two per motion cycle.
**Arch envelope** — the highest supra-threshold frequency per time bin; what
the predictor predicts.
**Block** — a 4096 s GWOSC strain file; the unit of download cost.
**BLRMS** — band-limited RMS; seismometer output in a frequency band.
**Bounce order** — number of round trips; sets the harmonic *n*.
**Event vs trigger** — an event is a physical episode; a trigger is one
Omicron detection. Several triggers per event.
**Livetime** — time the detector was observing; the denominator for a rate.
**Microseism** — 0.1–0.3 Hz ground motion from ocean waves; drives slow
scattering.
**NDS** — Network Data Server; how auxiliary channels are fetched.
**Normalised energy Z** — tile energy over its frequency-row mean; Exp(1) for
Gaussian noise; SNR = √(2Z).
**Operating point** — the score threshold set at a chosen false-positive rate.
**Q** — quality factor *f*/Δ*f* of a time–frequency tile.
**Witness** — an auxiliary channel that records the motion of a suspected
scattering surface.
