# CV material — LIGO scattered-light characterisation project

Everything below is accurate as of the current repository state (commit
`88d0b83`). Read the "What you may not claim yet" section at the end before
pasting anything.

---

## Option A — Projects section (recommended, ~4 lines)

**Scattered-Light Transient Characterisation in LIGO O3 and O4a** — *independent
project* · Python, GWpy, scikit-learn ·
[github.com/Hardik-SSingh/ligo-scattered-light-o3-o4a](https://github.com/Hardik-SSingh/ligo-scattered-light-o3-o4a)

- Built an end-to-end detector-characterisation pipeline that measures the
  time–frequency morphology of scattered-light glitches from Q-transform maps
  and links it to ground motion, using only public data (GWOSC strain, GWOSC
  auxiliary channels over NDS, Gravity Spy classifications); 8,700 lines of
  Python across 15 modules, 114 tests, 36-page LaTeX write-up.
- Exploited the scattering relation *f<sub>n</sub> = 2n|v|/λ* — which makes the
  arch repetition period a direct readout of the driving motion,
  *f*<sub>drive</sub> = 1/(2*T*<sub>arch</sub>) — to build a 20-element
  morphology feature vector; validated against synthetic injections with known
  parameters, recovering arch period to <1% and blind-fitting a witness-channel
  calibration to within 9% over an eight-decade search range.
- Established two previously undocumented limitations of the public O4a data
  release (no Gravity Spy classification catalogue; no seismometer,
  accelerometer or suspension witness among its 25 auxiliary channels) and
  redesigned the analysis around them, transferring an O3-trained strain-only
  classifier (ROC AUC 0.93) to O4a.
- Reduced runtime ~7× by discovering that GWOSC serves strain in 4096 s files,
  making the file — not the event — the unit of cost; block-level caching and
  block-first sampling took throughput from 1.4 to ~10 events per download.

---

## Option B — Research Experience section (longer, ~7 lines)

**Independent Research Project — Detector Characterisation, LIGO O3/O4a**
*Thapar Institute of Engineering and Technology* · 2026

Designed and implemented a reproducible pipeline characterising scattered-light
transient noise across LIGO's third and fourth observing runs, working entirely
from public data.

- **Physics.** Scattered light from a moving surface phase-modulates the beam by
  4π*x*/λ, producing harmonic arches at *f<sub>n</sub>* = 2*n*|*v*|/λ. Because
  frequency follows |*v*|, arches recur twice per cycle of the motion, making
  the arch period a strain-only measurement of the mechanical driver.
- **Implementation.** Q-transform maps, ridge tracking and a 20-element
  morphology feature vector (arch period, harmonic spacing and count, ridge
  slope/curvature, spectral distribution); gradient-boosted classifier;
  block-bootstrap and rank-correlation statistics with Benjamini–Hochberg
  correction and Cliff's δ effect sizes.
- **Validation.** Tested against synthetic multi-bounce injections with known
  parameters: arch period recovered to 0.8% (slow) and exactly (fast),
  harmonic spacing to 7%, and a counts-to-metres witness calibration recovered
  blind to within 9%. A pure-noise null test returns zero detections.
- **Auxiliary analysis.** Correlated hourly trigger rate against seismometer
  band-limited RMS using a moving-block bootstrap; in a 47-hour trial the
  microseismic band (0.1–0.3 Hz) gave Spearman ρ = 0.61–0.76 with intervals
  excluding zero while the anthropogenic band did not, reproducing the
  published slow-scattering picture for LIGO Livingston.
- **Data-availability findings.** Verified against live services that no public
  Gravity Spy catalogue exists for O4a and that the O4a auxiliary release
  contains no motion witness — both undocumented elsewhere and both decisive
  for the analysis design; encoded as regression tests so a future release is
  detected.

---

## Option C — One-liner (for a dense CV or a cover letter)

> Built a validated, fully reproducible detector-characterisation pipeline for
> scattered-light noise in LIGO O3/O4a from public data — Q-scan morphology,
> auxiliary-channel correlation, and an O3-trained classifier transferred to
> O4a — with 114 tests and a 36-page write-up.

---

## Skills keywords this project supports

**Domain:** gravitational-wave detector characterisation · transient noise /
glitch classification · time–frequency (Q-transform / Q-scan) analysis ·
scattered-light coupling · seismic and environmental coupling · data quality

**Software:** Python · GWpy · GWOSC / NDS2 · scikit-learn · NumPy · SciPy ·
pandas · Matplotlib · pytest · LaTeX · Git · conda

**Methods:** signal processing (whitening, band-limited RMS, autocorrelation
period estimation) · supervised classification and cross-validation ·
non-parametric statistics (Spearman, moving-block bootstrap, Cliff's δ,
Benjamini–Hochberg, Poisson and Newcombe intervals) · reproducible research
practice (provenance tracking, seeded determinism, regression testing)

---

## Numbers you can quote

| Quantity | Value |
|---|---|
| Scattered-light triggers in O3 after cuts | 119,162 |
| Distinct events after 16 s clustering | 48,824 |
| Python lines / modules / tests | 8,704 / 15 / 114 |
| Write-up length | 36 pages, LaTeX, 30 references |
| Classifier ROC AUC (real O3 features, 5-fold CV) | 0.93 |
| Arch-period recovery vs injection | 0.8% error (slow), exact (fast) |
| Blind witness-calibration recovery | within 9%, eight-decade search |
| Microseism rate correlation (47 h trial) | Spearman ρ = 0.61–0.76 |
| Throughput improvement from block caching | 1.4 → ~10 events per download |

---

## What you may NOT claim yet

The production run over the full O3 and O4a samples **has not been executed**.
So do not write, or let a reader infer:

- ✗ "Measured the scattered-light population across O3 and O4a"
- ✗ "Found that the slow-scattering fraction fell between O3 and O4a"
- ✗ "Produced the first public scattered-light catalogue for O4a"
- ✗ Any O3-vs-O4a comparison as a completed result

All of those become true after `scripts/02_run_qscans.py` and
`scripts/05_search_o4a.py` finish (15–40 h and 4–10 h respectively, both
resumable). Until then the honest framing is **"built and validated"**, not
**"measured"** — which is what Options A–C above use throughout.

The auxiliary correlation numbers (ρ = 0.61–0.76) come from a 47-hour trial on
one detector. If you quote them, say "in a 47-hour validation trial". They are
strong evidence the pipeline works, not a published-scale result.

Once the production run completes, swap in:
- the O3-vs-O4a slow-fraction change with its Newcombe interval;
- the per-hour rates with Poisson intervals;
- the full-run correlation coefficients.

---

## Suggested placement in your CV

Given this is one of three GW portfolio pieces, put it under **Projects** or
**Research Experience** alongside the PyCBC matched-filtering pipeline and the
InPTA/hasasia sensitivity work. Its distinguishing value for a detector-
characterisation or instrumentation-leaning group (AEI Potsdam, Cardiff) is
that it is the only one of the three demonstrating **instrument-facing** skills:
auxiliary channels, environmental coupling, and data quality — rather than
astrophysical search or inference.

For Birmingham and CITA, lead instead with the statistical-methods and
software-engineering bullets (Options A bullets 2 and 4).
