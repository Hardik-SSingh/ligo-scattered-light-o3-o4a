# Scattered-light transients in LIGO O3 and O4a

Q-scan morphology study of scattered-light glitches across the third and
fourth LIGO observing runs, with an auxiliary-channel correlation analysis
that ties the observed time-frequency morphology to measured ground motion.

Everything runs on **public data only**: GWOSC strain, GWOSC auxiliary
channels over NDS, and the Gravity Spy classification release on Zenodo. No
LIGO credentials are needed.

---

## What the analysis does

Scattered light is one of the most common transient noise sources in the LIGO
detectors. A small fraction of the circulating beam strikes a moving surface
and rejoins the main beam having acquired a phase `phi = 4*pi*x/lambda`. When
that phase excursion exceeds a radian the modulation is strongly non-linear,
and the noise appears as a train of *arches* in the time-frequency plane whose
instantaneous frequency is

```
f_n(t) = n * (2 / lambda) * |v(t)|
```

for surface velocity `v` and harmonic (bounce) order `n`. With
`lambda = 1064 nm` that is 1.88 Hz for every micron per second of surface
motion. Because the frequency follows `|v|` rather than `v`, one cycle of the
driving motion produces **two** arches, so the arch repetition period is half
the period of the motion. That makes the arch period a direct measurement of
what is shaking the interferometer:

```
f_drive = 1 / (2 * T_arch)
```

The pipeline:

1. builds a labelled O3 sample from Gravity Spy;
2. Q-scans each trigger and reduces it to a 20-element morphology feature
   vector: arch repetition period, harmonic spacing and count, ridge slope and
   curvature, and where the energy sits in frequency;
3. correlates the O3 trigger rate against seismometer band-limited RMS, and
   fits the `f = 2n|v|/lambda` predictor to a full-rate suspension witness on a
   trigger-by-trigger basis;
4. trains a strain-only identifier on the O3 features;
5. applies it to a livetime-limited blind search of O4a strain, since no
   labelled O4a catalogue exists;
6. compares the two populations: rate, slow/fast decomposition, and the full
   feature distributions.

## Two findings that shaped the design

Both were established by direct verification before any code was written, and
both are documented in [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).

* **There is no public Gravity Spy catalogue for O4a.** The Zenodo release
  covers O1 to O3b only. The O4a sample is therefore built from strain, using
  an excess-power trigger finder and a classifier trained on O3.
* **The O4a auxiliary release contains no motion witness.** Its 25 channels
  are all calibration lines, wavefront sensors, mains monitors and
  parametric-instability monitors. There is no seismometer, accelerometer or
  suspension channel, so the auxiliary correlation is necessarily an O3
  result, and it is what licenses the physical reading of the O4a arch
  periods.

## Three things the real data forced

Found only by running against the live archive, and each one changes a number
that would otherwise have been reported. All are in
[`docs/DECISIONS.md`](docs/DECISIONS.md) (D-12, D-13, D-15).

* **The catalogue lists triggers, not events.** 12.3 % of rows share an exact
  `(event_time, detector)`, and ~50 % of triggers fall inside another's Q-scan
  window, producing identical feature vectors. Clustering at 16 s gives 2.44
  triggers per event for scattered light but 1.35 for the controls — so
  uncorrected, scattering was over-represented by ~80 % relative to the
  control classes, biasing the classifier and every rate comparison.
* **The unit of cost is the 4096 s file, not the event.** A time-stratified
  sample of 4800 events touched 3484 distinct files, about three days of wall
  clock. Block caching plus block-first sampling brings that to ~10 events per
  download.
* **Not every block exists.** In the first trial run 12 of 40 events failed,
  all in a single unavailable block. Dead blocks are now cached negatively.

The O3 auxiliary release turned out to be considerably richer than assumed:
86 full-rate channels plus 402 second-trend channels, including
`L1:SUS-OM3_M1_DAMP_L_IN1_DQ` at 256 Hz (a genuine optic-motion witness, and
the only full-rate one in either release) and the complete STS
ground-seismometer band-limited-RMS family covering the microseism and
anthropogenic bands at both sites.

---

## Current status

The pipeline is validated end to end against simulated injections and has been
exercised against the live archive at every stage. The production run over the
full O3 and O4a samples has **not** been done — that is the remaining work, and
it is a matter of wall-clock time rather than method. Until it is, the paper's
results section shows red placeholders rather than numbers.

A three-day trial of the auxiliary correlation (L1, O3b, 47 usable hours) does
already reproduce the expected physics, which is the main evidence that the
chain works on real data:

| Ground-motion band | Spearman rho | 95 % block-bootstrap CI | significant |
|---|---|---|---|
| microseism, 0.1–0.3 Hz (5 channels) | 0.61 – 0.76 | excludes zero | yes |
| earthquake, 0.03–0.10 Hz | 0.53 | [0.21, 0.81] | yes |
| anthropogenic, 1–3 and 3–10 Hz | 0.10 – 0.29 | includes zero | no |

The top-to-bottom quartile rate ratio for the strongest channel is 136, and the
partial correlation of rate with the anthropogenic band, controlling for the
microseism, is not significant (rho = 0.25, p = 0.09) — the anthropogenic band
carries no information the microseism does not already supply. This is exactly
the slow-scattering picture Soni et al. (2021) report for L1 in O3.

Treat these as a smoke test, not a result: 47 hours is three days of one
detector, and the lag scan does not yet localise to zero on that little data.

The morphology side has its own non-circular check. On 126 real H1 O3a events
the three independently measured quantities — arch period 0.21 s, harmonic
spacing 15.0 Hz, maximum arch frequency 41.5 Hz — combine to imply a 2.4 Hz
driver, a peak optic velocity of 8 um/s, a displacement amplitude of 530 nm and
2.8 resolved harmonics. Half a micron of optic motion at a few hertz through
two or three bounce orders is an unremarkable description of fast scattering,
which is the point: an extractor measuring its own thresholds would have no
reason to produce three numbers that combine into a sensible displacement.

---

## Installation

```bash
conda env create -f environment.yml
```

```bash
conda activate gwglitch
```

```bash
pip install -e . --no-deps
```

`environment.yml` pins the **OpenBLAS** builds of `libblas`/`libcblas`/
`liblapack`. This is not cosmetic: with MKL 2026.1.0 on Windows every LAPACK
call terminates the interpreter with `STATUS_STACK_BUFFER_OVERRUN` and no
traceback at all. `gwdetchar` is deliberately absent -- its dependency chain
has no Windows build, and the Q-transform is taken from `gwpy` instead. See
[`docs/DECISIONS.md`](docs/DECISIONS.md) D-04 and D-08.

Verify everything before starting:

```bash
python scripts/00_check_environment.py
```

This checks package versions, exercises the linear-algebra backend directly,
parses the configuration, and confirms that GWOSC strain, the NDS auxiliary
server and the segment API all answer for both O3 and O4a.

---

## Running the pipeline

Each stage is standalone, resumable, and writes a provenance sidecar next to
its output.

```bash
python scripts/01_fetch_gravityspy_catalog.py
```

Downloads the four O3 CSVs (about 560 MB) from Zenodo 5649212, applies the
SNR, peak-frequency and ML-confidence cuts, and draws the time-stratified
morphology sub-samples.

```bash
python scripts/02_run_qscans.py --limit 50
```

Q-scans the sampled events and extracts features. This is the long stage, and
its cost is set by **GWOSC file downloads, not events**: open strain is served
in 4096 s HDF5 files, and asking for a 64 s window downloads the whole file
(2-4 minutes on a domestic connection). The pipeline caches whole blocks and
samples blocks before events, which measured out at about ten events per
download against 1.4 for a naive sample — but the full run is still 15-40
hours. It checkpoints every 100 events and skips work already done, so it can
be interrupted freely and resumed overnight. Drop `--limit` for the full run.

```bash
python scripts/03_aux_correlation.py
```

The O3 auxiliary analysis: hourly rate against ground-motion band-limited RMS
with block-bootstrap intervals, a quantile table, a lag scan and a partial
correlation, plus the per-trigger scattering predictor fitted on L1's OM3
witness.

```bash
python scripts/04_train_identifier.py
```

Trains and cross-validates the strain-only identifier, fixes the operating
point at the configured false-positive rate, and saves the model.

```bash
python scripts/05_search_o4a.py --epoch O4a
```

```bash
python scripts/05_search_o4a.py --epoch O3b
```

The blind search. Run it on O4a and on O3b so the comparison is like-for-like:
same trigger generator, same classifier, only the detector differs.

```bash
python scripts/06_population_comparison.py --mode like-for-like
```

```bash
python scripts/07_make_figures.py
```

Stage 06 also writes `paper/numbers.tex` and `paper/generated/*.tex`, so
recompiling the LaTeX picks up the measured values automatically.

---

## Repository layout

```
config/
  analysis.yaml                  every tunable parameter, with the reason for it
  channels_O3_bulk.csv           86 full-rate O3 auxiliary channels, from GWOSC
  channels_O3_second_trend.csv   402 1 Hz trend channels
  channels_O4_bulk.csv           all 25 O4a auxiliary channels
src/gwglitch/
  catalog.py     Gravity Spy download, cuts, stratified sampling
  dataio.py      cached GWOSC/NDS access, segments
  qscan.py       Q-transform maps
  morphology.py  ridge tracking and the 20 features
  scattering.py  the f = 2n|v|/lambda predictor and its calibration
  auxcorr.py     rate/ground-motion correlation statistics
  identify.py    trigger finder and the O3-trained classifier
  population.py  rates, effect sizes, slow/fast decomposition
  simulate.py    synthetic injections with known ground truth
  plotting.py    every figure in the write-up
  latex.py       emits numbers.tex and the generated table bodies
  tables.py      intermediate table I/O
scripts/         the eight pipeline stages
tests/           114 tests, no network access required
paper/           LaTeX write-up, Overleaf-ready
docs/            data availability and the decision log
```

---

## Validation

The morphology extractor is checked against synthetic injections whose
parameters are known exactly. The simulator sums over bounce orders,

```
h(t) = sum_n  K * r^(n-1) * sin(n * 4*pi*x(t) / lambda)
```

so the arches emerge from the phase modulation rather than being drawn on.
Current recovery on the two canonical injections:

| Quantity | Slow injection | Fast injection |
|---|---|---|
| arch period | 3.33 s injected, 3.36 s recovered | 0.250 s injected, 0.250 s recovered |
| harmonic spacing | 14.2 Hz injected, 13.9 Hz recovered | 15.1 Hz injected, 14.0 Hz recovered |
| witness calibration | 1.00e-9 m/count injected, 1.07e-9 recovered | 1.09e-9 recovered |
| envelope explained by the predictor | 95 % | 84 % |

On a pure-noise map the extractor finds nothing at all, which is the
complementary test: peak noise energy reaches Z = 14.6 against a threshold of
Z = 20.

`f_max` reads systematically about 15 % high, because a Q tile of bandwidth
`f/Q` extends above the true apex of the arch. That bias is measured, asserted
in the tests, and stated in the write-up rather than tuned away; it is used
only in the O3-versus-O4a comparison, where it cancels.

```bash
pytest
```

---

## Author

Hardik Sher Singh, Thapar Institute of Engineering and Technology.

## Acknowledgements

This research has made use of data obtained from the Gravitational Wave Open
Science Center (<https://gwosc.org>), a service of LIGO Laboratory, the LIGO
Scientific Collaboration, the Virgo Collaboration and KAGRA. It also uses the
Gravity Spy machine-learning classification data release
([10.5281/zenodo.5649212](https://doi.org/10.5281/zenodo.5649212)) and the
GWOSC O3 auxiliary channel release
([10.7935/3bh6-0k98](https://doi.org/10.7935/3bh6-0k98)).

## Licence

MIT. See [`LICENSE`](LICENSE).
