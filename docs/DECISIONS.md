# Decision log

Each entry records what was decided, why, and what would reverse it. Dates are
the date the decision was fixed.

---

### D-01 — Glitch class: Scattered Light
**2026-08-03.** Chosen over Blip, Koi Fish, Extremely Loud and Tomte.

Scattered light has a *known* coupling mechanism — a moving surface phase-
modulates stray light at `f = 2n|v|/λ` — so a correlation analysis against
ground motion has a guaranteed non-null result rather than an open-ended
search. It is benchmarkable against Soni et al. (2021), and it exercises
detector-characterisation and instrumentation skills that the other two
portfolio projects (matched filtering, PTA sensitivity) do not.

*Reversal condition:* none. Fixed.

---

### D-02 — O4a is analysed without labels
**2026-08-04.** Forced by the finding that no public Gravity Spy catalogue
exists for O4a (`docs/DATA_AVAILABILITY.md` §2).

Options considered:

1. *Run Omicron on O4a.* Rejected: Omicron's dependency chain has no Windows
   build, and running it over 200 days of strain is not a laptop-scale task.
2. *Restrict the study to O3.* Rejected: it discards the comparison that gives
   the project its point.
3. *Build O4a triggers from strain and classify them with an O3-trained
   model.* **Adopted.**

The cost is that every O4a number inherits an O3-derived definition of the
class. That assumption is stated in the write-up, and the cross-validated
efficiency and false-positive rate on held-out O3 folds are the quantitative
statement of how far it can be trusted.

*Reversal condition:* a public O4a trigger catalogue appears.

---

### D-03 — Auxiliary correlation is an O3 result
**2026-08-04.** Forced by the finding that the O4a auxiliary release contains
no seismometer, accelerometer or suspension witness.

The original title promises "auxiliary channel correlation analysis across
O3/O4a". Only the O3 half is possible. Rather than quietly drop it, the
auxiliary analysis is repositioned: it *calibrates* the strain-only morphology
against a measured physical driver in O3, which is what licenses interpreting
the O4a arch-period distribution in terms of ground motion. The absence of O4a
auxiliary data is itself reported, since it is a real limitation of the public
release that a reader of the paper would want to know about.

*Reversal condition:* GWOSC extends the O4a auxiliary release. Guarded by
`tests/test_pipeline.py::test_o4_auxiliary_release_has_no_motion_witness`.

---

### D-04 — `gwdetchar` is not used
**2026-08-04.** `gwdetchar` depends on `lalsuite`, `dqsegdb2` and
`gwdatafind`, none of which has a `win-64` build on conda-forge, so it cannot
be installed in a native Windows environment.

Everything needed from it — the Q-transform and the omega-scan rendering — is
reimplemented on `gwpy.timeseries.TimeSeries.q_transform`, which is pure
Python/SciPy and fully portable. The transform itself is the same algorithm
(Chatterji et al. 2004); only the plotting wrapper differs.

*Reversal condition:* the analysis moves to a Linux machine or WSL, at which
point `gwdetchar` could be added for cross-checking. It is not needed.

---

### D-05 — Q is capped at 16
**2026-08-04.** Discovered during validation against simulated injections.

`gwpy`'s multi-Q search selects the plane of maximum energy. For a scattering
train it will happily select Q ≈ 45, because a high-Q plane integrates the
quasi-periodic line structure coherently and reports enormous energy. But a
tile of quality factor Q at frequency f lasts `Q/(2πf)` seconds — 0.16 s at
Q = 45 and 45 Hz — which is comparable to the 0.25 s repetition period of fast
scattering. The arches are then unresolved, and the measured arch period came
out at 4.99 s against an injected 0.25 s: a factor of twenty.

Capping the range at `[4, 16]` keeps the tile duration well below the arch
period. With the cap the same injection returns 0.250 s.

Guarded by `tests/test_morphology.py::test_fast_scattering_arch_period`.

---

### D-06 — Ridge threshold is Z = 20, not Z = 5.5
**2026-08-04.** Also from validation.

The initial threshold of Z = 5.5 (single-tile SNR 3.3) was chosen to catch
faint upper harmonics. On a *noise-only* map it produced hundreds of
supra-threshold bins and `f_max` saturated at the top of the analysis band.
The reason is that a map of this size contains of order 10⁴ independent tiles,
whose expected maximum is `ln(10⁴) ≈ 9` before the multi-Q search, and
measured maxima on simulated Gaussian noise reach ≈ 16 after it.

Z = 20 (single-tile SNR 6.3) sits above that. A speck filter
(`min_island_tiles = 40`) removes the residue.

Guarded by `tests/test_morphology.py::test_pure_noise_has_no_ridge`.

---

### D-07 — Arch period from autocorrelation, not peak spacing
**2026-08-04.** Peak spacing is fragile in exactly the case that matters: when
the topmost harmonic dips below threshold the envelope drops to the next
harmonic down, creating a spurious local maximum that splits one arch in two.
On the slow injection this biased the period from 3.33 s to 2.80 s (16 % low).

The autocorrelation of the envelope is insensitive to this, because the
periodicity is present in every harmonic simultaneously. It returns 3.36 s
(0.8 % high). The peak count is still reported as the separate feature
`n_arches`.

---

### D-08 — OpenBLAS, not MKL
**2026-08-04.** The conda-forge solve initially selected MKL 2026.1.0, in
which **every LAPACK call terminates the interpreter** with
`STATUS_STACK_BUFFER_OVERRUN` (exit code `0xC0000409`) and no Python
traceback. `numpy.linalg.lstsq`, `numpy.linalg.solve`, `scipy.signal.detrend`
and anything in `sklearn` that fits a regression all die silently.

`environment.yml` pins the OpenBLAS builds of `libblas`, `libcblas` and
`liblapack`. `scripts/00_check_environment.py` check 2 exercises the backend
directly so that a recurrence is caught in one second rather than twelve
hours into a run.

---

### D-09 — Bounce-order model in the simulator
**2026-08-04.** The first version of the simulator produced one arch and no
harmonics however the parameters were tuned, because it conflated two
different things:

* the Jacobi–Anger expansion of a single `sin(β sin θ)` term gives harmonics
  of the *motion* frequency `f_m`, and they terminate at `k ≈ β` — that is, at
  exactly the apex frequency `β f_m` of the single arch;
* the harmonic *arches* in a real Q-scan come from light making `n` round
  trips between surfaces, accumulating `n` times the phase.

The simulator now sums explicitly over bounce orders,
`h = Σ_n K rⁿ⁻¹ sin(n · 4πx/λ)`. Raising the displacement makes one arch
taller; raising the bounce count adds arches.

---

### D-10 — Livetime-limited O4a search
**2026-08-04.** A full O4a search is out of reach on a laptop over a home
connection. The search draws 400 windows of 512 s per detector, uniformly in
livetime — about 2.4 days each, roughly 1.4 % of the run. (Originally 1200;
reduced once the measured GWOSC download cost was known, see D-12.)

All rates are quoted per hour of *analysed* livetime, and the same search is
run on a matched O3 sample so the comparison is like-for-like: the trigger
generator and the classifier are identical on both sides, and only the
detector differs. Coverage fraction is recorded in
`results/tables/*_search_livetime.csv`.

---

### D-11 — Both O3 comparison modes are reported
**2026-08-04.** `scripts/06_population_comparison.py --mode like-for-like`
compares O4a against the same strain-only search run on O3;
`--mode catalogue` compares against the Gravity Spy O3 sample.

The first controls for the trigger generator, the second has better labels.
Where they agree the result is robust; where they disagree the difference is a
selection effect, and is reported as one.

---

### D-12 — Sample GWOSC file blocks, not events
**2026-08-04.** Discovered on first contact with real data, and it changed the
sampling design.

GWOSC serves open strain as HDF5 files of 4096 s. A request for *any*
sub-interval downloads the whole containing file — measured at 60–90 s on a
domestic connection, essentially independent of how much was asked for. A 64 s
Q-scan window therefore costs one full download.

Two consequences:

1. `DataFetcher.strain` now fetches the containing **block**, keeps a
   three-entry in-memory LRU, and slices windows out of it. Measured on six
   real triggers in one block: **1 download instead of 6**, 113 s instead of
   about 11 minutes.
2. The sample is drawn as **blocks first, events within blocks second**
   (`block_stratified_sample`). A time-stratified sample of 4800 individual
   events touched 3484 distinct blocks — 1.4 events per download, roughly
   three days of wall clock. Sampling 100 blocks per detector-epoch and taking
   up to 12 events from each gives 2256 scattered-light events from 400 blocks
   at 5.6 events per block, about 12 hours.

The control sample is drawn from the **same blocks**, which costs no extra
downloads and removes a confound: signal and background then come from the
same stretches of data, so the classifier cannot separate them by learning
what a noisy night looks like.

*The cost, stated plainly:* events from one block share ground-motion
conditions and are correlated, so the effective sample size is smaller than
the row count. The **block** count, not the event count, carries the
statistical weight. Stage 02 also processes in `(detector, time)` order, since
interleaving detectors would thrash the block cache.

---

### D-13 — Cluster triggers into events
**2026-08-04.** Also from the real catalogue, and it is a bias, not just an
efficiency matter.

The Gravity Spy release lists Omicron **triggers**, not events. In the O3
catalogue after cuts:

* 12.3 % of rows share an exact `(event_time, detector)` — the same trigger
  under several Gravity Spy subject IDs, one per Q-scan duration view;
* ~50 % of triggers fall within 8 s of another one, i.e. inside each other's
  ±8 s Q-scan window, so they produce **identical** feature vectors.

Uncorrected, every morphology distribution is weighted by how many triggers a
given arch train happened to produce, so loud, long trains count dozens of
times and quiet ones once — a selection effect on exactly the axis being
measured.

Worse, the effect is class-dependent. Clustering at 16 s gives **2.44 triggers
per event for scattered light** but only **1.35 for the control classes**.
Scattering was therefore over-represented by about 80 % relative to the
controls, which would have biased both the classifier training set and every
rate comparison.

Triggers within 16 s (the full analysed Q-scan window) are collapsed, keeping
the loudest as representative, so no two retained events share a Q-scan. The
same window clusters the O4a search output, so the two runs count the same
kind of thing. The unclustered list is kept as well: the auxiliary rate
correlation legitimately uses *trigger* rate, and the write-up says which is
which.

---

### D-14 — Cache writes go through a file handle
**2026-08-04.** A bug, recorded because the class of mistake is easy to repeat.

`TimeSeriesData.save` and `QMap.save` wrote to `<hash>.npz.tmp` and then
renamed. But `numpy.savez_compressed` **appends `.npz`** to any filename that
does not already end in it, so the data landed in `<hash>.npz.tmp.npz` and the
rename failed on a file that had never been created. The very first real fetch
died on it.

Nothing in the suite caught it because the injection tests never exercised the
cache. Both writers now pass an open file handle, which suppresses the suffix
logic, and `tests/test_pipeline.py` has round-trip tests plus a check that a
failed write leaves nothing at the destination path.

---

### D-15 — Missing GWOSC blocks are cached negatively
**2026-08-04.** GWOSC does not serve every 4096 s block; there are genuine
gaps. In the first real trial run of stage 02, **12 of 40 events failed, all
in a single block** — probing it directly confirmed neither the whole block
nor a narrow window inside it can be fetched.

The failure handling was already correct (log and continue), but the cost was
not: each event in a dead block burned the full four-attempt retry budget on
the block *and* again on the direct-window fallback, and one bad block holds
many events.

`DataFetcher` now remembers blocks that could not be served. The first event
in a dead block still tries the block and then a direct window fetch, in case
only part of the file is missing; every later event in that block fails
immediately with `BlockUnavailable` and no network traffic. The count is
reported in the fetcher statistics and in each stage's provenance sidecar, so
events lost to data gaps are visible rather than silently absent from the
sample.

---

### D-16 — The control sample is class-balanced within blocks
**2026-08-04.** A consequence of D-12 that only showed up once real controls
were extracted.

Drawing controls from the same blocks as the signal is right for two reasons:
it costs no extra downloads, and it removes an observing-conditions confound,
since signal and background then come from identical stretches of data.

But those blocks were chosen *because* they contain scattering, so the classes
that co-occur with scattering dominate them. The first real control extraction
returned **41 `Low_Frequency_Burst` out of 46**. A classifier trained on that
learns to separate scattering from one adjacent class and little else — while
the O4a search it is meant to serve will meet every class.

Controls are now drawn round-robin across the classes present in each block: a
class with three members contributes all three before a class with three
hundred contributes its fourth. The resulting sample spans eleven classes
(Tomte 945, Extremely Loud 662, Koi Fish 538, Low Frequency Burst 339, Blip
339, Whistle 231, No Glitch 182, and the rest in the tens).

Note that this is a sample *for training a discriminator*, not an estimate of
the relative abundance of glitch classes — nothing downstream reads class
fractions off it.

---

### D-17 — Auxiliary fetching: 4-hour batched chunks, restricted to livetime
**2026-08-04.** Three measurements against the live NDS server, each of which
invalidated part of the first implementation.

**The server caps request length.** A 4-hour request for the second-trend
channels succeeds; 8, 12 and 24 hours all fail. The first version used 24-hour
chunks, so *every* request failed and the stage aborted with "no ground-motion
trend data recovered". `chunk_hours` is now 4, and `DataFetcher.MAX_NDS_SPAN`
rejects anything longer with a clear message rather than a network error.

**Batching is worth 3.6×.** Nine trend channels over a 4-hour span cost 11.3 s
requested together against 40.8 s one at a time — the per-request overhead
dominates, not the data volume. `auxiliary_many` now issues a single
`TimeSeriesDict` call for whatever is not already cached, falling back to
per-channel requests only if the batch fails.

**Apparent gaps were mostly non-observing time.** Probing at fixed calendar
offsets suggested the trend channels were missing two-thirds of the time.
Probing 18 windows drawn *uniformly in analysis-ready livetime* instead gave
**18/18 available** — the earlier failures were simply times when the detector
was not observing. Requests are now restricted to chunks that overlap
analysis-ready segments, which removes roughly a quarter of calendar time at
no cost to the analysis, since a chunk with no livetime contributes nothing to
a rate correlation anyway.

A genuine gap does remain at the very start of O3b: the first chunk has 2.26 h
of livetime but no auxiliary data. Empty chunks are now counted and reported
rather than being fatal, and the error message names this case so it is not
mistaken for a fault.

The retry budget on the batch path is cut from four attempts to two: an
auxiliary gap is a permanent property of the release, and the per-channel
fallback will try again anyway. Four attempts on both paths turned one missing
chunk into several minutes of retries.

---

### D-18 — The operating point must be resolvable by the control sample
**2026-08-04.** Found when the classifier was first trained on real features.

With 126 signal and 66 background events the cross-validated AUC was a stable
0.919 ± 0.033, but the efficiency at FPR = 0.01 came out 0.368 ± 0.334 —
individual folds gave 0.000, 0.160, 0.680, 0.760, 0.240. One fold returned a
threshold of infinity.

The cause is arithmetic, not statistical noise in the usual sense. An operating
point can only be placed as finely as the background sample resolves: at a
false-positive rate `p` with `n` background events per fold, the threshold is
set by `p·n` expected false positives. With 66 controls over 5 folds that is
13 per fold and **0.13 expected false positives** — the threshold is decided by
whether zero or one event happens to land above it.

`train_identifier` now computes `p·n` and warns when it falls below 5, and the
saved bundle carries an `efficiency_underresolved` flag that appears in its
summary. The AUC is threshold-free and unaffected, so it remains quotable.

The production control sample (~3300 events, 660 per fold, 6.6 expected false
positives) clears the bar. This matters because the efficiency is one of the
two numbers the write-up uses to bound how far the O3-trained definition can be
trusted on O4a — quoting a number with a ±0.33 spread as if it were a
measurement would misrepresent exactly the assumption it is meant to test.

A related consistency point surfaced at the same time: `search_segment` was
calling the feature extractor with its default island filter rather than the
configured one. The values happen to coincide today, so nothing was wrong — but
had the configuration been changed, O3 and O4a would have been reduced under
different thresholds and the like-for-like comparison would have been measuring
the pipeline instead of the detectors. Both arguments are now read from the
configuration, and a test asserts it.
