# Data availability

Everything below was verified against the live services on **2026-08-04** by
`scripts/00_check_environment.py`. Where a claim differs from what the project
assumed at scoping time, that is flagged explicitly, because two of those
differences reshaped the analysis.

---

## 1. Strain

| Run | Detectors | Status | Access |
|-----|-----------|--------|--------|
| O3a | H1, L1 | fully open | `TimeSeries.fetch_open_data` |
| O3b | H1, L1 | fully open | `TimeSeries.fetch_open_data` |
| O4a | H1, L1 | open since 2025-08-26, covering 2023-05-24 to 2024-01-16, 4 kHz and 16 kHz | `TimeSeries.fetch_open_data` |

Verified: 32 s of 4096 Hz strain fetched successfully for L1/O3b, and for both
H1 and L1 in O4a.

Analysis-ready segments come from the GWOSC segment API
(`DataQualityFlag.fetch_open_data("<IFO>_DATA", ...)`), verified for L1 in both
O3b and O4a.

---

## 2. Gravity Spy glitch labels

**There is no public Gravity Spy classification catalogue for O4a.**

This is the first of the two findings that changed the plan. The scoping
assumption was that labelled scattered-light times would be available from
Zenodo for both O3 and O4a. They are not.

| Release | Zenodo | Coverage | Notes |
|---------|--------|----------|-------|
| ML classifications | [5649212](https://zenodo.org/records/5649212) | O1, O2, O3a, O3b | 8 CSVs, ~560 MB, the labelled input this project uses |
| Volunteer classifications | [5911227](https://zenodo.org/records/5911227) | O1, O2, O3a, O3b | citizen-science votes, not used here |
| Volunteer classifications | [13904422](https://zenodo.org/records/13904422) | through 2024-07-20 | volunteer votes keyed by Zooniverse subject id; contains no GPS times, IFO or trigger metadata, so it cannot serve as an O4a trigger list |

The O4a Gravity Spy model is described in Wu et al. (2025), but that paper
releases no classification catalogue. Independent confirmation: the O4a
unsupervised-morphology study of Cirfeta et al. (2026) works directly from
GWOSC strain and clusters it without labels, and uses the **O3b** Gravity Spy
set as its reference index — precisely the workaround that would be
unnecessary if O4a labels existed.

**Consequence.** The labelled analysis is O3-only. The O4a sample is built
from strain by `scripts/05_search_o4a.py`, using an excess-power trigger
finder and a classifier trained on O3. See `docs/DECISIONS.md` (D-02).

The schema of the O3 CSVs is:

```
event_time, ifo, peak_time, peak_time_ns, start_time, start_time_ns,
duration, peak_frequency, central_freq, bandwidth, channel, amplitude,
snr, q_value, gravityspy_id, ml_label, ml_confidence,
<22 class-posterior columns>, url1, url2, url3, url4
```

Triggers are Omicron events with SNR > 7.5 and peak frequency in
[10, 2048] Hz.

---

## 3. Auxiliary channels

Both GWOSC auxiliary releases are served over NDS2 at `nds.gwosc.org`, and can
also be bulk-downloaded over OSDF. Only small windows around trigger times are
needed here, so NDS is used throughout.

### O3 — rich, and richer than assumed

Two sets, listed verbatim in `config/channels_O3_*.csv`:

* **bulk**, 86 channels (40 H1, 46 L1) at full rate;
* **second-trend**, 402 channels (190 H1, 212 L1) at 1 Hz.

The ones this analysis depends on:

| Channel | Rate | Why it matters |
|---------|------|----------------|
| `L1:SUS-OM3_M1_DAMP_L_IN1_DQ` | 256 Hz | OM3 top-stage **length** damping error signal — a genuine optic-motion witness, and the only full-rate one in either release. This is what makes a per-trigger scattering predictor possible at all. |
| `[HL]1:ISI-GND_STS_ITMY_Z_BLRMS_30M_100M` | 16 Hz | corner-station seismometer, 0.03–0.10 Hz earthquake band |
| `[HL]1:ISI-GND_STS_*_{X,Y,Z}_BLRMS_100M_300M.rms` | 1 Hz | **0.1–0.3 Hz microseism band** at ETMX, ETMY, ITMX, ITMY and HAM5 — the driver of slow scattering |
| `[HL]1:ISI-GND_STS_*_BLRMS_{1_3,3_10}.rms` | 1 Hz | 1–3 and 3–10 Hz anthropogenic bands — the driver of fast scattering |
| `L1:PEM-EY_ACC_BEAMTUBE_MAN_Y_DQ` | 4096 Hz | beam-tube accelerometer |

All four representative channels were fetched successfully over NDS.

H1 has **no** full-rate suspension witness in the public O3 release. Its
per-trigger predictor is therefore not available; the H1 auxiliary analysis is
limited to the rate level. This is recorded in `config/analysis.yaml` under
`auxiliary.witness.H1.full_rate_witness_available: false` and is enforced in
code by `scripts/03_aux_correlation.py`.

### O4a — 25 channels, none of them useful for scattering

This is the second finding that changed the plan. The release is 14 H1 + 11 L1
channels, reproduced in full in `config/channels_O4_bulk.csv`. Every one of
them is a calibration line, an input-mode-cleaner wavefront sensor, a PCAL
readout, a mains monitor, or a parametric-instability RMS monitor:

```
H1: CAL-PCALX/Y_RX_PD_OUT_DQ, IMC-WFS_A_DC_{PIT,YAW}_OUT_DQ,
    LSC-POP_A_LF_OUT_DQ, LSC-REFL_A_RIN_OUT_DQ,
    OAF-IMC_WFS_A_DC_{PIT,YAW}_PREFILT_OUT_DQ, OAF-REFL_A_RIN_PREFILT_OUT_DQ,
    PEM-EY_MAINSMON_EBAY_1_DQ, SUS-ETMX_L{1,2,3}_CAL_LINE_OUT_DQ,
    SUS-PI_PROC_COMPUTE_MODE29_RMSMON
L1: ASC-X_TR_A_NSUM_OUT_DQ, CAL-PCALX/Y_RX_PD_OUT_DQ,
    IMC-WFS_B_I_PIT_OUT_DQ, OAF-IMC_WFS_B_I_PIT_PREFILT_OUT_DQ,
    PEM-EX_VMON_ETMX_ESDPOWER24_DQ, PEM-EY_MAINSMON_EBAY_1_DQ,
    SUS-ETMX_L{1,2,3}_CAL_LINE_OUT_DQ, SUS-PI_PROC_COMPUTE_MODE5_RMSMON
```

There is **no seismometer, no accelerometer, no microphone and no suspension
witness**. Neither the rate-level correlation against ground motion nor the
per-trigger scattering predictor can be computed for O4a from public data.

`tests/test_pipeline.py::test_o4_auxiliary_release_has_no_motion_witness`
asserts this. If GWOSC extends the release, that test fails — which is the
intended signal to revisit the O4a strategy.

---

## 3a. Two practical properties of the archive

Both were measured, not assumed, and both changed the analysis design
(`docs/DECISIONS.md` D-12, D-13, D-15).

**Strain is served per 4096 s file.** A request for any sub-interval downloads
the whole containing HDF5 file — timed at 2–4 minutes each on a domestic
connection, essentially independent of how much was requested. The unit of
cost is therefore the file, not the event. The pipeline caches whole blocks and
samples blocks before events; measured yield in a production run is about ten
events per download, against 1.4 for a naive time-stratified sample.

**Not every block exists.** There are genuine gaps. In the first real trial
run, 12 of 40 events failed — all in one block, which serves neither as a whole
nor as a narrow window inside it. Dead blocks are cached negatively so the
retry budget is spent once, and the count is reported with every stage.

**Auxiliary requests are capped at 4 hours, and batching matters.** The GWOSC
NDS server serves a 4-hour request for the second-trend channels but refuses 8
hours or more. Nine channels fetched together take 11.3 s against 40.8 s one at
a time. Availability looked patchy when probed at fixed calendar offsets, but
18 of 18 windows drawn uniformly from analysis-ready livetime returned data —
the apparent gaps were non-observing time. One genuine gap remains at the very
start of O3b (2.26 h of livetime, no auxiliary data).

**The Gravity Spy release lists triggers, not events.** 12.3 % of rows share an
exact `(event_time, detector)`; ~50 % of triggers fall within 8 s of another,
i.e. inside each other's Q-scan window. Clustering at 16 s gives 2.44 triggers
per event for scattered light but only 1.35 for the control classes, so
leaving it uncorrected would over-represent scattering by ~80 % relative to the
controls.

---

## 4. What this means for the study

| Analysis | O3 | O4a |
|----------|----|-----|
| Q-scan morphology | yes, on Gravity Spy labels | yes, on classifier output |
| Rate vs ground motion | yes | **no** — no seismic channels |
| Per-trigger scattering predictor | L1 only | **no** — no motion witness |
| Slow/fast decomposition | yes | yes (strain-only, from arch period) |

The O3-versus-O4a comparison is therefore a **morphology** comparison, made
like-for-like by running the same strain-only search on both runs. The
auxiliary correlation analysis is an O3 result that *calibrates* the
strain-only morphology against a known physical driver; it is what licenses
reading the O4a arch periods as statements about ground motion. That
inference, and its limits, are stated in the discussion of the write-up.

---

## 5. Reference URLs

* O4a data release — <https://gwosc.org/O4/O4a/>
* O4 auxiliary channel release — <https://gwosc.org/O4/auxiliary/>
* O3 auxiliary channel release — <https://gwosc.org/O3/auxiliary/> (DOI [10.7935/3bh6-0k98](https://doi.org/10.7935/3bh6-0k98))
* Channel lists — <https://git.ligo.org/gwosc/tutorials/gwosc-aux-tutorials/-/tree/main/Channels>
* Gravity Spy ML classifications — <https://zenodo.org/records/5649212>
