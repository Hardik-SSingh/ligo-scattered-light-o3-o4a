"""Configuration, catalogue bookkeeping, segments, statistics and the classifier.

Nothing here touches the network.  Anything that would has its input built by
hand so that the surrounding logic can still be exercised offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gwglitch import load_config
from gwglitch.catalog import hourly_rate, stratified_sample
from gwglitch.constants import FEATURE_NAMES
from gwglitch.dataio import SegmentList
from gwglitch.population import (
    benjamini_hochberg,
    cliffs_delta,
    compare_feature,
    poisson_rate,
    proportion_difference,
)
from gwglitch.utils import gps_to_utc, hash_key, utc_to_gps


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_loads_and_validates(cfg):
    assert cfg.epoch_names == ["O3a", "O3b", "O4a"]
    assert cfg.epochs_for_run("O3") == ["O3a", "O3b"]
    assert cfg.epochs_for_run("O4") == ["O4a"]


def test_config_is_read_only(cfg):
    with pytest.raises(TypeError):
        cfg.qtransform.q_range = [1, 2]


def test_padding_exceeds_analysed_window(cfg):
    """Filter transients must fall outside the region that gets analysed."""
    assert cfg.strain.window_half_width > cfg.strain.qscan_half_width


def test_target_class_is_not_also_a_control(cfg):
    assert cfg.gravityspy.target_label not in list(cfg.sampling.control_classes)


def test_o4a_has_no_gravityspy_files(cfg):
    """Guards the central data-availability fact of this project."""
    assert not any(k.endswith("O4a") for k in cfg.gravityspy.files)


def test_o4_auxiliary_release_has_no_motion_witness(cfg):
    """The O4a auxiliary channel list must not contain a seismic witness.

    If a future GWOSC release adds one, this test fails -- which is the point:
    the whole O4a strategy would then need revisiting.
    """
    path = cfg.resolve(cfg.auxiliary.channel_list_files["O4_bulk"])
    text = path.read_text(encoding="utf-8", errors="replace")
    for token in ("ISI-GND", "_SEIS_", "_ACC_", "SUS-OM", "WIT_L"):
        assert token not in text, f"{token} unexpectedly present in the O4 list"


def test_o3_auxiliary_release_does_have_a_motion_witness(cfg):
    path = cfg.resolve(cfg.auxiliary.channel_list_files["O3_bulk"])
    text = path.read_text(encoding="utf-8", errors="replace")
    assert cfg.auxiliary.witness["L1"]["channel"] in text
    assert "ISI-GND_STS_ITMY_Z_BLRMS_30M_100M" in text


# ---------------------------------------------------------------------------
# GPS time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gps,utc",
    [
        (1238166018, "2019-04-01 15:00:00"),   # O3a start
        (1256655618, "2019-11-01 15:00:00"),   # O3b start
        (1368975618, "2023-05-24 15:00:00"),   # O4a start
        (1389456018, "2024-01-16 16:00:00"),   # O4a end
    ],
)
def test_gps_to_utc_matches_the_published_epoch_boundaries(gps, utc):
    assert gps_to_utc(gps) == utc
    assert utc_to_gps(utc.replace(" ", "T")) == pytest.approx(gps)


def test_epoch_boundaries_in_config_are_consistent(cfg):
    for name in cfg.epoch_names:
        epoch = cfg.epoch(name)
        assert epoch.gps_end > epoch.gps_start
        span_days = (epoch.gps_end - epoch.gps_start) / 86400.0
        assert 100 < span_days < 300


def test_hash_key_is_order_independent():
    a = hash_key({"b": 2, "a": 1})
    b = hash_key({"a": 1, "b": 2})
    assert a == b
    assert hash_key("x") != hash_key("y")


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


def test_segment_livetime_and_containment():
    seg = SegmentList([[100.0, 200.0], [300.0, 350.0]])
    assert seg.livetime == pytest.approx(150.0)
    assert bool(seg.contains(150.0))
    assert not bool(seg.contains(250.0))
    assert not bool(seg.contains(105.0, pad=10.0))


def test_livetime_per_hour_sums_to_total():
    seg = SegmentList([[3500.0, 7300.0]])
    per_hour = seg.livetime_per_hour(0.0, 10800.0)
    assert per_hour.sum() == pytest.approx(seg.livetime)
    assert len(per_hour) == 4


def test_sample_windows_are_inside_segments_and_disjoint():
    seg = SegmentList([[0.0, 5000.0], [10000.0, 15000.0]])
    starts = seg.sample_windows(length=100.0, n=20, seed=1, pad=4.0)
    assert len(starts) == 20
    assert np.all(seg.contains(starts))
    assert np.all(np.diff(starts) >= 100.0)


def test_sample_windows_refuses_impossible_requests():
    seg = SegmentList([[0.0, 50.0]])
    with pytest.raises(ValueError):
        seg.sample_windows(length=512.0, n=1, seed=0)


# ---------------------------------------------------------------------------
# Catalogue bookkeeping
# ---------------------------------------------------------------------------


def _fake_catalog(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(1_238_166_018, 1_238_166_018 + 90 * 86400, size=n)
    return pd.DataFrame(
        {
            "event_time": np.sort(t),
            "detector": rng.choice(["H1", "L1"], size=n),
            "epoch": "O3a",
            "snr": rng.uniform(7.5, 60.0, size=n),
            "hour_bin": np.floor(np.sort(t) / 3600.0).astype(np.int64),
        }
    )


def test_stratified_sample_hits_the_quota_and_spreads_in_time():
    df = _fake_catalog()
    out = stratified_sample(df, n_total=300, seed=1)
    counts = out.groupby(["detector", "epoch"]).size()
    assert set(counts.to_numpy()) == {300}

    # The sample must span most of the parent range, not clump.
    span_in = df["event_time"].max() - df["event_time"].min()
    span_out = out["event_time"].max() - out["event_time"].min()
    assert span_out > 0.9 * span_in


def test_stratified_sample_returns_everything_when_asked_for_too_much():
    df = _fake_catalog(n=100)
    out = stratified_sample(df, n_total=10_000, seed=1)
    assert len(out) == len(df)


def test_hourly_rate_keeps_genuine_zeros_and_drops_dead_hours():
    """Zero triggers in a live hour is data; zero livetime is not."""
    df = pd.DataFrame({"hour_bin": [10, 10, 12]})
    livetime = pd.Series(
        [1.0, 0.0, 1.0, 0.5],
        index=pd.RangeIndex(10, 14, name="hour_bin"),
    )
    out = hourly_rate(df, 10 * 3600, 13 * 3600 + 1, livetime_hours=livetime)

    assert 11 not in out.index          # dead hour dropped
    assert out.loc[13, "count"] == 0    # live but quiet hour kept
    assert out.loc[10, "rate"] == pytest.approx(2.0)
    assert out.loc[12, "rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_poisson_interval_brackets_the_estimate():
    est = poisson_rate(count=100, livetime_hours=50.0)
    assert est.rate == pytest.approx(2.0)
    assert est.lo < est.rate < est.hi
    # 100 counts -> roughly 10 per cent fractional width each side.
    assert (est.hi - est.lo) / est.rate == pytest.approx(0.40, abs=0.10)


def test_poisson_interval_handles_zero_counts():
    est = poisson_rate(count=0, livetime_hours=10.0)
    assert est.rate == 0.0
    assert est.lo == 0.0
    assert est.hi > 0.0


def test_cliffs_delta_signs_and_limits():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(0, 1, 2000)
    assert cliffs_delta(a, b) == pytest.approx(0.0, abs=0.06)
    assert cliffs_delta(a, a + 100) == pytest.approx(-1.0, abs=1e-6)
    assert cliffs_delta(a + 100, a) == pytest.approx(1.0, abs=1e-6)


def test_benjamini_hochberg_is_monotone_and_conservative():
    p = np.array([0.001, 0.01, 0.02, 0.5, 0.9])
    adj = benjamini_hochberg(p)
    assert np.all(adj >= p)
    assert np.all(np.diff(adj) >= -1e-12)
    assert np.all(adj <= 1.0)


def test_benjamini_hochberg_passes_nans_through():
    adj = benjamini_hochberg(np.array([0.01, np.nan, 0.2]))
    assert np.isnan(adj[1])
    assert np.isfinite(adj[0]) and np.isfinite(adj[2])


def test_compare_feature_detects_a_real_shift():
    rng = np.random.default_rng(2)
    a = rng.normal(10.0, 1.0, 3000)
    b = rng.normal(11.0, 1.0, 3000)
    result = compare_feature(a, b, name="test")
    assert result["p_value"] < 1e-10
    assert result["delta"] < -0.3
    assert result["median_a"] < result["median_b"]


def test_compare_feature_is_silent_on_identical_samples():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 3000)
    b = rng.normal(0, 1, 3000)
    result = compare_feature(a, b, name="test")
    assert result["p_value"] > 0.01
    assert abs(result["delta"]) < 0.06


def test_proportion_difference_interval_covers_the_truth():
    out = proportion_difference(k1=300, n1=1000, k2=200, n2=1000)
    assert out["difference"] == pytest.approx(-0.10, abs=1e-9)
    assert out["ci_low"] < -0.10 < out["ci_high"]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def _fake_features(n=800, seed=0):
    """Two separable but overlapping classes in the real feature space."""
    rng = np.random.default_rng(seed)
    rows = []
    for label, offset in [("Scattered_Light", 1.0), ("Blip", -1.0)]:
        block = {
            name: rng.normal(offset * (1 if i < 6 else 0), 1.0, n)
            for i, name in enumerate(FEATURE_NAMES)
        }
        frame = pd.DataFrame(block)
        frame["ml_label"] = label
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_identifier_trains_and_separates():
    from gwglitch.identify import score_frame, train_identifier

    cfg = load_config()
    features = _fake_features()
    bundle = train_identifier(cfg, features)

    assert bundle.cv_auc > 0.9
    assert 0.0 < bundle.threshold < 1.0
    assert bundle.n_signal == 800 and bundle.n_background == 800
    assert set(bundle.feature_importance) == set(FEATURE_NAMES)

    scored = score_frame(bundle, features)
    signal = scored.loc[scored["ml_label"] == "Scattered_Light", "score"]
    background = scored.loc[scored["ml_label"] == "Blip", "score"]
    assert signal.median() > background.median()


def test_identifier_refuses_a_degenerate_training_set():
    from gwglitch.identify import train_identifier

    cfg = load_config()
    features = _fake_features(n=800)
    features.loc[features["ml_label"] == "Blip", "ml_label"] = "Scattered_Light"
    with pytest.raises(ValueError, match="at least 50 examples"):
        train_identifier(cfg, features)


def test_identifier_tolerates_missing_features():
    """NaNs are informative here, so they must be imputed rather than fatal."""
    from gwglitch.identify import train_identifier

    cfg = load_config()
    features = _fake_features()
    rng = np.random.default_rng(5)
    mask = rng.random(len(features)) < 0.3
    features.loc[mask, "arch_period"] = np.nan

    bundle = train_identifier(cfg, features)
    assert np.isfinite(bundle.cv_auc)


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------
#
# These exist because the first contact with real data failed here and nothing
# in the suite noticed: `numpy.savez_compressed` appends ".npz" to any filename
# that lacks it, so writing to "<hash>.npz.tmp" produced "<hash>.npz.tmp.npz"
# and the atomic rename then failed on a file that had never been created.
# Everything upstream was exercised by the injection tests; only the cache
# round trip was not.


def test_timeseries_cache_round_trip(tmp_path):
    from gwglitch.dataio import TimeSeriesData

    original = TimeSeriesData(
        values=np.linspace(-1.0, 1.0, 4096),
        t0=1256655800.0,
        dt=1.0 / 4096,
        channel="L1:GWOSC-4KHZ_R1_STRAIN",
        unit="strain",
    )
    path = tmp_path / "deadbeefcafe1234.npz"
    original.save(path)

    assert path.exists(), "save() did not produce the file it claims to"
    assert not list(tmp_path.glob("*.tmp*")), "a temporary file was left behind"

    loaded = TimeSeriesData.load(path)
    assert loaded.t0 == original.t0
    assert loaded.dt == pytest.approx(original.dt)
    assert loaded.channel == original.channel
    # Stored as float32 to halve the cache size.
    assert np.allclose(loaded.values, original.values, atol=1e-6)


def test_qmap_cache_round_trip(tmp_path, slow_analysis):
    from gwglitch.qscan import QMap

    original, _, _ = slow_analysis
    path = tmp_path / "0123456789abcdef.npz"
    original.save(path)

    assert path.exists()
    assert not list(tmp_path.glob("*.tmp*"))

    loaded = QMap.load(path)
    assert loaded.shape == original.shape
    assert loaded.q == pytest.approx(original.q)
    assert loaded.t0 == pytest.approx(original.t0)
    assert np.allclose(loaded.frequencies, original.frequencies)
    assert np.allclose(loaded.energy, original.energy, rtol=1e-5)


def test_cache_write_is_atomic_under_failure(tmp_path, monkeypatch):
    """A crash mid-write must not leave a truncated file at the real path."""
    from gwglitch.dataio import TimeSeriesData

    data = TimeSeriesData(values=np.zeros(128), t0=0.0, dt=1.0)
    path = tmp_path / "abcdef0123456789.npz"

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", explode)
    with pytest.raises(OSError):
        data.save(path)
    assert not path.exists(), "a failed write left a file at the destination"


# ---------------------------------------------------------------------------
# Trigger clustering
# ---------------------------------------------------------------------------
#
# The Gravity Spy release is a list of Omicron triggers, not of events. In the
# real O3 catalogue 12.3 % of rows share an exact (event_time, detector) and
# about half of all triggers fall within 8 s of another one -- inside each
# other's Q-scan window, therefore yielding identical feature vectors. Without
# clustering, every morphology distribution is weighted by how many triggers a
# given arch train happened to produce.


def test_cluster_triggers_collapses_a_burst():
    from gwglitch.catalog import cluster_triggers

    df = pd.DataFrame({
        "event_time": [100.0, 101.0, 105.0, 400.0, 402.0, 900.0],
        "snr": [10.0, 25.0, 12.0, 8.0, 30.0, 15.0],
        "detector": "L1",
    })
    out = cluster_triggers(df, window=16.0)

    assert len(out) == 3
    # The loudest member represents each cluster.
    assert sorted(out["snr"]) == [15.0, 25.0, 30.0]
    assert sorted(out["cluster_size"]) == [1, 2, 3]


def test_cluster_triggers_keeps_distant_triggers_separate():
    from gwglitch.catalog import cluster_triggers

    df = pd.DataFrame({
        "event_time": [0.0, 100.0, 200.0],
        "snr": [1.0, 2.0, 3.0],
        "detector": "H1",
    })
    assert len(cluster_triggers(df, window=16.0)) == 3


def test_cluster_triggers_chains_a_continuous_episode():
    """A scattering episode longer than the window is still one event."""
    from gwglitch.catalog import cluster_triggers

    times = np.arange(0.0, 300.0, 5.0)          # every 5 s for 5 minutes
    df = pd.DataFrame({
        "event_time": times,
        "snr": np.linspace(10, 20, len(times)),
        "detector": "L1",
    })
    out = cluster_triggers(df, window=16.0)
    assert len(out) == 1
    assert out.iloc[0]["cluster_size"] == len(times)


def test_cluster_triggers_does_not_merge_across_detectors():
    from gwglitch.catalog import cluster_triggers

    df = pd.DataFrame({
        "event_time": [100.0, 100.5],
        "snr": [10.0, 20.0],
        "detector": ["H1", "L1"],
    })
    out = cluster_triggers(df, window=16.0)
    assert len(out) == 2
    assert set(out["detector"]) == {"H1", "L1"}


def test_cluster_triggers_removes_exact_duplicates():
    from gwglitch.catalog import cluster_triggers

    df = pd.DataFrame({
        "event_time": [100.0, 100.0, 100.0],
        "snr": [10.0, 10.0, 10.0],
        "detector": "L1",
    })
    assert len(cluster_triggers(df, window=16.0)) == 1


def test_cluster_window_matches_the_analysed_qscan_window(cfg):
    """No two retained events may share a Q-scan.

    The analysed window is +/- qscan_half_width, so the coalescence window has
    to be at least its full width for that guarantee to hold.
    """
    assert float(cfg.sampling.cluster_window) >= 2 * float(cfg.strain.qscan_half_width)


def test_cluster_triggers_handles_an_empty_frame():
    from gwglitch.catalog import cluster_triggers

    empty = pd.DataFrame(columns=["event_time", "snr", "detector"])
    assert cluster_triggers(empty, window=16.0).empty


# ---------------------------------------------------------------------------
# Missing GWOSC blocks
# ---------------------------------------------------------------------------
#
# GWOSC does not serve every 4096 s block: there are genuine gaps. A block that
# is missing takes every event inside it with it, and without a negative cache
# each of those events burns the full retry budget twice -- once on the block,
# once on the direct fallback. One bad block held 12 of the first 40 events in
# a real trial run.


def _fetcher(cfg, monkeypatch, fail_all=True):
    from gwglitch.dataio import DataFetcher

    fetcher = DataFetcher(cfg)
    calls = {"n": 0}

    def boom(detector, start, end, sample_rate):
        calls["n"] += 1
        if fail_all:
            raise RuntimeError("failed to get data from any source")
        raise AssertionError("should not be reached")

    monkeypatch.setattr(fetcher, "_fetch_open_data", boom)
    return fetcher, calls


def test_dead_block_is_remembered(cfg, monkeypatch):
    from gwglitch.dataio import BlockUnavailable

    fetcher, calls = _fetcher(cfg, monkeypatch)
    t0 = 1238367295.0

    # First event in the block: one attempt on the block, one on the fallback.
    with pytest.raises(Exception):
        fetcher._from_blocks("H1", t0 - 32, t0 + 32, 4096)
    first_round = calls["n"]
    assert first_round == 2

    # Every later event in the same block must fail without touching the
    # network at all.
    for offset in (100.0, 200.0, 300.0):
        with pytest.raises(BlockUnavailable):
            fetcher._from_blocks("H1", t0 + offset - 32, t0 + offset + 32, 4096)
    assert calls["n"] == first_round, "a known-dead block was retried"
    assert fetcher.stats["dead_blocks"] == 1


def test_dead_block_does_not_poison_its_neighbour(cfg, monkeypatch):
    from gwglitch.dataio import BlockUnavailable

    fetcher, _ = _fetcher(cfg, monkeypatch)
    t0 = 1238367295.0
    with pytest.raises(Exception):
        fetcher._from_blocks("H1", t0 - 32, t0 + 32, 4096)

    far = t0 + 3 * fetcher.BLOCK_SECONDS
    with pytest.raises(Exception):
        fetcher._from_blocks("H1", far - 32, far + 32, 4096)
    # Two distinct blocks now known bad, not one marking the whole detector.
    assert fetcher.stats["dead_blocks"] == 2


def test_block_index_is_stable():
    from gwglitch.dataio import DataFetcher

    assert DataFetcher.BLOCK_SECONDS == 4096


def test_control_sampling_is_class_balanced_within_blocks():
    """Controls share the signal's blocks, which are scattering-active.

    Without balancing, whichever class co-occurs with scattering swamps the
    sample: a first real run returned 41 Low_Frequency_Burst out of 46. A
    classifier trained on that separates scattering from one class and little
    else.
    """
    from gwglitch.catalog import GWOSC_BLOCK_SECONDS, sample_within_blocks

    block = 300000
    t0 = block * GWOSC_BLOCK_SECONDS
    labels = ["Low_Frequency_Burst"] * 200 + ["Blip"] * 5 + ["Tomte"] * 5
    df = pd.DataFrame({
        "event_time": t0 + np.arange(len(labels), dtype=float),
        "detector": "H1",
        "ml_label": labels,
        "snr": 10.0,
    })
    out = sample_within_blocks(df, {("H1", block)}, max_per_block=12, seed=1)

    assert len(out) == 12
    counts = out["ml_label"].value_counts()
    # Every class present should appear, and the abundant one must not take all.
    assert set(counts.index) == {"Low_Frequency_Burst", "Blip", "Tomte"}
    assert counts.max() <= 6


def test_control_sampling_can_be_left_unbalanced():
    from gwglitch.catalog import GWOSC_BLOCK_SECONDS, sample_within_blocks

    block = 300000
    t0 = block * GWOSC_BLOCK_SECONDS
    df = pd.DataFrame({
        "event_time": t0 + np.arange(100, dtype=float),
        "detector": "L1",
        "ml_label": ["Blip"] * 100,
        "snr": 10.0,
    })
    out = sample_within_blocks(
        df, {("L1", block)}, max_per_block=7, seed=1, balance_column=None
    )
    assert len(out) == 7


def test_sample_within_blocks_ignores_blocks_it_was_not_given():
    from gwglitch.catalog import GWOSC_BLOCK_SECONDS, sample_within_blocks

    df = pd.DataFrame({
        "event_time": [300000 * GWOSC_BLOCK_SECONDS + 1.0,
                       999999 * GWOSC_BLOCK_SECONDS + 1.0],
        "detector": "H1",
        "ml_label": ["Blip", "Blip"],
        "snr": [10.0, 10.0],
    })
    out = sample_within_blocks(df, {("H1", 300000)}, max_per_block=10, seed=0)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Auxiliary fetching
# ---------------------------------------------------------------------------


def test_auxiliary_span_longer_than_the_server_allows_is_refused(cfg):
    """4 h succeeds against the live NDS server, 8 h does not."""
    from gwglitch.dataio import DataFetcher

    fetcher = DataFetcher(cfg)
    with pytest.raises(ValueError, match="refuses spans"):
        fetcher.auxiliary_many(["X:Y"], 0.0, 8 * 3600.0)


def test_empty_chunk_costs_one_probe_not_one_per_channel(cfg, monkeypatch):
    """A chunk with no auxiliary data must not burn the whole retry budget.

    Measured on the first chunk of O3b before this was fixed: nine channels at
    four attempts each took nine minutes and recovered nothing.
    """
    from gwglitch.dataio import DataFetcher

    fetcher = DataFetcher(cfg)
    calls = {"batch": 0, "single": 0}

    def no_batch(channels, start, end):
        calls["batch"] += 1
        raise RuntimeError("failed to get data from any source")

    def no_single(channel, start, end):
        calls["single"] += 1
        raise RuntimeError("failed to get data from any source")

    monkeypatch.setattr(fetcher, "_fetch_nds_many", no_batch)
    monkeypatch.setattr(fetcher, "_fetch_nds", no_single)

    channels = [f"L1:CHAN_{i}" for i in range(9)]
    out = fetcher.auxiliary_many(channels, 1256655618.0, 1256655618.0 + 14400)

    assert out == {}
    assert calls["batch"] == 1
    assert calls["single"] == 1, "every channel was probed instead of one canary"


def test_a_single_missing_channel_does_not_lose_the_others(cfg, monkeypatch):
    """If the batch fails but the canary works, fetch the rest individually."""
    import numpy as _np
    from gwglitch.dataio import DataFetcher
    from gwpy.timeseries import TimeSeries

    fetcher = DataFetcher(cfg)

    def no_batch(channels, start, end):
        raise RuntimeError("one channel in the batch is missing")

    def single(channel, start, end):
        if channel.endswith("_3"):
            raise RuntimeError("failed to get data from any source")
        return TimeSeries(_np.zeros(int(end - start)), t0=start, dt=1.0)

    monkeypatch.setattr(fetcher, "_fetch_nds_many", no_batch)
    monkeypatch.setattr(fetcher, "_fetch_nds", single)

    channels = [f"L1:CHAN_{i}" for i in range(5)]
    out = fetcher.auxiliary_many(channels, 1257671199.0, 1257671199.0 + 3600)

    assert set(out) == {c for c in channels if not c.endswith("_3")}


def test_cached_auxiliary_channels_are_not_refetched(cfg, monkeypatch, tmp_path):
    import numpy as _np
    from gwglitch.dataio import DataFetcher
    from gwpy.timeseries import TimeSeries

    fetcher = DataFetcher(cfg)
    fetcher.cache_root = tmp_path
    calls = {"n": 0}

    def batch(channels, start, end):
        calls["n"] += 1
        return {c: TimeSeries(_np.zeros(int(end - start)), t0=start, dt=1.0)
                for c in channels}

    monkeypatch.setattr(fetcher, "_fetch_nds_many", batch)
    args = (["L1:A", "L1:B"], 1257671199.0, 1257671199.0 + 3600)

    first = fetcher.auxiliary_many(*args)
    second = fetcher.auxiliary_many(*args)

    assert set(first) == set(second) == {"L1:A", "L1:B"}
    assert calls["n"] == 1, "the second call went back to the network"
    assert fetcher.stats["hits"] == 2


def test_lag_scan_works_on_a_short_series():
    """A fixed 50-point floor returned an all-NaN scan for any trial run.

    A three-day trial gives about 47 usable hours, which is a perfectly good
    sample for a lag scan but was silently rejected, and the caller then died
    on an all-NA idxmax.
    """
    from gwglitch.auxcorr import lagged_correlation

    rng = np.random.default_rng(0)
    x = rng.normal(size=47)
    y = 0.7 * x + 0.7 * rng.normal(size=47)
    out = lagged_correlation(x, y)

    assert out["rho"].notna().all()
    assert int(out.loc[out["rho"].idxmax(), "lag_hours"]) == 0


def test_lag_scan_recovers_an_injected_lag():
    from gwglitch.auxcorr import lagged_correlation

    rng = np.random.default_rng(1)
    driver = rng.normal(size=600)
    rate = np.roll(driver, 3) + 0.3 * rng.normal(size=600)
    out = lagged_correlation(rate, driver, max_lag_hours=8)
    assert int(out.loc[out["rho"].idxmax(), "lag_hours"]) == 3


def test_lag_scan_gives_up_gracefully_on_a_tiny_series():
    from gwglitch.auxcorr import lagged_correlation

    out = lagged_correlation(np.arange(5.0), np.arange(5.0), max_lag_hours=2)
    assert len(out) == 5
    assert out["rho"].isna().all()


def test_o3_and_o4a_features_use_identical_extraction_settings(cfg):
    """The two samples must be reduced under the same thresholds.

    Stage 02 (O3) reads energy_threshold and min_island_tiles from the
    configuration. If the O4a search used the function defaults instead, the
    like-for-like comparison would be measuring the pipeline rather than the
    detectors -- and it would do so silently, because the defaults happen to
    match the configured values today.
    """
    import inspect

    from gwglitch import identify

    source = inspect.getsource(identify.search_segment)
    assert "cfg.qtransform.energy_threshold" in source
    assert "cfg.qtransform.min_island_tiles" in source
    assert "extract_features(qmap, threshold, islands)" in source


def test_identifier_flags_an_unresolvable_operating_point():
    """FPR = 0.01 cannot be placed with a handful of background events."""
    from gwglitch.identify import train_identifier

    cfg = load_config()
    features = _fake_features(n=60)          # 60 signal, 60 background
    bundle = train_identifier(cfg, features)

    assert bundle.efficiency_underresolved
    assert "under-resolved" in bundle.summary()


def test_identifier_does_not_flag_a_well_resolved_operating_point():
    from gwglitch.identify import train_identifier

    cfg = load_config()
    features = _fake_features(n=4000)
    bundle = train_identifier(cfg, features)

    assert not bundle.efficiency_underresolved
    assert "under-resolved" not in bundle.summary()
