"""Detectors: paper-style simulations, online vs batch, memory, edge cases."""
import numpy as np
import pytest

from onlineboxplot import (
    BatchBoxplot, OnlineBoxplot, available_methods, batch_detect, detect,
    expanding_detect, medcouple, online_detect,
)

N = 10_000
WARMUP = 500


def rng(seed=42):
    return np.random.default_rng(seed)


def idx(outliers):
    return {o.index for o in outliers}


def f1(found, truth):
    tp = len(found & truth)
    if not found or not truth or tp == 0:
        return 0.0
    p, r = tp / len(found), tp / len(truth)
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------- medcouple
def test_medcouple_symmetric_near_zero():
    assert abs(medcouple(rng().normal(size=3000))) < 0.05


def test_medcouple_sign_follows_skew():
    g = rng().gamma(2.0, size=3000)
    assert medcouple(g) > 0.1
    assert medcouple(-g) < -0.1


def test_medcouple_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.stattools")
    x = rng().gamma(2.0, size=2000)
    assert medcouple(x) == pytest.approx(float(sm.medcouple(x)), abs=1e-6)


# ---------------------------------------------------------------- batch
def test_batch_tukey_matches_manual_fences():
    x = rng().normal(size=1000)
    q1, q3 = np.percentile(x, [25, 75])
    expect = set(np.flatnonzero((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))))
    assert idx(batch_detect(x, "tukey")) == expect


def test_batch_unknown_rule():
    with pytest.raises(ValueError):
        BatchBoxplot("nope")


@pytest.mark.parametrize("sign", [1, -1])
def test_batch_adjusted_beats_tukey_on_skewed(sign):
    x = sign * rng().gamma(0.3, 10.0, N)
    assert len(batch_detect(x, "adjusted")) < len(batch_detect(x, "tukey")) / 3


# ---------------------------------------------------------------- online: paper simulations
def test_normal_data_has_almost_no_false_alarms():
    x = rng().normal(0, 1, N)
    assert len(online_detect(x, warmup=WARMUP)) / (N - WARMUP) < 0.001


def test_injected_spikes_are_found():
    x = rng().normal(0, 1, N)
    spikes = [2000, 4500, 7000, 9000]
    x[spikes] = [12, -11, 15, 13]
    found = idx(online_detect(x, warmup=WARMUP))
    assert set(spikes) <= found


@pytest.mark.parametrize("sign", [1, -1])
def test_skewed_data_flags_far_fewer_points_than_tukey(sign):
    """Paper, section 5.1: Tukey flags about 10% of heavily skewed data, the
    online boxplot only the far tail. With the default constants (a=-4, b=0.5)
    this is about 3%; the exact share depends on a and b."""
    x = sign * rng().gamma(shape=0.3, scale=10.0, size=N)
    tukey_rate = len(batch_detect(x, "tukey")) / N
    online_rate = len(online_detect(x, warmup=WARMUP)) / (N - WARMUP)
    assert tukey_rate > 0.08
    assert online_rate < tukey_rate / 2
    assert online_rate < 0.05


def test_larger_b_flags_fewer_points_on_right_skew():
    x = rng().gamma(shape=0.3, scale=10.0, size=N)
    rates = [len(online_detect(x, warmup=WARMUP, n_bins=150, b=b)) for b in (0.5, 1, 2, 3)]
    assert rates == sorted(rates, reverse=True) and rates[0] > rates[-1]


@pytest.mark.parametrize("sign", [1, -1])
def test_online_agrees_with_expanding_batch(sign):
    """Paper, Fig. 2 and 6: online results are close to the batch scenario.
    Same rule on both sides, so only the quartile estimate differs. Fine bins
    (150) here; the std-wide bins of the paper's simulation are covered in
    test_paper_reproduction.py."""
    x = sign * rng().gamma(shape=0.3, scale=10.0, size=N)
    on = idx(online_detect(x, warmup=WARMUP, n_bins=150))
    ex = idx(expanding_detect(x, rule="qsm", warmup=WARMUP))
    assert f1(on, ex) > 0.9


@pytest.mark.parametrize("n_bins", [50, 150, "std"])
def test_coarse_bins_do_not_flood_skewed_data_with_false_alarms(n_bins):
    """On right-skewed data with few bins Q1, Q2 (and with 'std' bins, Q3 too)
    fall in the first bin. The box is then widened by one bin, as in the
    author's reference code, instead of flagging the bulk of the data.
    Even without the extra floor the flagged share stays small."""
    x = rng().gamma(shape=0.3, scale=10.0, size=N)
    rate = len(online_detect(x, warmup=WARMUP, n_bins=n_bins, floor_bins=0.0)) / (N - WARMUP)
    assert rate < 0.05


def test_std_bins_match_paper_simulation_scale():
    """Paper 5.1 (std-wide bins): about 0.2% of right-skewed points flagged."""
    x = rng(0).gamma(shape=0.3, scale=10.0, size=6000)
    rate = len(online_detect(x, warmup=WARMUP, n_bins="std")) / (6000 - WARMUP)
    assert rate < 0.005


def test_box_is_widened_when_quartiles_share_a_bin():
    det = OnlineBoxplot(n_bins="std").fit(rng(0).gamma(0.3, 10.0, 500))
    q1, q2, q3 = det.q
    assert q1 < q2 < q3
    assert q3 - q1 == pytest.approx(2 * det.bin_width)


# ---------------------------------------------------------------- NaN
def test_nan_is_skipped_everywhere():
    x = rng().normal(0, 1, 3000)
    x[[10, 600, 1500]] = np.nan
    x[2000] = 30
    for mode, kw in [("online", {"warmup": 300}), ("expanding", {"warmup": 300}), ("batch", {})]:
        found = idx(detect(x, mode=mode, **kw))
        assert 2000 in found
        assert not (found & {10, 600, 1500})


def test_nan_inside_warmup_buffer_is_ignored():
    x = rng().normal(0, 1, 1000)
    x[5] = np.nan
    det = OnlineBoxplot().fit(x[:300])
    assert det._counts.sum() == 299


# ---------------------------------------------------------------- calibration
def test_calibrate_qsm_returns_finite_constants():
    pytest.importorskip("scipy")
    from onlineboxplot.calibrate import calibrate_qsm
    x = rng(0).gamma(0.3, 10.0, 1000)
    p = calibrate_qsm(x, n_bins="std")
    assert set(p) == {"a", "b"} and all(np.isfinite(list(p.values())))


def test_calibrated_detection_runs_and_finds_spike():
    pytest.importorskip("scipy")
    x = rng().normal(0, 1, 3000)
    x[2500] = 25
    assert 2500 in idx(online_detect(x, warmup=300, calibrate=True))


def test_calibrate_rejects_other_rules():
    with pytest.raises(ValueError):
        online_detect(rng().normal(size=1000), rule="tukey", warmup=200, calibrate=True)


# ---------------------------------------------------------------- online: mechanics
def test_constant_memory():
    x = rng().normal(0, 1, 50_000)
    det = OnlineBoxplot(max_bins=300).fit(x[:500])
    peak = 0
    for v in x[500:]:
        det.update(v)
        peak = max(peak, det.n_bins_now)
    assert peak <= 300


def test_quartiles_close_to_exact():
    x = rng().normal(10, 2, 5000)
    det = OnlineBoxplot(n_bins=150).fit(x[:500])
    for v in x[500:]:
        det.update(v)
    assert np.allclose(det._quartiles(), np.percentile(x, [25, 50, 75]), atol=0.3)


def test_histogram_count_equals_points_learned():
    x = rng().normal(size=3000)
    det = OnlineBoxplot().fit(x[:300])
    for v in x[300:]:
        det.update(v)
    assert det._counts.sum() == det._n


def test_chebyshev_point_is_flagged_but_not_learned():
    x = rng().normal(0, 1, 1000)
    det = OnlineBoxplot(k=55).fit(x[:500])
    before = int(det._counts.sum())
    assert det.update(1e6) is True            # far beyond 55 std
    assert int(det._counts.sum()) == before   # histogram untouched
    assert det.update(0.0) is False
    assert int(det._counts.sum()) == before + 1


def test_chebyshev_can_be_switched_off():
    det = OnlineBoxplot(k=None).fit(rng().normal(0, 1, 500))
    assert det.update(1e6) is True
    assert det._counts.sum() == 501


def test_constant_series_does_not_crash():
    det = OnlineBoxplot().fit([5.0] * 100)
    assert det.update(5.0) is False
    assert det.update(500.0) is True


def test_discrete_series_flags_rare_values():
    """Like the paper's 'state of service' metric: almost always 6, rare dips."""
    x = np.full(5000, 6.0)
    x[[1500, 3000]] = [4.0, 2.0]
    assert idx(online_detect(x, warmup=300)) == {1500, 3000}


def test_new_extreme_value_grows_histogram_to_the_left_and_right():
    det = OnlineBoxplot(k=None, n_bins=50, max_bins=200).fit(rng().normal(0, 1, 500))
    det.update(-30.0)
    det.update(40.0)
    assert det._counts.sum() == 502
    assert det._lo <= -30.0 and det._lo + det._width * det.n_bins_now > 40.0


def test_update_before_fit_raises():
    with pytest.raises(RuntimeError):
        OnlineBoxplot().update(1.0)


def test_scores_are_positive_and_rank_by_distance():
    x = rng().normal(0, 1, 3000)
    x[[1000, 2000]] = [8, 30]
    out = {o.index: o for o in online_detect(x, warmup=300)}
    assert out[1000].score > 0 and out[2000].score > out[1000].score


def test_flag_warmup_finds_spike_inside_buffer():
    x = rng().normal(0, 1, 2000)
    x[100] = 25
    assert 100 not in idx(online_detect(x, warmup=500))
    assert 100 in idx(online_detect(x, warmup=500, flag_warmup=True))


# ---------------------------------------------------------------- entry point
@pytest.mark.parametrize("method", available_methods())
def test_every_method_runs_and_finds_a_big_spike(method):
    mode, rule = method.split("-")
    x = rng().normal(0, 1, 2000)
    x[1500] = 40
    found = idx(detect(x, mode=mode, rule=rule, **({"warmup": 300} if mode != "batch" else {})))
    assert 1500 in found


def test_detect_is_deterministic():
    x = rng().gamma(0.3, 10, 3000)
    assert detect(x) == detect(x)


def test_detect_unknown_mode():
    with pytest.raises(ValueError):
        detect([1, 2, 3, 4, 5], mode="nope")
