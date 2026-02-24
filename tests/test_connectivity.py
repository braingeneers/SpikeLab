import numpy as np
import math

from spikes.connectivity import compute_spike_transmission, TransmissionResult


def test_empty_inputs():
    res = compute_spike_transmission(np.array([]), np.array([]))
    assert res.n_pres_spikes == 0
    assert math.isnan(res.z_score)
    assert res.transmission_prob == 0.0


def test_simple_synapse():
    # create pres spikes at 0.1, 0.2, ..., posts spikes shortly after each pres spike
    pres = np.arange(0.1, 1.1, 0.1)
    posts = pres + 0.0015  # 1.5 ms latency
    res = compute_spike_transmission(
        pres, posts, bin_size=0.0005, window=0.01, n_jitter=0, min_pres_spikes=1
    )
    # expecting about 1 posts spike per pres spike -> transmission_prob ~1
    assert res.n_pres_spikes == pres.size
    assert res.observed_peak_count >= pres.size - 1
    assert res.transmission_prob > 0.5
    assert not math.isnan(res.latency_ms)


def test_noise_and_jitter():
    rng = np.random.default_rng(42)
    pres = np.sort(rng.uniform(0, 10.0, size=500))
    # posts are random noise plus a subset that follow pres by 2ms for ~10% of pres
    posts = list(rng.uniform(0, 10.0, size=1000))
    for t in pres[:50]:
        posts.append(t + 0.002)
    posts = np.sort(np.asarray(posts))
    res = compute_spike_transmission(pres, posts, n_jitter=100, random_seed=1)
    assert res.p_value is not None
    assert 0.0 <= res.p_value <= 1.0
    assert res.surrogate_mean is not None


def test_small_number_pres():
    pres = np.array([0.1, 0.2, 0.3])
    posts = pres + 0.001
    res = compute_spike_transmission(pres, posts, min_pres_spikes=5)
    assert res.n_pres_spikes == 3
    assert res.transmission_prob == 0.0


def test_ccg_centers_length():
    pres = np.array([0.1, 0.2, 0.3])
    posts = np.array([0.15, 0.25, 0.35])
    res = compute_spike_transmission(
        pres, posts, bin_size=0.001, window=0.01, min_pres_spikes=1
    )
    centers = res.ccg_bin_centers_ms
    # window 10ms each side, bin 1ms -> 21 bins
    assert centers.size == 21
    assert centers[0] < centers[-1]
