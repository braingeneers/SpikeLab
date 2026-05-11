"""Worked example: evoked-response analysis using new SpikeLab methods.

Demonstrates every method added in the Maxwell-scripts integration PR
using a small synthetic stim-response dataset. Designed to be runnable
end-to-end without external data.

Generates:
    - A SpikeData object with baseline + evoked spikes per unit
    - A SpikeSliceStack aligned to fake stim events
    - Responsive-unit identification
    - Per-unit response-amplitude regression across cycles
    - Slice-to-slice similarity matrices (cosine, pearson, cross_entropy)
    - Per-unit CV / CV2 firing regularity
    - Mixed-effects regression of response amplitude vs treatment

Run with: python examples/maxwell_response_analysis.py
"""

import numpy as np

from spikelab.spikedata.spikedata import SpikeData
from spikelab.spikedata.stat_utils import mixed_effects_compare


def build_synthetic_recording(seed=0):
    """Build a 10-unit SpikeData with baseline spikes plus evoked spikes
    around 10 fake stim events. Units 0-4 are responsive; 5-9 are not."""
    rng = np.random.default_rng(seed)
    recording_length_ms = 10000.0
    n_units = 10
    stim_times_ms = np.linspace(500.0, 9500.0, 10)

    trains = []
    for u in range(n_units):
        # Baseline Poisson at ~10 Hz
        n_baseline = rng.poisson(recording_length_ms * 0.01)
        baseline = np.sort(rng.uniform(0, recording_length_ms, n_baseline))

        # Evoked spikes for responsive units: 5 spikes at 20-30 ms post-stim,
        # decaying linearly with cycle number (facilitation/depression model).
        if u < 5:
            evoked = []
            for cycle, t_stim in enumerate(stim_times_ms):
                amplitude = max(1, 6 - cycle // 2)  # 6, 6, 5, 5, 4, 4, 3, 3, 2, 2
                evoked.append(t_stim + rng.uniform(20.0, 30.0, amplitude))
            evoked = np.concatenate(evoked) if evoked else np.array([])
            train = np.sort(np.concatenate([baseline, evoked]))
        else:
            train = np.sort(baseline)

        trains.append(train)

    return SpikeData(trains, length=recording_length_ms, N=n_units), stim_times_ms


def main():
    sd, stim_times = build_synthetic_recording(seed=42)
    print(
        f"SpikeData: N={sd.N}, length={sd.length} ms, {sum(len(t) for t in sd.train)} spikes"
    )

    # ----- 1. CV / CV2 firing regularity -----
    cv = sd.cv_isi()
    cv2 = sd.cv2_isi()
    print("\n[cv_isi / cv2_isi] Per-unit firing regularity")
    print(f"  cv_isi:  {np.round(cv, 3)}")
    print(f"  cv2_isi: {np.round(cv2, 3)}")
    print("  (Poisson units should have CV ≈ 1; responsive units have evoked")
    print("   bursts that lower CV / raise CV2 slightly.)")

    # ----- 2. Event-aligned SpikeSliceStack via align_to_events -----
    # Each slice covers (-100, +200) ms around its stim event.
    sss = sd.align_to_events(stim_times, pre_ms=100, post_ms=200, kind="spike")
    print(f"\n[align_to_events] SpikeSliceStack: N={sss.N}, S={len(sss)}")

    # ----- 3. responsive_units -----
    mask = sss.responsive_units(
        bin_size=10.0,
        baseline_window_ms=(-100.0, 0.0),
        response_window_ms=(20.0, 30.0),
        z_threshold=2.0,
    )
    print("\n[responsive_units] z>2 in response window:")
    print(f"  responsive: {np.where(mask)[0].tolist()}  (expected: 0..4)")
    print(f"  silent:     {np.where(~mask)[0].tolist()}  (expected: 5..9)")

    # ----- 4. baseline_normalized_raster -----
    norm = sss.baseline_normalized_raster(
        bin_size=10.0,
        baseline_window_ms=(-100.0, 0.0),
        mode="zscore",
    )
    print(f"\n[baseline_normalized_raster] zscore shape: {norm.shape}")
    print(
        f"  responsive units (0..4) peak z mean: "
        f"{np.nanmean(np.nanmax(norm[:5], axis=(1, 2))):.2f}"
    )
    print(
        f"  silent units (5..9)     peak z mean: "
        f"{np.nanmean(np.nanmax(norm[5:], axis=(1, 2))):.2f}"
    )

    # ----- 5. per_unit_response_regression -----
    reg = sss.per_unit_response_regression(
        bin_size=10.0,
        response_window_ms=(20.0, 30.0),
        baseline_window_ms=(-100.0, 0.0),
    )
    print("\n[per_unit_response_regression] slope per unit (response decay vs cycle):")
    for u in range(sd.N):
        sig = "**" if reg["p_value"][u] < 0.05 else "  "
        print(
            f"  unit {u}: slope={reg['slope'][u]:+.3f}  "
            f"p={reg['p_value'][u]:.3f}  r^2={reg['r_squared'][u]:.2f} {sig}"
        )
    print("  (responsive units should have negative slope — decaying response.)")

    # ----- 6. slice_to_slice_similarity -----
    cosine = sss.slice_to_slice_similarity(metric="cosine", bin_size=10.0)
    pearson = sss.slice_to_slice_similarity(metric="pearson", bin_size=10.0)
    ce = sss.slice_to_slice_similarity(metric="cross_entropy", bin_size=10.0)
    print(f"\n[slice_to_slice_similarity] shape: {cosine.matrix.shape}")
    print(f"  cosine    diag: {np.diag(cosine.matrix).round(2)}")
    print(f"  pearson   diag: {np.diag(pearson.matrix).round(2)}")
    print(f"  cross_ent diag: {np.diag(ce.matrix).round(4)}")
    print(
        "  Off-diagonal mean cosine across all slice pairs: "
        f"{np.nanmean(cosine.matrix[np.tril_indices_from(cosine.matrix, k=-1)]):.3f}"
    )

    # ----- 7. mixed_effects_compare -----
    # Build a long-form per-(slice, unit) table:
    #   - response amplitude = reg['amplitudes'][u, s]
    #   - treatment: 'responsive' vs 'silent'
    #   - random effect: unit id (repeated measurements per unit across slices)
    n_units = sd.N
    n_slices = len(sss)
    values = reg["amplitudes"].ravel()  # shape (U*S,), order = (u, s) C-order
    treatment = np.array(
        [
            ("responsive" if u < 5 else "silent")
            for u in range(n_units)
            for _ in range(n_slices)
        ],
        dtype=object,
    )
    unit_id = np.array(
        [f"u{u}" for u in range(n_units) for _ in range(n_slices)], dtype=object
    )
    result = mixed_effects_compare(
        values,
        {"treatment": treatment},
        unit_id,
    )
    treatment_terms = [k for k in result["params"] if "treatment" in k]
    print("\n[mixed_effects_compare] Response amplitude ~ treatment, unit random:")
    for t in treatment_terms:
        marker = "**" if result["pvalues"][t] < 0.05 else "  "
        print(
            f"  {t}: beta={result['params'][t]:+.3f}  p={result['pvalues'][t]:.3g}  {marker}"
        )
    print(
        f"  n_obs={result['n_obs']}, n_groups={result['n_groups']}, converged={result['converged']}"
    )


if __name__ == "__main__":
    main()
