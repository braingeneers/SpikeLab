# RT-Sort attribution

This subpackage (`spikelab.spike_sorting.rt_sort`) contains code
vendored from the RT-Sort algorithm. RT-Sort is an action-potential
propagation-based algorithm for real-time spike detection and sorting,
developed at UCSB and published in:

> van der Molen, T., Lim, M., Bartram, J., Cheng, Z., Robbins, A.,
> Parks, D. F., ... & Kosik, K. S. (2024).
> *RT-Sort: An action potential propagation-based algorithm for real
> time spike detection and sorting with millisecond latencies.*
> PLoS ONE, 19(12), e0312438.
> <https://doi.org/10.1371/journal.pone.0312438>

## Source

The vendored code is derived from the RT-Sort repository at
<https://github.com/KosikLabUCSB/RT-Sort> (MIT License,
Copyright (c) 2024 Tjitse van der Molen, Max Lim) and from the
user-friendly refactor subsequently contributed to the braindance
repository by Max Lim. Both versions implement the same algorithm
described in the paper above; no algorithmic changes were introduced
during the refactor.

## Files in this subpackage

| File | Origin |
|---|---|
| `_algorithm.py` | Forked from `braindance/core/spikesorter/rt_sort.py` |
| `model.py` | Forked from `braindance/core/spikedetector/model.py` |
| `detection_models/mea/` | Pretrained MEA weights (bundled verbatim) |
| `detection_models/neuropixels/` | Pretrained Neuropixels weights (bundled verbatim) |
| `__init__.py` | Written fresh for SpikeLab (lazy public API) |

## Changes from upstream

The adaptations made during porting are documented in the module
docstrings of `_algorithm.py` and `model.py`. In summary:

- Optional runtime dependencies (`torch`, `pynvml`, `threadpoolctl`,
  `diptest`) that were hard imports in braindance are now
  soft/lazy imports.
- The closed-loop streaming entry point (`rt_sort_maxwell_env_process`)
  and its supporting hardware code were removed — SpikeLab uses
  RT-Sort for offline sorting only.
- The default pretrained model paths now point at weights bundled
  under `detection_models/` inside this subpackage rather than at a
  separate braindance install.
- Training and plotting helpers that previously lived in
  `braindance.core.spikedetector.utils` and `plot` were replaced with
  minimal local stand-ins. SpikeLab only exercises the inference
  path; training methods on `ModelSpikeSorter` will no-op or raise if
  called.

## License

The vendored code is redistributed under the MIT License, matching
both the upstream RT-Sort repository and SpikeLab's own license. See
the top-level `LICENSE` file in this repository for the full text.

If you use RT-Sort via SpikeLab in published work, please cite the
paper above.
