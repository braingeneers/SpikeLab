# RT-Sort backend — testing handover

This document explains how to test the new RT-Sort backend on a machine
with PyTorch and a CUDA GPU. The integration is structurally complete
and passes all import/config smoke tests on a torch-less machine, but
end-to-end sorting needs validation on real MEA/Neuropixels data.

---

## Branch

```
SpikeLab/feature/rt-sort-backend
```

Not pushed yet. To transfer:

```bash
# On the dev machine where the branch exists
cd SpikeLab
git push -u origin feature/rt-sort-backend

# On the test machine
git fetch origin
git checkout feature/rt-sort-backend
pip install -e .
```

---

## What got added

A new `rt_sort` SorterBackend that plugs into the existing
`sort_recording()` / `sort_multistream()` pipeline, identical in
usage to Kilosort2/4:

```
src/spikelab/spike_sorting/
├── rt_sort/                              # forked from braindance, MIT-attributed
│   ├── __init__.py                       # lazy public API
│   ├── NOTICE.md                         # attribution to van der Molen, Lim et al. 2024
│   ├── _algorithm.py                     # detect_sequences, RTSort, clustering, merging
│   ├── model.py                          # ModelSpikeSorter (DL detection network)
│   └── detection_models/                 # pretrained weights (bundled, ~740 KB)
│       ├── mea/{init_dict.json, state_dict.pt}
│       └── neuropixels/{init_dict.json, state_dict.pt}
├── backends/rt_sort.py                   # RTSortBackend(SorterBackend)
└── rt_sort_runner.py                     # detect_sequences + sort_offline wrapper
```

Modified files:
- `config.py` — added `RTSortConfig`, `RT_SORT_MEA` and `RT_SORT_NEUROPIXELS` presets, flat-map entries
- `_globals.py` — added `RT_SORT_*` globals
- `backends/__init__.py` — registered `"rt_sort"`

---

## Dependencies

Required on the test machine (on top of the existing SpikeLab deps):

| Package | Purpose | Install |
|---|---|---|
| `torch` | DL model inference (needs CUDA) | `pip install torch --index-url https://download.pytorch.org/whl/cu121` (match your CUDA version — see https://pytorch.org/get-started/locally/) |
| `diptest` | Hartigan dip test for amplitude-based cluster splitting | `pip install diptest` |
| `h5py` | Maxwell MEA trace I/O | `pip install h5py` |
| `scikit-learn` | `GaussianMixture` for latency-based cluster splitting | `pip install scikit-learn` |
| `tqdm` | Progress bars | `pip install tqdm` |
| `threadpoolctl` (optional) | BLAS thread control during parallel GMM fits | `pip install threadpoolctl` |
| `pynvml` (optional) | GPU memory reporting | `pip install nvidia-ml-py` |

`RTSortBackend.__init__` checks these upfront and raises a clear
`ImportError` listing any that are missing, so you don't have to
guess.

The bundled model weights at `rt_sort/detection_models/{mea,neuropixels}/`
are already in the repo — no separate download needed.

---

## Smoke test to run first

On the test machine, before touching any real data, verify the import
chain works end-to-end with torch present:

```python
import sys
sys.path.insert(0, "SpikeLab/src")  # or use `pip install -e SpikeLab/`

# 1. Basic imports
from spikelab.spike_sorting.backends import list_sorters, get_backend_class
from spikelab.spike_sorting.config import RT_SORT_MEA, RT_SORT_NEUROPIXELS
assert "rt_sort" in list_sorters()

# 2. Backend instantiation (should NOT raise now that torch is installed)
backend = get_backend_class("rt_sort")(RT_SORT_MEA)
print("Backend ready:", backend)

# 3. Load the bundled pretrained MEA model
from spikelab.spike_sorting.rt_sort.model import ModelSpikeSorter
model = ModelSpikeSorter.load_mea()
print("MEA model:", type(model).__name__)
print("  num_channels_in:", model.num_channels_in)
print("  sample_size:", model.sample_size)

# 4. Confirm the algorithm module imports
from spikelab.spike_sorting.rt_sort import detect_sequences, RTSort
print("detect_sequences:", callable(detect_sequences))
print("RTSort:", RTSort.__name__)
```

Expected output: all prints succeed without exception. If the backend
instantiation step raises, the ImportError message will list exactly
which packages are missing.

---

## End-to-end sort on a real recording

Use the same API as Kilosort2/4 — just swap in the `RT_SORT_MEA` (or
`RT_SORT_NEUROPIXELS`) preset:

```python
from spikelab.spike_sorting import sort_recording
from spikelab.spike_sorting.config import RT_SORT_MEA

spike_data_list = sort_recording(
    recording_files=["/path/to/recording.raw.h5"],
    config=RT_SORT_MEA,
    intermediate_folders=["/path/to/inter"],
    results_folders=["/path/to/results"],
    # Flat kwargs for common overrides:
    rt_sort_device="cuda",                          # or "cpu" for a slow test
    rt_sort_recording_window_ms=(0, 5 * 60 * 1000), # optional: first 5 min only
    rt_sort_verbose=True,
    # Curation is shared with Kilosort backends:
    fr_min=0.1,
    snr_min=5.0,
)

sd = spike_data_list[0]
print(f"Got {sd.N} curated units")
print(f"Recording length: {sd.length} ms")
```

The trained `RTSort` object is persisted automatically to
`intermediate_folders[0]/<rec_name>/sorter_output/rt_sort.pickle`
(control with `rt_sort_save_pickle=False` to disable). Phase 2 of this
project (stim-aware sorting) will reload that file to re-use the
detected sequences against a stimulation recording.

### Available `sort_recording` kwargs for RT-Sort

All the existing Kilosort-shared kwargs work (`fr_min`, `snr_min`,
`isi_viol_max`, `curate_first`, `n_jobs`, etc.), plus:

| Flat kwarg | Maps to | Default | Purpose |
|---|---|---|---|
| `rt_sort_model_path` | `rt_sort.model_path` | `None` (bundled) | Explicit model folder |
| `rt_sort_probe` | `rt_sort.probe` | `"mea"` | `"mea"` or `"neuropixels"` for bundled model selection |
| `rt_sort_device` | `rt_sort.device` | `"cuda"` | PyTorch device |
| `rt_sort_num_processes` | `rt_sort.num_processes` | `None` (auto) | Parallel worker count |
| `rt_sort_recording_window_ms` | `rt_sort.recording_window_ms` | `None` (full) | `(start_ms, end_ms)` window |
| `rt_sort_save_pickle` | `rt_sort.save_rt_sort_pickle` | `True` | Persist `RTSort` object for Phase 2 |
| `rt_sort_delete_inter` | `rt_sort.delete_inter` | `False` | Clean up intermediate cache after sort |
| `rt_sort_verbose` | `rt_sort.verbose` | `True` | Progress messages |
| `rt_sort_params` | `rt_sort.params` | `None` | Override dict merged into `detect_sequences` kwargs |

For fine-grained RT-Sort algorithm tuning (detection thresholds,
radii, merge params), pass `rt_sort_params={"stringent_thresh": 0.2,
"inner_radius": 60, ...}` — keys must match
`detect_sequences` parameter names in `rt_sort/_algorithm.py`.

---

## Things to verify during testing

**Structural / integration correctness:**

1. `detect_sequences` runs to completion on a known-good recording
   and returns a populated `RTSort` object.
2. `RTSort.sort_offline(return_spikeinterface_sorter=True)` returns a
   `NumpySorting` with a nonzero number of units and spikes per unit.
3. Waveform extraction runs on RT-Sort's `NumpySorting` — this path
   is where the existing pipeline would be most likely to break,
   since it was only tested on Kilosort sortings previously. Watch
   for channel-index or unit-id mismatches.
4. `SpikeData` conversion succeeds, curation runs, compilation
   writes the expected `.npz` / `.h5` outputs.
5. The output `SpikeData` is structurally identical to what you'd
   get from `sort_recording(..., config=KILOSORT4)`: has
   `neuron_attributes` populated, `metadata`, `raw_data`, etc.
6. Compare unit counts and rough spike train shapes against a
   Kilosort4 sort of the same recording — they won't match exactly
   (different algorithms), but should be in the same ballpark.

**Resource-use:**

7. Confirm GPU utilization during `detect_sequences` (should be
   heavy during the DL inference stage).
8. Check disk usage in the intermediate folder — `detect_sequences`
   writes `scaled_traces.npy`, `model_outputs.npy`, `model_traces.npy`
   which can be tens of GB for long recordings.
9. Confirm `rt_sort.pickle` is written to the sorter output folder
   and is loadable:
   ```python
   import pickle
   with open("<inter>/<rec_name>/sorter_output/rt_sort.pickle", "rb") as f:
       rt_sort = pickle.load(f)
   print(rt_sort.num_seqs, "sequences")
   ```

**Reproducibility:**

10. Re-running `sort_recording` on the same recording + config
    should use the cached `rt_sort.pickle` and `sorting.npz` and
    skip the DL inference stage. Control with
    `recompute_sorting=True` to force a full re-run.

---

## Known points of fragility

Because `_algorithm.py` was forked from braindance wholesale (~2990
lines), there are a few things to watch out for:

1. **Trace caching paths.** `detect_sequences` expects to be able to
   write `scaled_traces.npy` etc. to `inter_path`. On some filesystems
   (network mounts, read-only caches) this can fail silently or cause
   weird re-runs. Use a local fast disk for `intermediate_folders`.

2. **`h5py` Maxwell-specific fast paths.** The braindance
   `save_traces_mea` path uses direct h5py reads against the Maxwell
   HDF5 structure for speed. If your test recording has a nonstandard
   layout, this path may fail and fall back to the slower
   SpikeInterface path. Watch the logs for "save_traces_si" vs
   "save_traces_mea".

3. **Multiprocessing on Windows.** The braindance parallel clustering
   uses `multiprocessing.Pool` which behaves differently on Windows
   (spawn vs fork). If you see pickling errors during
   `form_all_clusters`, set `rt_sort_num_processes=1` to fall back to
   sequential processing.

4. **Device mismatch.** If `rt_sort_device="cuda"` but no GPU is
   available, torch will raise a cryptic error deep inside the model.
   Pre-check with `torch.cuda.is_available()` and fall back to `"cpu"`
   if needed (slow but works).

5. **`threadpoolctl` optional.** If not installed, the GMM fits will
   run unconstrained and may oversubscribe cores. Not a correctness
   issue, just a performance one. `pip install threadpoolctl` to fix.

6. **Neuropixels params.** The `RT_SORT_NEUROPIXELS` preset hard-codes
   the threshold/merge params from `neuropixels_params` in the
   braindance source. Verify these match what you expect for your
   Neuropixels data; override via `rt_sort_params={...}` if not.

---

## What's NOT in Phase 1 (deferred to Phase 2)

These are explicitly out of scope for the current branch and will be a
separate session:

- **Stimulation artifact removal.** Needs clean-room reimplementation
  because the braindance `artifact_removal.py` is GPL-3 and not part
  of the RT-Sort algorithm proper. Phase 2 will add a `stim_sorting/`
  submodule with template subtraction and polynomial subtraction
  methods implemented fresh from standard DSP.
- **Stim time recentering.** Finding actual stim artifact peaks near
  logged stim times. Simple argmax-in-window algorithm, Phase 2.
- **`sort_stim_recording()` workflow.** The end-to-end pipeline that
  takes a trained `RTSort` object (from Phase 1) + a stim recording
  and produces a `SpikeSliceStack` aligned to corrected stim times.
- **Real-time / streaming `running_sort` path.** Not ported. Can be
  added later if needed for online sorting.

---

## If something breaks

1. **Import errors**: re-run the smoke test from this document. The
   dependency check in `RTSortBackend.__init__` will name the missing
   package.
2. **CUDA OOM**: reduce `rt_sort_recording_window_ms` to a shorter
   window (e.g. 60 seconds) or use `rt_sort_device="cpu"` for
   debugging.
3. **Output folder permissions**: make sure
   `intermediate_folders[0]` is writable and has enough disk space
   (estimate: 2–3× the raw recording size for the intermediate
   caches).
4. **Unexpected failure deep inside `_algorithm.py`**: the forked
   code is intentionally a minimal adaptation of the braindance
   version. If a bug is reproducible on the braindance version too,
   it's a pre-existing issue, not something the port introduced. Run
   the same recording through braindance's original `detect_sequences`
   (in a separate Python env with braindance installed) to confirm.
5. **Waveform extraction crashes**: this is the most likely
   integration-specific failure point since SpikeLab's custom
   `WaveformExtractor` has only been tested against Kilosort outputs.
   If it crashes with a channel-index or unit-id error, the
   workaround is to compare the `NumpySorting` returned by
   `rt_sort_runner.spike_sort()` against the equivalent `Kilosort4`
   `BaseSorting` and look for shape/dtype/unit-id differences.

---

## Files for a reviewer to look at first

In priority order:

1. `backends/rt_sort.py` — the integration surface, ~230 lines, all
   new code
2. `rt_sort_runner.py` — the DL sort wrapper, ~200 lines, all new
   code
3. `config.py` — scroll to `RTSortConfig` and the `RT_SORT_*` presets
4. `rt_sort/NOTICE.md` — attribution
5. `rt_sort/__init__.py` — public API, lazy loading
6. `rt_sort/_algorithm.py` lines 1–90 — the adaptation header plus
   changed imports (the rest is forked verbatim, ~2900 lines of
   algorithm, out of scope for close review)
7. `rt_sort/model.py` lines 1–95 — the adaptation header plus
   changed imports

---

## Questions that might come up

**Q: Why is `_algorithm.py` a single 3000-line file instead of split
into detection / clustering / merging / sorter modules?**
A: The braindance source has hundreds of internal cross-references
between these pieces. A blind split would break them. A future
refactor can split it cleanly once the integration is validated
working; forking it whole was the lower-risk starting point.

**Q: Why isn't `h5py` a hard SpikeLab dep even though RT-Sort needs
it?**
A: SpikeLab already treats `h5py` as an optional dep (the SpikeData
loaders/exporters have the same pattern). Only the RT-Sort backend
pulls it in, and only when you instantiate the backend — not at
package import time.

**Q: Where do the model weights come from?**
A: Copied verbatim from `braindance/core/spikedetector/detection_models/`
(and ultimately from the upstream KosikLabUCSB/RT-Sort training runs).
Total size is 742 KB so bundling in the package is fine. See
`NOTICE.md` for attribution.

**Q: Can I use my own trained model?**
A: Yes. Pass `rt_sort_model_path="/path/to/folder/with/init_dict.json/and/state_dict.pt"`
to `sort_recording`, and it'll override the bundled weights.
