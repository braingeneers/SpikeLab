# HIPPIE Integration — Codebase Changes

## Overview

HIPPIE is added as an **optional dependency** of SpikeLab.  
Users who want cell-type classification install it with:

```bash
pip install "spikelab[hippie]"
```

Nothing in the base SpikeLab install changes — all new code is either in new files or behind lazy imports that only run when HIPPIE is actually present.

---

## Repository map

```
HIPPIE/
└── hippie/
    ├── __init__.py          ← MODIFIED  (+2 lines)
    ├── inference.py         ← NEW
    └── checkpoint.py        ← NEW

SpikeLab/
├── pyproject.toml           ← MODIFIED  (+7 lines)
├── src/spikelab/
│   ├── spikedata/
│   │   └── hippie_adapter.py            ← NEW
│   └── mcp_server/
│       ├── tools/
│       │   └── analysis.py              ← MODIFIED  (+90 lines at EOF)
│       └── server.py                    ← MODIFIED  (+60 lines: schema + dispatch)
├── tests/
│   └── test_hippie_adapter.py           ← NEW
└── docs/source/guides/
    ├── index.rst                        ← MODIFIED  (+1 line)
    └── hippie_classification.rst        ← NEW
```

---

## File-by-file changes

### `HIPPIE/hippie/__init__.py` — modified

**What changed:** Added one export line.

```python
# Before (4 lines)
from .multimodal_model import MultiModalCVAE, ...
from .dataloading import ...
from .augmentations import ...
from .backbones import ...

# After (+1 line)
from .inference import HIPPIEClassifier, TECHNOLOGY_IDS
```

**Why:** Allows `from hippie import HIPPIEClassifier` — the clean public API.

---

### `HIPPIE/hippie/checkpoint.py` — new file

**Purpose:** Load a pretrained checkpoint and return a ready `MultiModalCVAE`.

```
infer_model_dims(state_dict)
  → reads source_embed.weight / class_embed.weight shapes
  → returns (num_sources, num_classes)

build_model(ckpt_path)
  → torch.load(..., weights_only=False)
  → strips Lightning "model." prefix from state_dict keys
  → calls infer_model_dims to auto-detect architecture
  → instantiates MultiModalCVAE with hardcoded inference config:
        modalities = {"wave": 50, "isi": 100, "acg": 100}
        z_dim      = 30
        config     = ExperimentConfigs.class_decoder_source_bn_aug_reg()
        backbone_base_width = 64
  → model.load_state_dict(sd, strict=False)
  → returns model in eval() mode on CPU
```

**Key detail:** `strict=False` on `load_state_dict` — the checkpoint may contain
decoder weights that are not needed for inference; this suppresses the error.

---

### `HIPPIE/hippie/inference.py` — new file

**Purpose:** High-level inference API used by both direct callers and the SpikeLab adapter.

```
TECHNOLOGY_IDS: dict
  {"neuropixels": 0, "silicon_probe": 1, "juxtacellular": 2, "tetrodes": 3}

class HIPPIEClassifier:

  from_pretrained(repo_id, filename, device, cache_dir)
    → hf_hub_download(repo_id, filename)
    → build_model(ckpt_path)
    → returns HIPPIEClassifier

  from_checkpoint(checkpoint_path, device)
    → build_model(str(path))
    → returns HIPPIEClassifier

  get_embeddings(wave, isi, acg, tech_id, batch_size) → np.ndarray (N, 30)
    → accepts tech_id as int or name string
    → batches input; calls model.encode({wave, isi, acg}, source_labels=tech_id)
    → returns z_mean concatenated across batches

  umap_reduce(embeddings, n_components, n_neighbors, min_dist, metric, random_state)
    → static method; lazy-imports umap.UMAP
    → returns (N, n_components) float32 coords

  hdbscan_cluster(embeddings, min_cluster_size, min_samples, metric)
    → static method; lazy-imports hdbscan.HDBSCAN
    → returns (N,) int32 labels; -1 = noise
```

**Input contract for `get_embeddings`:**

| Modality | Shape   | Normalization                            |
|----------|---------|------------------------------------------|
| `wave`   | (N, 50) | min-max → [-1, 1]                        |
| `isi`    | (N, 100)| log(x+1), then min-max → [-1, 1]        |
| `acg`    | (N, 100)| min-max → [-1, 1]                        |

---

### `SpikeLab/pyproject.toml` — modified

**What changed:** Added the `hippie` optional extra.

```toml
[project.optional-dependencies]
# ... existing extras unchanged ...
hippie = [
  "hippie @ git+https://github.com/braingeneers/HIPPIE.git",
  "huggingface-hub>=0.20",
  "torch>=2.0",
  "umap-learn>=0.5.0",
  "hdbscan>=0.8",
]
```

**Note:** PyTorch with CUDA must still be installed separately if GPU inference is desired.

---

### `SpikeLab/src/spikelab/spikedata/hippie_adapter.py` — new file

**Purpose:** Bridge between `SpikeData` and the HIPPIE encoder.  
Not exported from `spikedata/__init__.py` — imported explicitly to avoid import errors for users without HIPPIE.

```
# Guards
_require_hippie()
  → tries `import hippie`; raises ImportError with install hint if missing

# Preprocessing helpers (pure numpy/torch, no HIPPIE dependency)
_preprocess_waveform(wave, target=50) → (50,) float32
  → F.interpolate to target length, min-max → [-1, 1]

_isi_histogram(spike_times, n_bins=100) → (100,) float32
  → np.diff(sorted spikes) * 1000 → ms
  → np.histogram with log-spaced bins [1 ms, 5000 ms]
  → log1p transform, then min-max → [-1, 1]
  → silent neurons (< 2 spikes) return flat -1 array

_autocorrelogram(spike_times, max_lag_ms=100, n_bins=100) → (100,) float32
  → forward-looking searchsorted loop (O(N) per lag window)
  → normalised to sum=1, then min-max → [-1, 1]
  → empty trains return zeros

# Public API
extract_features(sd, isi_bins, acg_bins, acg_max_lag_ms) → dict
  → reads avg_waveform from neuron_attributes (raises ValueError if missing)
  → stacks waveforms, ISI histograms, ACGs into (N, bins) arrays
  → returns {"wave": (N,50), "isi": (N,100), "acg": (N,100)}

classify_neurons(sd, repo_id, tech_id, device, run_umap, run_hdbscan,
                 umap_kwargs, hdbscan_kwargs, batch_size, cache_dir) → dict
  → calls _require_hippie() then imports HIPPIEClassifier inside function
  → calls extract_features(sd)
  → HIPPIEClassifier.from_pretrained(repo_id, device, cache_dir)
  → clf.get_embeddings(wave, isi, acg, tech_id, batch_size)
  → optionally: clf.umap_reduce(embeddings, **umap_kwargs)
  → optionally: clf.hdbscan_cluster(umap_coords or embeddings, **hdbscan_kwargs)
  → returns {"embeddings", "umap_coords"?, "cluster_labels"?}
```

---

### `SpikeLab/src/spikelab/mcp_server/tools/analysis.py` — modified

**What changed:** ~90 lines appended at end of file.

```
async def classify_neurons_hippie(
    workspace_id, namespace,
    tech_id=0, run_umap=True, run_hdbscan=True,
    min_cluster_size=5, umap_n_neighbors=30, umap_min_dist=0.1,
    device="cpu", cache_dir=None
) → dict

  Calls:
    _get_workspace(workspace_id)
    _get_spikedata(ws, namespace)
    classify_neurons(sd, ...)          ← imported lazily inside function
    sd.set_neuron_attribute(...)       ← writes hippie_embedding,
                                          hippie_umap_x/y, hippie_cluster
    ws.store(namespace, "spikedata", sd)

  Returns JSON summary:
    n_neurons, embedding_dim,
    umap_computed, hdbscan_computed,
    n_clusters, n_noise_neurons,
    neuron_attributes_added
```

**Import note:** `from ....spikedata.hippie_adapter import classify_neurons` is
inside the function body — the MCP server starts fine without HIPPIE installed
and only fails at call time with a clear error message.

---

### `SpikeLab/src/spikelab/mcp_server/server.py` — modified

**Two edits:**

1. **Tool schema** — inserted before the "Workspace management tools" section:

```python
types.Tool(
    name="classify_neurons_hippie",
    description="Classify neurons using the pretrained HIPPIE multimodal model ...",
    inputSchema={
        "required": ["workspace_id", "namespace"],
        "properties": {
            workspace_id, namespace,
            tech_id (int, default 0),
            run_umap (bool, default True),
            run_hdbscan (bool, default True),
            min_cluster_size (int, default 5),
            umap_n_neighbors (int, default 30),
            umap_min_dist (float, default 0.1),
            device (str, default "cpu"),
            cache_dir (str, optional),
        }
    }
)
```

2. **Dispatch entry** — added to `_TOOL_DISPATCH`:

```python
"classify_neurons_hippie": analysis.classify_neurons_hippie,
```

---

### `SpikeLab/tests/test_hippie_adapter.py` — new file

**13 tests across 5 classes**, all skipped automatically if `hippie` is not installed (`pytest.importorskip`).

```
TestPreprocessWaveform   (3 tests)
  ✓ output shape is (50,)
  ✓ values in [-1, 1]
  ✓ flat waveform (all zeros) does not crash

TestISIHistogram          (3 tests)
  ✓ output shape is (100,)
  ✓ values in [-1, 1]
  ✓ silent neuron (1 spike) does not crash

TestAutocorrelogram       (3 tests)
  ✓ output shape is (100,)
  ✓ values in [-1, 1]
  ✓ empty spike train returns zeros

TestExtractFeatures       (3 tests)
  ✓ shapes (N, 50) / (N, 100) / (N, 100)
  ✓ dtype is float32
  ✓ missing avg_waveform raises ValueError

TestClassifyNeurons       (4 tests — fully mocked, no download)
  @patch("hippie.inference.HIPPIEClassifier")   ← correct patch target
  ✓ returns embeddings + umap_coords + cluster_labels by default
  ✓ run_umap=False, run_hdbscan=False omits those keys
  ✓ embedding shape is (N, 30)
  ✓ tech_id string is forwarded unchanged to get_embeddings
```

---

### `SpikeLab/docs/source/guides/hippie_classification.rst` — new file

Full Sphinx guide covering:
- Installation
- Quick start
- Return value table
- Technology ID table
- Advanced options (custom UMAP/HDBSCAN params, embeddings only, direct HIPPIE API)
- MCP server usage + example agent prompts
- How it works (feature extraction → encoding → UMAP → HDBSCAN)
- Checkpoint info

### `SpikeLab/docs/source/guides/index.rst` — modified

Added `hippie_classification` to the toctree (+1 line).

---

## Data flow

```
SpikeData
  │
  │  avg_waveform (neuron_attributes)
  │  spike trains (sd.train)
  ▼
hippie_adapter.extract_features()
  │
  ├── _preprocess_waveform()  → (N, 50)  min-max [-1,1]
  ├── _isi_histogram()        → (N, 100) log1p + min-max [-1,1]
  └── _autocorrelogram()      → (N, 100) normalized + min-max [-1,1]
  │
  ▼
HIPPIEClassifier.get_embeddings()
  │
  │  model.encode({wave, isi, acg}, source_labels=tech_id)
  │  ↳ ResNet18Enc × 3 → fusion_encoder → z_mean (30-D)
  │
  ▼
  (N, 30) embeddings
  │
  ├──[run_umap=True]──► umap_reduce()  → (N, 2) coords
  │                        cosine metric, n_neighbors=30
  │
  └──[run_hdbscan=True]► hdbscan_cluster() on umap_coords
                            → (N,) labels  (-1 = noise)
  │
  ▼
classify_neurons() returns
  {"embeddings": (N,30), "umap_coords": (N,2), "cluster_labels": (N,)}
  │
  ▼  [via MCP tool classify_neurons_hippie]
neuron_attributes:
  hippie_embedding  (N, 30)
  hippie_umap_x     (N,)
  hippie_umap_y     (N,)
  hippie_cluster    (N,)    -1 = noise
```

---

## Bug fixed during audit

| File | Issue | Fix |
|------|-------|-----|
| `tests/test_hippie_adapter.py` | `@patch("spikelab.spikedata.hippie_adapter.HIPPIEClassifier")` — wrong target; `HIPPIEClassifier` is imported *inside* `classify_neurons` via `from hippie.inference import HIPPIEClassifier`, so the mock must be placed on the source module | Changed to `@patch("hippie.inference.HIPPIEClassifier")` |

---

## What was NOT changed

- `spikedata/__init__.py` — `hippie_adapter` is intentionally **not** re-exported here; importing it would crash base installs without HIPPIE
- Any existing analysis, data loader, or exporter tool
- HIPPIE training code (`multimodal_model.py`, `dataloading.py`, `augmentations.py`, etc.)
- The HIPPIE `pyproject.toml`
