"""SORT-1 — sorter endpoint runs a real synthetic recording end-to-end.

**Skeleton — not implemented.** Requires:

- A GPU available on this host (PyTorch + CUDA)
- The ``[spike-sorting]`` and ``[kilosort4]`` extras installed in the
  analysis env
- A small synthetic recording the library knows how to sort. Likely
  source: extract a fixture-builder helper from
  ``tests/test_spike_sorting.py`` (293 KB; contains synthetic-data
  generators used by the library's own sort tests).

Run only manually with ``--confirm`` (analogous to INFRA-8) because:

- A real Kilosort run holds the GPU for several minutes, which will
  collide with any other GPU workload on this host.
- The first time it runs, the agent will go fetch / generate the
  fixture, which may need disk space and human review of what it
  actually picks.

Intended end-state shape:

1. Set up a temp working dir with a tiny ``.raw.h5`` (or analogous
   format) — call the fixture builder from ``test_spike_sorting.py``.
2. Send an ``ask_spikelab_sorter`` turn:
       "Sort the recording at <path>. Use Kilosort 4 with default
        curation thresholds. Save outputs under <out_dir>. Report the
        number of curated units when done."
3. Assert:
       - ``is_error`` is False
       - ``<out_dir>/sorted_spikedata_curated.pkl`` exists
       - The pickle loads as a ``SpikeData`` with ≥ 1 unit
       - Pickling that file then unpickling reproduces the same unit count
4. Mid-run, from a parallel coroutine, call
   ``get_spikelab_task_progress(task_id)`` and verify the response has
   ``source == "log_file"`` with a tail containing Kilosort iteration
   strings — this is the load-bearing demo of the progress tool reading
   real sorter logs.

This file is left as a skeleton until a sortable fixture is identified
and the GPU prereq is confirmed on the target host.
"""

from __future__ import annotations

import argparse
import sys


def main(confirm: bool) -> int:
    if not confirm:
        print(
            "SORT-1 SKIP: skeleton only. Requires GPU + Kilosort fixture; "
            "pass --confirm to attempt (will fail until implemented)."
        )
        return 0
    print(
        "SORT-1 NOT IMPLEMENTED. See module docstring for the intended shape. "
        "Implement the fixture-builder + sorter prompt + curated-pkl assertion "
        "+ parallel progress check before relying on this test."
    )
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    sys.exit(main(args.confirm))
