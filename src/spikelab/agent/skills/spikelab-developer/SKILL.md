---
name: spikelab-developer
description: Integrates analysis code into the SpikeLab library. Takes user-provided scripts, identifies which computations already exist in the library, rewrites the script to use existing methods, integrates novel computations as new library methods, writes tests, exposes through MCP, and submits a PR. Use when the user wants to promote analysis code into the library.
---

# SpikeLab Developer

You are acting as the **Developer** for the SpikeLab library. Your responsibility is to integrate analysis code from user-provided scripts into the library as reusable methods, write tests for them, expose them through MCP where appropriate, and submit the result as a pull request.

---

## File Boundaries

You are authorized to create or edit files in:
- `SpikeLab/src/spikelab/` — library source code (all subpackages)
- `SpikeLab/tests/` — test suite

You must **never** modify:
- Repo map files (`REPO_MAP.md`, `REPO_MAP_DETAILED.md`, `REPO_MAP_TESTS.md`) — updates are handled separately
- Files outside `SpikeLab/`

**This skill assumes an editable install from a source clone (`pip install -e SpikeLab/`).** PyPI-only installs are read-only and cannot be modified or PR'd — if the user is on a PyPI install, ask them to clone the repository first.

---

## Before Starting

### Step 1: Read the repo maps
Read `REPO_MAP.md` for a broad overview and `REPO_MAP_DETAILED.md` for the full API reference. Both are located in `<clone>/SpikeLab/src/spikelab/agent/skills/spikelab-map-updater/`. You need to know what already exists before assessing what is new. If the repo maps are not present, run the `spikelab-map-updater` skill to generate them before proceeding.

### Step 2: Read the source script
Read the user-provided analysis script in full. Do not assume you know what it does. Understand every computation, its inputs, outputs, and how it relates to the library's data classes.

---

## Integration Pipeline

The integration follows a fixed sequence of phases. Complete each phase before moving to the next. Present a plan to the user at the start and confirm before proceeding.

### Phase 1: Audit — what already exists?

Compare every computation in the script against the library's existing methods:

1. **Already integrated**: computations that directly call library methods (e.g., `sd.get_bursts()`, `rss.rank_order_correlation()`). No action needed.
2. **Reimplemented**: computations that replicate logic already available in the library but were written manually in the script (e.g., a manual burst participation loop when `sd.get_frac_active()` exists). These should be replaced.
3. **Novel**: computations that have no equivalent in the library and are general enough to be reusable. These are candidates for integration.

Present this audit to the user as a table:

| Script location | What it does | Library status | Action |
|---|---|---|---|
| lines 50-65 | ISI CV with filtering | Reimplemented (`interspike_intervals()` exists) | Replace |
| lines 80-120 | Burst rank-order z-score | Novel | Integrate |

Wait for user confirmation before proceeding.

### Phase 2: Rewrite — replace reimplementations

For each "reimplemented" computation:
- Replace the manual code with the corresponding library method call
- Verify that the replacement produces identical results (same output shapes, values, and types)
- Do **not** change the script's analysis results — only its implementation

### Phase 3: Integrate — add novel methods to the library

For each "novel" computation confirmed by the user:

**Identify the right home:**
- Which data class does the method belong to? (SpikeData, RateData, PairwiseCompMatrix, SpikeSliceStack, RateSliceStack, etc.)
- Does a closely related method already exist that should be extended instead?
- Could the logic reuse existing methods for intermediate steps? (e.g., use `sparse_raster()` internally, call `get_bursts()` for burst edges)
- Is the logic better placed as a standalone function in `utils.py`?

**Adapt the code to library conventions:**
- Match the existing code style, docstring format, and parameter naming conventions
- Replace script-level data loading with proper method parameters
- Remove `sys.path` hacks, print statements, and script-level boilerplate
- Guard any new optional dependencies with `try/except ImportError`
- All spike times must remain in **milliseconds**
- Reuse existing library methods for intermediate computations wherever possible
- Add graceful error handling: validate inputs at method boundaries, raise clear `ValueError` or `TypeError` messages that tell the caller what went wrong and what was expected, and handle degenerate cases (e.g., empty arrays, zero units) by returning sensible defaults (empty arrays, NaN) rather than crashing

**Docstring format:**
```
One sentence describing what the method does.

Parameters:
    param_name (type): Description.

Returns:
    name (type): Description.

Notes:
    - Only include when something non-obvious needs documenting.
```

**Code quality:**
- Format all code with **black**
- Consider parallelisation for loops over units, pairs, or slices (see existing patterns: Numba for pure numerical loops, ThreadPoolExecutor for numpy/scipy calls)
- Never add hard module-level imports for optional dependencies

### Phase 4: Test — write tests for new methods

Write tests for every method added in Phase 3. Follow these rules:

**Organization:**
- Add tests to the existing test file for that module (e.g., `test_spikedata.py` for SpikeData methods)
- One test class per method or closely related group: `Test{ClassName}{MethodOrFeature}`
- Do not create catch-all classes like `TestEdgeCases` or `TestRecentFixes`

**Framework:**
- All tests use **pytest** — plain test classes, bare `assert` statements
- Follow existing patterns for helpers, fixtures, and assertion methods in the file

**Coverage requirements:**
- **Main usage**: test the primary use case with realistic inputs and verify outputs (shapes, dtypes, values)
- **Edge cases**: empty arrays, single-element inputs, NaN values, zero-length recordings, out-of-bounds indices
- **Boundary conditions**: minimum valid inputs, maximum expected sizes, threshold boundaries
- **Error paths**: verify that invalid inputs raise appropriate exceptions

**Docstring format for tests:**
```python
def test_something(self):
    """
    One sentence overview.

    Tests:
        (Test Case 1) Description.
        (Test Case 2) Description.
    """
```

**Scientific computing focus:**
- Verify mathematical invariants: spike times monotonic, array shapes match conventions (`(U, T)`, `(N, N, S)`), numpy dtypes consistent
- For methods on multi-dimensional arrays, test across all relevant axes

**Run tests** after writing them to confirm they pass:
```bash
conda run -n spikelab python -m pytest SpikeLab/tests/<test_file>.py::<TestClass> -v
```

### Phase 5: MCP — expose through MCP server (if applicable)

Not all methods need MCP exposure. Expose a method if it:
- Produces a result that would be stored in the workspace
- Would be useful as a standalone tool call (not just an internal helper)
- Follows the input/output pattern of existing MCP tools

If MCP exposure is appropriate:

**Write the async wrapper** in `src/spikelab/mcp_server/tools/analysis.py`:
```python
async def new_method(
    workspace_id: str,
    namespace: str,
    key: str,
    ...
) -> Dict[str, Any]:
    ws = _get_workspace(workspace_id)
    sd = _get_spikedata(ws, namespace)
    result = sd.new_method(...)
    ws.store(namespace, key, result)
    return {"workspace_id": workspace_id, "namespace": namespace, "key": key}
```

**Register the tool** in `src/spikelab/mcp_server/server.py`:
- Add `inputSchema` with proper types and descriptions
- Add dispatch entry mapping tool name to wrapper function
- Use `_WS_PROPS` for shared workspace/namespace properties

### Phase 6: PR — submit a pull request

Once all phases are complete:

1. Create a new branch (never commit to `main`)
2. Stage and commit all changes with a descriptive message
3. Push to remote
4. Create a PR using `gh pr create` with:
   - A concise title
   - Body listing: methods added, tests written, MCP tools registered
   - Reference to the source script that motivated the integration
5. Request review from `TjitsevdM` using `gh pr edit --add-reviewer TjitsevdM`

---

## Checklist After Each Integration

Before moving to the PR phase, verify:

- [ ] All reimplementations in the source script replaced with library calls
- [ ] New methods have complete docstrings
- [ ] `__init__.py` exports updated if the new method should be importable at package level
- [ ] No hard imports for optional dependencies
- [ ] All existing tests still pass (run the full test suite)
- [ ] New tests cover main usage, edge cases, and error paths
- [ ] New tests pass
- [ ] Black formatting passes on all modified files
- [ ] MCP tools registered and documented (if applicable)

---

## Key Conventions

- All spike times in **milliseconds**
- Data structure shapes: `RateData (U, T)`, `PairwiseCompMatrix (N, N)`, `PairwiseCompMatrixStack (N, N, S)`, `RateSliceStack (U, T, S)`, `SpikeSliceStack` stores `list[SpikeData]` of length S
- Optional dependencies must remain optional
- All code formatted with **black**
- Prefer reusing existing library methods over reimplementing logic
