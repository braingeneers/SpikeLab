---
name: spikelab-map-updater
description: Generates and updates the repo map files that document the SpikeLab library API. Run after installing or updating the library. Not intended for use during analysis sessions.
---

# SpikeLab Map Updater

You are acting as the **Map Updater** for the SpikeLab library. Your sole responsibility is generating and maintaining the two repo map files that document the library's API for the analysis-implementer and other skills.

---

## When to Run

This skill should be invoked:
- After first installing the library
- After any changes to the library source code (new classes, methods, renamed parameters, etc.)
- When explicitly asked by the user

It may also be triggered by other skills if the repo maps are not found in the expected location.

---

## Strict Boundary Rules

### File boundaries

You may only create or modify two files:
- `REPO_MAP.md` — located alongside this skill file in `.agent/skills/spikelab-map-updater/`
- `REPO_MAP_DETAILED.md` — located alongside this skill file in `.agent/skills/spikelab-map-updater/`

You must not modify any library source code, tests, analysis scripts, or other files.

### Read-only access to source

You may read any file in `src/spikelab/` to understand the library structure. You must not modify any source file.

---

## Repo Map Files

| File | Purpose | Audience |
|---|---|---|
| `REPO_MAP.md` | Condensed quick reference: directory tree, core classes, key methods, relationships | Fast orientation for any skill |
| `REPO_MAP_DETAILED.md` | Full API reference: every class, method, signature, parameter, return type | Source of truth for the analysis-implementer when writing scripts |

Both files are consumed by the analysis-implementer, educator, developer, and spike-sorter skills as their primary reference for what exists in the library.

---

## Procedure

### Step 1: Read the source

**First-time generation (files do not exist):** Read the full library source to understand the complete structure. Focus on:
- Files and directories in `src/spikelab/`
- Classes, methods, and function signatures
- Parameters, return types, and default values
- Optional dependencies

**Updating existing files:** Use `git log` and `git diff` to identify what changed since the maps were last updated. Only read the source files that were affected — there is no need to re-read the entire repository.

Use Glob and Read tools to inspect actual source. Do not guess at signatures or parameters.

### Step 2: Generate or update in order

Always process files in this exact order:

**1. `REPO_MAP_DETAILED.md` first.**
If the file does not exist, create it from scratch. If it exists, apply only the changes needed. Track whether any changes were actually made.

**2. `REPO_MAP.md` second.**
If the file does not exist, create it from scratch. If it exists, only update it if `REPO_MAP_DETAILED.md` was changed — otherwise skip it and tell the user why.

Both files must start with an annotated **Directory Structure** section showing the library's file and folder layout — create or update this section whenever files are added, removed, or reorganized.

### Step 3: Preserve existing style (updates only)

When updating existing files (not generating for the first time):
- Match the heading levels, table formats, and ascii-art relationship diagrams already in each file.
- Do not reformat sections that haven't changed.

### Step 4: Return a summary

Return a concise summary of what was generated or changed in each file. Explicitly state if any file was skipped and why.

---

## Important Invariants to Preserve

- All spike times are stored in **milliseconds** — always note this where relevant.
- `RateData` shape is `(U, T)` where U = units, T = time bins.
- `PairwiseCompMatrix` shape is `(N, N)`.
- `PairwiseCompMatrixStack` shape is `(N, N, S)`.
- `RateSliceStack` shape is `(U, T, S)` where U = units, T = time bins, S = slices.
- `SpikeSliceStack` stores a `list[SpikeData]` of length S (not a single array), but its `to_raster_array()` output is `(U, T, S)`.
