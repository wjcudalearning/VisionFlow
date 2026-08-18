# Repository Agent Instructions

These rules apply to all future Codex work in this repository.

## Project and environment

VisionFlow AOI is a recipe-driven OpenCV inspection system with a PySide6 GUI and an optional CUDA DLL backend.

Primary entry points:

- CLI: `python main.py --image <image> --recipe <recipe.yaml> --output <directory>`
- GUI: `python main.py --gui`
- Packaged GUI entry/smoke: `gui_launcher.py` and `VisionFlow AOI.exe --smoke-test`
- Windows package build: `build_exe.ps1` using the tracked `VisionFlow AOI.spec`
- Standalone utilities: `export_ng_tiles_by_area.py`, `export_pattern_grid_tiles.py`, `export_matrix_summary.py`, and `export_scatter_plots.py`
- Utility bundle build: `build_utility_tools.ps1`; individual utility builds use their dedicated `build_*_exporter.ps1` or `build_ng_tile_area_tool.ps1` entry point
- CUDA build: `gpu/build_cuda_dll.ps1`
- CUDA validation: `gpu/validate_cuda_dll.py`
- CUDA source/ABI preflight: `gpu/preflight_cuda_build.py`

Use the workspace virtual environment for every Python command:

```powershell
.\env\Scripts\python.exe
```

The normal development machine may not have `nvcc`, CMake, or an NVIDIA GPU. Never claim that CUDA source compiled or passed runtime validation unless those commands actually ran. Record outstanding RTX 3090 validation in `Todo.md`.

## Canonical roadmap

- `Todo.md` is the only project task list. Read it before implementation work.
- Do not create separate CPU, GPU, CUDA, GUI, release, or feature Todo files.
- Mark only work that is genuinely complete. Hardware-dependent tasks remain unchecked until tested on the target machine.
- After a completed change, update the relevant checkbox and append a dated entry under `完成紀錄`.
- Keep CPU correctness, GPU optimization, deployment, and acceptance criteria in the same roadmap.

## Module ownership

- Top-level entry points: keep CLI orchestration in `main.py`, packaged startup/smoke in `gui_launcher.py`, main packaging in `build_exe.ps1` and `VisionFlow AOI.spec`, and standalone utilities in `export_*.py` with their dedicated spec/build scripts.
- `core/`: pipeline, recipe loading/building, tiling, aggregation, reporting, profiling, batch/monitor processing, result schemas/compaction, GPU sessions/bridge, preprocessing plans and executors.
- `detectors/`: detector-specific feature extraction, geometry, filtering, and result metadata.
- `gpu/`: CUDA C ABI, kernels, persistent contexts, build scripts, native smoke tests, and CPU/GPU validation.
- `gui/`: PySide6 screens, widgets, workers, status, and preview behavior.
- `recipes/`: YAML configuration and production defaults.
- `tests/`: automated correctness, fallback, routing, and regression tests.
- `.github/workflows/`: CI only; keep GPU runtime jobs isolated from ordinary hosted runners.
- `cuda_practice/`: independent learning/device-check programs; do not make production runtime depend on them.
- `design_handoff_aoi_gui/`: design reference only; production UI behavior belongs in `gui/`.

Put behavior in the narrowest appropriate module. Do not duplicate pipeline or fallback policy inside individual detectors.

## CPU/GPU architecture contract

- CPU-only operation is a fully supported product mode and the correctness reference.
- Preserve `gpu.mode` semantics: `cpu` never requests/loads CUDA, `auto` may fall back, and `cuda` requires CUDA success and forbids hidden CPU fallback.
- Missing GPU, missing/old DLL, unsupported operator, CUDA initialization failure, kernel error, or OOM must not break CPU execution when fallback is enabled.
- A failed GPU step must restart the entire detector on CPU. Never combine partial GPU intermediate results with a CPU continuation.
- Preserve recipe semantics, PASS/NG, coordinates, defect metadata, output formats, and ordering. Define and test any allowed numerical tolerance.
- Do not create one CUDA workflow or exported function per detector.
- Detectors declare backend-neutral immutable `PreprocessPlan` objects using shared typed operators.
- `CpuPreprocessExecutor` defines OpenCV fallback semantics. `CudaPreprocessExecutor` selects a generic native plan, compatibility adapter, reusable primitives, or explicit fallback.
- Add a shared operator when an algorithm is reusable. Detector-named native adapters are compatibility code, not the extension model.
- Do not silently substitute a faster operation with different semantics, such as nearest-neighbor for OpenCV `INTER_AREA`.
- Prefer one upload, multiple device operators, and one necessary download. Reuse context buffers across operators, tiles, and images where lifetime permits.
- Preserve context-owned resident image/device ROI lifetime and generation checks. Batch and monitor share one `GpuExecutionSession`; do not create one runtime per image or per worker.
- Keep small contour/geometry work, YAML, aggregation, GUI control, CSV/JSON, PNG encoding, and disk I/O on CPU unless profiling proves otherwise.
- Tile-level CPU parallelism is opt-in. Use thread-local detector instances, preserve input ordering, and keep GPU detector or resident-image execution on the single serialized GPU path.
- Recipe caching must invalidate on file metadata changes and return independent deep copies; never expose a mutable cached recipe.
- Debug intermediate images are opt-in runtime payloads. Strip them from JSON and public tile results, and never enable them in production defaults.
- GPU default enablement requires RTX 3090 equivalence, stability, and end-to-end performance evidence.

## Compatibility and OOP rules

- Preserve the public ABI v1 primitive API unless an explicit versioned migration is planned.
- Add native capabilities through optional export probing so old DLLs retain legacy GPU or CPU fallback paths.
- Device pointers belong to native context objects; do not expose ownerless raw device pointers to Python.
- Keep runtime lifecycle explicit with `close()`/context manager behavior and safe cleanup.
- Keep shared runtime calls thread-safe. A single bounded GPU queue is preferred over competing workers.
- Avoid module globals that hold mutable detector, recipe, image, or GPU state.
- Inject runtime/backend dependencies where tests need CPU, fake DLL, legacy DLL, or failing GPU behavior.

## GUI interaction contract

- Preserve the existing visual language and status hierarchy: TopBar owns global progress and actual backend, operation panels own step detail, and the status bar contains only short events.
- Backend labels must come from runtime result metadata. Never infer CUDA active from a recipe request; expose CPU fallback reasons in text or a tooltip.
- Use inline notices for recoverable feedback. Reserve modal dialogs for blocked operations, destructive choices, unsaved-change confirmation, and close prevention while background work is active.
- Recipe Designer changes must participate in dirty tracking and shared `RecipeManager` validation. Loading programmatic values must not create false dirty state.
- Persist user working context through `GuiPreferences`/`QSettings`; ignore stale paths safely and keep tests isolated with injected temporary settings.
- Batch and monitor histories use Qt model/view and bounded incremental updates. Keep status filtering proxy-based and sample oversized scatter data deterministically.
- Replace large viewer overlay sets as one bounded update: suspend per-item viewport updates, invalidate the scene, and request a repaint after the batch. Preserve explicit repaint behavior when overlays are toggled or the viewer is resized; Windows must not retain stale strips or distorted content.
- Keep hidden Results content lazy. Defer table/output population until the screen is opened and create large thumbnail collections in bounded event-loop batches so inspection completion remains responsive.
- New operator-facing text is Traditional Chinese except established industrial abbreviations such as PASS, NG, ERROR, CPU, CUDA, ROI and DLL. Status must remain understandable without color alone and keyboard paths require tests.

## Detector parameter access contract

- Every registered Detector parameter must be classified by the shared `ParameterSpec.parameter_group` contract as `outer` or `inner`. `parameter_group` is authoritative; the derived `engineer_visible` field exists only for compatibility and detector source must not set it directly.
- `outer` is intentionally narrow: only physical acceptance geometry that engineering personnel can tune without image-processing knowledge, such as area, width/height, radius/length, spacing/gap, dimensional tolerance, crop padding, and ROI or mask extents/insets.
- Everything else is `inner`, including enable/mode switches, coordinates or origin selection, threshold/max-value/inversion, blur, adaptive parameters, morphology, contour mode, scale, circularity/fill/white-pixel ratios, model/class selection, confidence, NMS, backend, and precision.
- `ParameterSpec` must remain fail-closed: an omitted or newly introduced classification defaults to admin-only `inner`. Never infer access from a parameter name in the GUI and never make an unknown parameter engineer-visible automatically.
- Recipe Designer exposes only `outer` parameters in Engineer mode and exposes both groups in Admin mode. OP does not gain Detector editing access. Mode changes and programmatic loads must not create false dirty state.
- Loading or saving a Recipe in Engineer mode must preserve every hidden `inner` value exactly. Parameter grouping is UI/access metadata only and must not rename Recipe fields, change defaults, alter Detector decisions, or break legacy Recipe ID aliases.
- Before adding or changing a Detector, audit the registry ID, `default_params`, `PARAM_SPEC`, every runtime parameter read, every behavior-affecting hard-coded constant or formula, tracked Recipes, and GUI metadata together. A tuning value must be represented in the shared schema and used by runtime, unless it is deliberately fixed as part of the Detector identity and documented as non-configurable; auditing only existing `self.params` reads is insufficient.
- Regression tests must enumerate the exact `outer` key set and required optical/algorithm `inner` keys for every registered Detector, assert `default_params` and `PARAM_SPEC` key equality, verify Engineer/Admin visibility for all registered Detectors, prove Admin edits are saved back to Recipe, and prove an Engineer-mode Recipe round trip preserves hidden `inner` values.

## Future detector development contract

- Every new traditional CV detector must express reusable image preprocessing as a cached immutable `PreprocessPlan`; detector code keeps only detector-specific geometry, filtering, PASS/NG decisions, defect metadata, and deterministic ordering.
- Cache keys must cover the input shape/dtype and every detector parameter that changes preprocessing semantics. Use the bounded shared plan cache rather than mutable module globals or rebuilding plans for every tile.
- `CpuPreprocessExecutor` is the correctness reference. Optional CUDA execution must use shared typed operators, capability reporting, and full-detector CPU restart on unsupported semantics or failure.
- Do not add detector-specific CUDA workflows or exports for new detectors. When a reusable operation is missing, add a backend-neutral typed operator and its CPU reference first; temporary compatibility adapters require an explicit migration item in `Todo.md`.
- A new traditional CV detector is not complete without tests for direct OpenCV/CPU equivalence, plan cache reuse and invalidation, missing/legacy/failing backend routing, PASS/NG, defect count, bbox, area, confidence, metadata, and deterministic ordering as applicable.
- When detector behavior is defined by an external tuning or reference tool, preserve its exact operation order, border/channel semantics, masks, and parameter meaning. Add a direct pixel-level reference comparison that can distinguish the approved order from plausible but incorrect reorderings.
- DL model inference, framework sessions, and TensorRT/ONNX Runtime execution are not required to fit inside `PreprocessPlan`. Reusable traditional CV preprocessing and postprocessing around the model should still use shared typed operators or an equivalent shared DL preprocessing abstraction.
- DL detectors must share model/session lifecycle, GPU scheduling, VRAM budget, warm-up, capability metrics, error handling, and fallback policy. GUI, monitor, and batch workers must not each load an independent model copy.
- A DL detector must preserve traceable preprocessing, model version, backend, input/output shape, thresholds, and fallback metadata, with CPU or approved reference-backend accuracy tests before GPU acceleration becomes a default.

## Required workflow

Before editing:

1. Run `git status --short --branch`.
2. Read the relevant `Todo.md` sections and nearby implementation/tests.
3. Identify user-owned or unrelated working-tree changes and preserve them.

While editing:

1. Make focused changes with the existing module boundaries.
2. Add or update automated tests for behavior, CPU equivalence, old-DLL routing, and failure fallback.
3. Update `Todo.md` accurately; do not mark source-only CUDA work as hardware-validated.
4. Keep generated files under ignored validation/output directories.
5. Keep `README.md` user-facing and evidence-based; keep this file focused on contributor/agent invariants. Update both when commands, architecture, packaging, or validation policy changes.

Before finishing, always run:

```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -v
.\env\Scripts\python.exe -m compileall main.py gui_launcher.py export_ng_tiles_by_area.py export_pattern_grid_tiles.py export_matrix_summary.py export_scatter_plots.py core detectors gui gpu
.\env\Scripts\python.exe gpu\preflight_cuda_build.py
git diff --check
```

For pipeline, detector, recipe, tiling, GPU bridge, or reporter changes, also run a CLI smoke test with a synthetic image and write only to `outputs_validation/`.

For GUI changes, also run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\env\Scripts\python.exe -c "from pathlib import Path; from PySide6.QtWidgets import QApplication; from gui.main_window import MainWindow; app=QApplication([]); w=MainWindow(); w.recipe_panel.load_recipe(Path('recipes/PRODUCT_A_AOI_01.yaml')); print(w.windowTitle(), w.recipe_panel.detector_list.count())"
```

For packaging, `gui_launcher.py`, or spec changes, build through `build_exe.ps1` and run the packaged `--smoke-test` when the local environment can support a package build. The smoke must cover bundled recipe/MainWindow startup, CPU-only execution, missing-DLL fallback equivalence with zero GPU calls, and explicit strict-CUDA failure.

For standalone utility or utility spec/build changes, use the matching dedicated build script and run that utility's packaged `--smoke-test`. Keep utility bundle tags (`utility-tools-vX.Y.Z`) and the legacy NG Tile tool tag namespace separate from VisionFlow AOI application tags (`vX.Y.Z`).

For CUDA header/source/API changes:

- Run all available Python/fake-DLL/static checks locally.
- Inspect public declarations, native smoke coverage, validation tooling, and brace/argument consistency.
- If `nvcc` is unavailable, explicitly report that the DLL was not rebuilt.
- Leave RTX 3090 compile, primitive matrix, production recipe equivalence, benchmark, and stress tasks unchecked until executed.

If any required validation fails, fix it and rerun the relevant full set before commit.

## Git and artifacts

- Default branch and push target: `main` → `origin/main`.
- Stage only files that belong to the current task. Do not use `git add .` in a dirty workspace.
- Never commit user-provided release ZIPs, `outputs/logs/`, `outputs_validation/`, temporary images, generated reports, packaged validation archives, DLL build outputs, or unrelated changes.
- Do not discard, reset, overwrite, or reformat unrelated user changes.
- Use a concise commit message describing the completed outcome.
- Push every completed validated change unless the user explicitly says not to push.

Typical safe sequence:

```powershell
git status --short --branch
git add -- <explicit files>
git diff --cached --check
git commit -m "<concise outcome>"
git push origin main
git status -sb
```

## Final handoff

Report:

- What changed and which roadmap items were marked.
- CPU, fallback, GUI, CUDA, and compatibility impact as applicable.
- Exact validation commands and results.
- Any validation that could not run, especially `nvcc`/RTX 3090 work.
- Commit hash and push result.
- Remaining untracked user artifacts only when relevant.
