---
name: aoi-detector-development
description: Add, modify, register, or migrate detectors in the local AOI_CVbased project while preserving recipe behavior, metadata, CPU correctness, and optional GPU fallback. Use for detector_*.py work, detector registry/labels/recipe parameters, PreprocessPlan operators, CPU/CUDA routing, detector equivalence tests, or requests to add many future detectors safely.
---

# AOI Detector Development

Build detectors around shared preprocessing and a stable CPU reference instead of detector-specific GPU workflows.

## Workflow

1. Read repository `AGENT.md` and the detector/P1/P3 sections of `Todo.md`.
2. Inspect the nearest detector, `BaseDetector`, `core/detector_manager.py`, relevant recipes, GUI labels/designer behavior, reporter assumptions, and tests.
3. Define the detector contract before coding: ID, defaults, input/ROI semantics, PASS/NG rule, defect type, bbox coordinates, confidence, metadata, ordering, and tolerances.
4. Keep detector-specific geometry, filtering, and metadata in `detectors/`.
5. Express reusable image preprocessing as an immutable `PreprocessPlan` with shared typed operators. Add a reusable operator when necessary; do not add a detector-named CUDA export.
6. Make `CpuPreprocessExecutor` behavior the correctness reference. Reject unsupported CUDA semantics rather than silently substituting a different algorithm.
7. Register a new detector in `core/detector_manager.py`; update labels, designer parameters, recipes, reporting special cases, and documentation only where the contract requires it.
8. Preserve full-detector CPU restart on GPU failure and old/missing DLL behavior.

## Tests

Cover the relevant matrix:

- CPU result and edge inputs.
- Detector manager registration and recipe parameter round trip.
- PreprocessPlan output against direct OpenCV.
- Fused/native-plan routing when supported.
- Legacy primitive routing when the optional export is missing.
- Missing GPU and injected GPU failure with full CPU restart.
- PASS/NG, defect count, bbox, area, confidence, metadata, and deterministic ordering.

Run the repository validation and delivery workflow from `AGENT.md`; use `aoi-verify-push` for the common Todo/commit/push finish.
