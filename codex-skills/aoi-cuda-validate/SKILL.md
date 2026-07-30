---
name: aoi-cuda-validate
description: Build, inspect, and validate the optional CUDA backend for the local AOI_CVbased project. Use for CUDA .cu/header/C ABI changes, visionflow_cuda.dll or test_cuda_api.exe builds, RTX 3090/sm_86 testing, CPU/GPU equivalence, native plan validation, benchmarks, VRAM/stress checks, DLL export/dependency inspection, or diagnosing CUDA fallback and performance.
---

# AOI CUDA Validation

Treat source validation and real NVIDIA runtime validation as separate evidence. Never mark hardware work complete from static inspection alone.

## Preflight

1. Read repository `AGENT.md` and the CUDA/RTX sections of `Todo.md`.
2. Record commit hash, GPU, Driver, CUDA Toolkit, MSVC, Windows, and Python environment.
3. Confirm `nvidia-smi`, `nvcc --version`, and `where.exe cl` before claiming a native build is possible.
4. Preserve ABI v1 primitives and optional-export compatibility unless a versioned migration is explicitly approved.

## Source or API changes

1. Keep public declarations, `.cu` definitions, ctypes signatures, C++ smoke calls, validation tooling, and README usage synchronized.
2. Keep DLL and test executable source manifests separate; never compile every `.cu` by glob.
3. Test missing/old DLL routing and whole-detector CPU fallback with automated Python tests.
4. Run all local unit, compileall, static, argument/brace, and diff checks even when `nvcc` is unavailable.
5. Leave RTX compile/runtime Todo items unchecked when hardware commands did not run.

## RTX build and validation

1. Build from an x64 Native Tools PowerShell with `gpu\build_cuda_dll.ps1 -Architecture sm_86` for RTX 3090.
2. Run `gpu\test_cuda_api.exe` and verify ABI, device, compute capability, context, and current smoke operations.
3. Inspect `dumpbin /exports` and `dumpbin /dependents`; verify expected `vf_` exports and deployment dependencies.
4. Run `gpu\validate_cuda_dll.py` for structured primitives, preprocessing plans, persistent allocation reuse, and 4K benchmarks.
5. Compare all production PASS/NG samples on CPU and GPU, including tiles, defects, bbox, area, confidence, metadata, and fallback logs.
6. Warm up before timing. Report kernel/preprocess, pure detection, and output-inclusive end-to-end metrics separately with median and P95.
7. Run repeated/long tests while watching VRAM, CUDA errors, crashes, temperature, and power. Do not accept unexplained fallback.

Update only the Todo checks supported by captured evidence. Follow `aoi-verify-push` for repository commit/push; do not commit DLL/LIB/EXE/build logs unless an explicitly approved release workflow requires an artifact outside git.
