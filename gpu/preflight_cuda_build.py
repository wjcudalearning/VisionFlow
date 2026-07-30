from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_DIR.parent
REQUIRED_ABI_V1_EXPORTS = {
    "vf_adaptive_mean_u8",
    "vf_bgr_to_gray_u8",
    "vf_bgr_to_rgb_u8",
    "vf_context_create",
    "vf_context_destroy",
    "vf_context_stats",
    "vf_crop_u8",
    "vf_gaussian_blur_u8",
    "vf_gpu_abi_version",
    "vf_gpu_compute_capability",
    "vf_gpu_device_count",
    "vf_gpu_device_name",
    "vf_gpu_error_message",
    "vf_morphology_rect_u8",
    "vf_preprocess_401_2_u8",
    "vf_resize_gray_u8",
    "vf_threshold_u8",
}
REQUIRED_SMOKE_EXPORTS = {
    "vf_gpu_abi_version",
    "vf_gpu_device_count",
    "vf_gpu_device_name",
    "vf_gpu_compute_capability",
    "vf_gpu_error_message",
    "vf_bgr_to_gray_u8",
    "vf_context_create",
    "vf_context_destroy",
    "vf_context_stats",
    "vf_preprocess_401_2_u8",
}
OPTIONAL_EXPORT_GROUPS = {
    "generic_plan": {
        "vf_plan_query",
        "vf_plan_create",
        "vf_plan_execute",
        "vf_plan_destroy",
        "vf_dag_plan_query",
        "vf_dag_plan_create",
        "vf_dag_plan_execute",
        "vf_dag_plan_destroy",
    },
    "resident_roi": {
        "vf_context_upload_u8",
        "vf_plan_execute_roi",
        "vf_dag_plan_execute_roi",
    },
    "roi_batch": {
        "vf_gpu_memory_info",
        "vf_roi_batch_create",
        "vf_roi_batch_info",
        "vf_roi_batch_download_u8",
        "vf_roi_batch_destroy",
    },
    "timings": {"vf_context_last_timings"},
}
OPTIONAL_GENERIC_PLAN_EXPORTS = OPTIONAL_EXPORT_GROUPS["generic_plan"]
OPTIONAL_RESIDENT_ROI_EXPORTS = OPTIONAL_EXPORT_GROUPS["resident_roi"]
OPTIONAL_ROI_BATCH_EXPORTS = OPTIONAL_EXPORT_GROUPS["roi_batch"]
OPTIONAL_TIMING_EXPORTS = OPTIONAL_EXPORT_GROUPS["timings"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    path = PROJECT_DIR / "cuda_project.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AssertionError(f"Missing CUDA project manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Invalid JSON in {path}: {exc}") from exc
    if data.get("schema_version") != 1:
        raise AssertionError("cuda_project.json schema_version must be 1")
    return data


def _manifest_paths(manifest: dict, field: str) -> list[Path]:
    values = manifest.get(field)
    if not isinstance(values, list) or not values:
        raise AssertionError(f"cuda_project.json field '{field}' must be a non-empty list")
    paths = [PROJECT_DIR / str(value) for value in values]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise AssertionError(f"Missing files declared by '{field}': {missing}")
    return paths


def inspect_contract() -> dict:
    manifest = _load_manifest()
    header = PROJECT_DIR / "include" / "visionflow_cuda.h"
    source = PROJECT_DIR / "visionflow_cuda.cu"
    smoke = PROJECT_DIR / "test_cuda_api.cu"
    build_script = PROJECT_DIR / "build_cuda_dll.ps1"
    required_files = [header, source, smoke, PROJECT_DIR / "cuda_project.json", build_script]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise AssertionError(f"Missing action build contract files: {missing}")

    dll_sources = _manifest_paths(manifest, "dll_sources")
    if smoke in dll_sources:
        raise AssertionError("test_cuda_api.cu must not be included in dll_sources")

    test_targets = manifest.get("test_targets", [])
    if not isinstance(test_targets, list):
        raise AssertionError("cuda_project.json field 'test_targets' must be a list")
    declared_test_sources: set[Path] = set()
    for target in test_targets:
        if not isinstance(target, dict) or not target.get("name"):
            raise AssertionError("Every test target must contain a name")
        sources = target.get("sources")
        if not isinstance(sources, list) or not sources:
            raise AssertionError(f"Test target {target.get('name')} has no sources")
        for relative in sources:
            test_source = PROJECT_DIR / str(relative)
            if not test_source.is_file():
                raise AssertionError(f"Missing test source: {test_source}")
            declared_test_sources.add(test_source)
    if smoke not in declared_test_sources:
        raise AssertionError("test_cuda_api.cu must be declared under test_targets")

    include_dirs = manifest.get("include_dirs", [])
    if not isinstance(include_dirs, list):
        raise AssertionError("cuda_project.json field 'include_dirs' must be a list")
    missing_include_dirs = [
        str(PROJECT_DIR / str(relative))
        for relative in include_dirs
        if not (PROJECT_DIR / str(relative)).is_dir()
    ]
    if missing_include_dirs:
        raise AssertionError(f"Missing include directories: {missing_include_dirs}")

    header_text = header.read_text(encoding="utf-8")
    source_text = source.read_text(encoding="utf-8")
    smoke_text = smoke.read_text(encoding="utf-8")
    build_text = build_script.read_text(encoding="utf-8")
    runtime_text = (REPOSITORY_ROOT / "core" / "gpu_runtime.py").read_text(encoding="utf-8")
    validator_text = (PROJECT_DIR / "validate_cuda_dll.py").read_text(encoding="utf-8")

    header_exports = set(re.findall(r"VF_CUDA_API\s+int\s+(vf_[A-Za-z0-9_]+)\s*\(", header_text))
    source_exports = set(re.findall(r"VF_CUDA_API\s+int\s+(vf_[A-Za-z0-9_]+)\s*\(", source_text))
    errors: list[str] = []
    if header_exports != source_exports:
        errors.append(
            "header/source exports differ: "
            f"missing_definitions={sorted(header_exports - source_exports)}, "
            f"undeclared_definitions={sorted(source_exports - header_exports)}"
        )
    missing_required = REQUIRED_ABI_V1_EXPORTS - header_exports
    if missing_required:
        errors.append(f"required ABI v1 exports missing: {sorted(missing_required)}")
    missing_smoke = {name for name in REQUIRED_SMOKE_EXPORTS if name not in smoke_text}
    if missing_smoke:
        errors.append(f"native smoke does not reference required exports: {sorted(missing_smoke)}")
    for group_name, group in OPTIONAL_EXPORT_GROUPS.items():
        declared = group & header_exports
        if declared and declared != group:
            errors.append(f"optional export group '{group_name}' is incomplete: {sorted(group - declared)}")
        if declared:
            missing_runtime = {name for name in group if name not in runtime_text}
            if missing_runtime:
                errors.append(
                    f"Python runtime does not reference optional export group "
                    f"'{group_name}': {sorted(missing_runtime)}"
                )
            missing_smoke = {name for name in group if name not in smoke_text}
            if missing_smoke:
                errors.append(
                    f"native smoke does not reference optional export group "
                    f"'{group_name}': {sorted(missing_smoke)}"
                )
    missing_runtime = {name for name in REQUIRED_ABI_V1_EXPORTS if name not in runtime_text}
    if missing_runtime:
        errors.append(f"Python runtime does not reference ABI v1 exports: {sorted(missing_runtime)}")
    missing_validator = {
        name
        for name in ("vf_context_stats", "vf_preprocess_401_2_u8")
        if name not in runtime_text or "validate_context_reuse_matrix" not in validator_text
    }
    if missing_validator:
        errors.append(f"validator/runtime coverage is incomplete: {sorted(missing_validator)}")
    if "dll_sources" not in build_text or "test_targets" not in build_text:
        errors.append("project Action builder does not read the explicit source manifest")
    if errors:
        raise AssertionError("CUDA Action preflight failed:\n- " + "\n- ".join(errors))

    abi_match = re.search(r"#define\s+VF_CUDA_ABI_VERSION\s+(\d+)", header_text)
    if abi_match is None:
        raise AssertionError("VF_CUDA_ABI_VERSION is missing from the public header")

    tracked = [header, source, smoke, PROJECT_DIR / "cuda_project.json", build_script]
    return {
        "schema_version": 1,
        "project": manifest.get("output_name", PROJECT_DIR.name),
        "abi_version": int(abi_match.group(1)),
        "exports": sorted(header_exports),
        "optional_generic_plan_exports": sorted(OPTIONAL_GENERIC_PLAN_EXPORTS & header_exports),
        "optional_resident_roi_exports": sorted(OPTIONAL_RESIDENT_ROI_EXPORTS & header_exports),
        "optional_roi_batch_exports": sorted(OPTIONAL_ROI_BATCH_EXPORTS & header_exports),
        "optional_timing_exports": sorted(OPTIONAL_TIMING_EXPORTS & header_exports),
        "dll_sources": [
            str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
            for path in dll_sources
        ],
        "smoke_sources": [
            str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
            for path in sorted(declared_test_sources)
        ],
        "test_sources": sorted(str(path.relative_to(PROJECT_DIR)) for path in declared_test_sources),
        "sha256": {
            str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): _sha256(path)
            for path in tracked
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the standalone CUDA Action build contract.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()
    result = inspect_contract()
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
