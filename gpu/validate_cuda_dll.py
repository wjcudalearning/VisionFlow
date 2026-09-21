from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpu_runtime import GpuRuntime  # noqa: E402
from core.gpu_equivalence import compare_arrays  # noqa: E402
from core.pipeline import AOIPipeline  # noqa: E402
from core.preprocess_plan import (  # noqa: E402
    AdaptiveMean,
    CpuPreprocessExecutor,
    CpuPreprocessDagExecutor,
    Gaussian,
    Gray,
    Morphology,
    PreprocessPlan,
    PreprocessDagNode,
    PreprocessDagPlan,
    Resize,
    Threshold,
)
from core.recipe_manager import RecipeManager  # noqa: E402


PRODUCTION_RECIPES = (
    "PRODUCT_A_AOI_01.yaml",
    "PRODUCT_A_CIRCLE_401_1_AOI_01.yaml",
    "PRODUCT_A_NEGATIVE_401_AOI_01.yaml",
    "PRODUCT_A_WHITE_RATIO_401_2_AOI_01.yaml",
    "PRODUCT_A_FRAME_900_AOI_01.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate visionflow_cuda.dll against the CPU AOI path.")
    parser.add_argument("--dll", default="gpu/visionflow_cuda.dll", help="CUDA DLL path.")
    parser.add_argument("--image", help="Optional real image for full CPU/GPU pipeline comparison.")
    parser.add_argument("--recipe", help="Recipe used with --image.")
    parser.add_argument(
        "--production-manifest",
        help="YAML manifest containing one PASS and one NG case for every production recipe.",
    )
    parser.add_argument("--benchmark", type=int, default=20, help="Primitive benchmark repetitions.")
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up calls excluded from benchmark statistics.")
    parser.add_argument(
        "--stress", type=int, nargs="*", default=[], metavar="COUNT",
        help="Run cumulative persistent-plan stress checkpoints, for example: --stress 10 100 1000.",
    )
    parser.add_argument(
        "--crossover", action="store_true",
        help="Benchmark small-to-large CPU/GPU native-plan crossover without changing production policy.",
    )
    parser.add_argument(
        "--morphology-profile", action="store_true",
        help="Profile detector-401-style morphology iterations and native CUDA event share.",
    )
    parser.add_argument(
        "--roi-batch-matrix", action="store_true",
        help="Validate 8/16/32/64 ROI batches of production-sized ROIs on a 16384x13000 resident image.",
    )
    parser.add_argument(
        "--resize-area-pipeline", action="store_true",
        help="Compare full CPU/GPU pipelines of the 401-CS-AP-1 production recipe across process_scale values.",
    )
    parser.add_argument("--json-output", help="Write validation, benchmark, device and commit metadata as JSON.")
    args = parser.parse_args()
    if bool(args.image) != bool(args.recipe):
        parser.error("--image and --recipe must be provided together")
    if any(count <= 0 for count in args.stress):
        parser.error("--stress checkpoints must be positive")
    return args


def compare(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    contract_id: str = "comparison.bit_exact",
) -> dict:
    result = compare_arrays(name, actual, expected, contract_id)
    print(f"PASS {name}: {result}")
    return result


def area_resize_cases(seed: int = 20260914, random_cases: int = 160) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Source/target (height, width) pairs covering every OpenCV INTER_AREA branch."""
    rng = np.random.default_rng(seed)
    cases = [
        ((3, 7), (1, 2)), ((17, 19), (16, 18)), ((31, 47), (10, 15)), ((64, 96), (21, 32)),
        ((101, 173), (100, 172)), ((64, 96), (64, 96)),          # copy
        ((64, 96), (32, 48)), ((65, 97), (32, 48)),              # 2x2 fast / non-integer
        ((60, 90), (20, 30)), ((64, 64), (16, 32)),              # integer fast area
        ((50, 70), (50, 35)), ((70, 50), (35, 50)),              # one axis unchanged
        ((1, 500), (1, 7)), ((500, 1), (7, 1)), ((401, 301), (7, 3)),
        ((2160, 3840), (1080, 1920)), ((2160, 3840), (1000, 1777)), ((4096, 4096), (1, 1)),
        ((13000, 2300), (4333, 767)),
    ]
    for _ in range(random_cases):
        height, width = (int(value) for value in rng.integers(1, 1500, size=2))
        cases.append(((height, width), (int(rng.integers(1, height + 1)), int(rng.integers(1, width + 1)))))
    return cases


def validate_area_resize_matrix(runtime: GpuRuntime) -> dict:
    """CUDA Resize(area) must be pixel-identical to cv2.INTER_AREA (no tolerance)."""
    rng = np.random.default_rng(20260915)
    compared = 0
    pixels = 0
    for index, ((source_height, source_width), (target_height, target_width)) in enumerate(area_resize_cases()):
        pattern = index % 3
        if pattern == 0:
            source = rng.integers(0, 256, (source_height, source_width), dtype=np.uint8)
        elif pattern == 1:
            source = ((rng.random((source_height, source_width)) > 0.5) * 255).astype(np.uint8)
        else:
            source = rng.integers(0, 256, (source_height * 2, source_width + 3), dtype=np.uint8)[::2, 3:]
        expected = cv2.resize(source, (target_width, target_height), interpolation=cv2.INTER_AREA)
        label = f"resize_area_{source_height}x{source_width}_to_{target_height}x{target_width}"
        contiguous = np.ascontiguousarray(source)
        outputs = {"stateless": runtime.resize_gray(contiguous, target_width, target_height)}
        if runtime.supports_native_plan:
            plan = PreprocessPlan((Resize(target_width, target_height, "area"),), name=label)
            outputs["native"] = runtime.execute_plan(contiguous, plan)
        for route, actual in outputs.items():
            compare(
                f"{route}_{label}", actual, expected, "preprocess.resize_area_u8"
            )
            compared += 1
        pixels += int(expected.size)
    result = {"name": "resize_area_matrix", "outputs": compared, "target_pixels": pixels, "max_diff": 0}
    print(f"PASS resize_area_matrix: {result}")
    return result


def validate_context_reuse_matrix(runtime: GpuRuntime) -> list[dict]:
    if not runtime.supports_native_plan:
        print(f"SKIP context reuse matrix: {runtime.native_plan_unavailable_reason}")
        return []
    cases = (
        (
            "bgr_initial",
            np.zeros((64, 96, 3), dtype=np.uint8),
            PreprocessPlan((Gray(), Threshold(1, 255, False)), name="reuse_bgr_initial"),
        ),
        (
            "bgr_grow",
            np.zeros((96, 128, 3), dtype=np.uint8),
            PreprocessPlan((Gaussian(5), Gray(), Threshold(1, 255, False)), name="reuse_bgr_grow"),
        ),
        (
            "gray_channel_switch",
            np.zeros((48, 64), dtype=np.uint8),
            PreprocessPlan((Threshold(1, 255, False), Morphology("close", 3, 2)), name="reuse_gray"),
        ),
        (
            "bgr_shrink_parameter_change",
            np.zeros((32, 48, 3), dtype=np.uint8),
            PreprocessPlan((Gray(), Threshold(127, 255, False)), name="reuse_bgr_changed"),
        ),
    )
    executor = CpuPreprocessExecutor()
    metrics = []
    for cycle in ("warm", "reused"):
        for case_name, image, plan in cases:
            metrics.append(compare(
                f"context_{cycle}_{case_name}",
                runtime.execute_plan(image, plan),
                executor.execute(image, plan),
            ))
        if cycle == "warm":
            warmed = runtime.performance_stats()["persistent_context"]
    reused = runtime.performance_stats()["persistent_context"]
    if warmed.get("allocation_count") != reused.get("allocation_count"):
        raise AssertionError(
            "Persistent context allocated again after shape/channel/parameter matrix warm-up: "
            f"warmed={warmed}, reused={reused}"
        )
    if reused.get("accounting") == "detailed_v1":
        breakdown = reused.get("breakdown", {})
        if sum(int(value) for value in breakdown.values()) != reused.get("reserved_bytes"):
            raise AssertionError(f"CUDA context memory breakdown does not sum to total: {reused}")
        if int(reused.get("peak_reserved_bytes") or 0) < int(reused.get("reserved_bytes") or 0):
            raise AssertionError(f"CUDA context peak memory is below current memory: {reused}")
    print(f"PASS context reuse matrix: warmed={warmed}, reused={reused}")
    return metrics


def validate_primitives(runtime: GpuRuntime) -> list[dict]:
    rng = np.random.default_rng(20260714)
    bgr = rng.integers(0, 256, size=(128, 192, 3), dtype=np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
    metrics = []
    metrics.append(compare(
        "bgr_to_rgb", runtime.bgr_to_rgb(bgr), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        "preprocess.bgr_to_rgb_u8",
    ))
    metrics.append(compare("bgr_to_gray", runtime.bgr_to_gray(bgr), gray, "preprocess.gray_u8"))
    metrics.append(compare(
        "crop_bgr", runtime.crop(bgr, 17, 13, 91, 67), bgr[13:80, 17:108],
        "preprocess.crop_u8",
    ))
    metrics.append(
        compare(
            "resize_gray",
            runtime.resize_gray(gray, 96, 64),
            cv2.resize(gray, (96, 64), interpolation=cv2.INTER_AREA),
            "preprocess.resize_area_u8",
        )
    )
    metrics.append(validate_area_resize_matrix(runtime))
    metrics.append(
        compare(
            "gaussian_blur_gray",
            runtime.gaussian_blur(gray, 5),
            cv2.GaussianBlur(gray, (5, 5), 0),
            "preprocess.gaussian_u8",
        )
    )
    metrics.append(compare(
        "global_threshold", runtime.threshold(gray, 128, 255, False), binary,
        "preprocess.threshold_u8",
    ))
    structured_gray = {
        "random_odd": rng.integers(0, 256, size=(65, 97), dtype=np.uint8),
        "black": np.zeros((63, 79), dtype=np.uint8),
        "white": np.full((63, 79), 255, dtype=np.uint8),
        "checker": ((np.indices((63, 79)).sum(axis=0) % 2) * 255).astype(np.uint8),
        "non_contiguous": gray[:, ::2],
    }
    for case_name, case in structured_gray.items():
        for kernel_size in (3, 5, 15, 25, 45):
            expected_gaussian = cv2.GaussianBlur(case, (kernel_size, kernel_size), 0)
            metrics.append(
                compare(
                    f"gaussian_{case_name}_k{kernel_size}",
                    runtime.gaussian_blur(case, kernel_size),
                    expected_gaussian,
                    "preprocess.gaussian_u8",
                )
            )
    expected_gaussian_bgr = cv2.GaussianBlur(bgr, (15, 15), 0)
    metrics.append(
        compare(
            "gaussian_bgr_k15",
            runtime.gaussian_blur(bgr, 15),
            expected_gaussian_bgr,
            "preprocess.gaussian_u8",
        )
    )
    adaptive_cases = (
        ("random_binary_b3_c2", structured_gray["random_odd"], 3, 2.0, False),
        ("random_binary_b11_cneg2", structured_gray["random_odd"], 11, -2.0, False),
        ("random_inverse_b35_c24", structured_gray["random_odd"], 35, 2.4, True),
        ("black_binary_b11", structured_gray["black"], 11, 2.0, False),
        ("white_inverse_b11", structured_gray["white"], 11, 2.0, True),
        ("checker_binary_b35", structured_gray["checker"], 35, -2.0, False),
        ("non_contiguous_inverse_b11", structured_gray["non_contiguous"], 11, -2.0, True),
    )
    for case_name, case, block_size, adaptive_c, invert in adaptive_cases:
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        expected_adaptive = cv2.adaptiveThreshold(
            case,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            threshold_type,
            block_size,
            adaptive_c,
        )
        metrics.append(
            compare(
                f"adaptive_{case_name}",
                runtime.adaptive_threshold(case, block_size, adaptive_c, 255, invert),
                expected_adaptive,
                "preprocess.adaptive_mean_u8",
            )
        )
    if runtime.supports_fused_401_2:
        fused_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        fused_expected = cv2.adaptiveThreshold(
            cv2.GaussianBlur(fused_gray, (25, 25), 0),
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            35,
            -2.0,
        )
        metrics.append(
            compare(
                "fused_401_2_bgr",
                runtime.preprocess_401_2(bgr, 25, 35, -2.0, 255, True),
                fused_expected,
                "preprocess.adaptive_mean_u8",
            )
        )
        first_context_stats = runtime.performance_stats()["persistent_context"]
        repeated = runtime.preprocess_401_2(bgr, 25, 35, -2.0, 255, True)
        metrics.append(
            compare(
                "fused_401_2_bgr_reused_context",
                repeated,
                fused_expected,
                "preprocess.adaptive_mean_u8",
            )
        )
        second_context_stats = runtime.performance_stats()["persistent_context"]
        if first_context_stats.get("allocation_count") != second_context_stats.get("allocation_count"):
            raise AssertionError(
                "fused_401_2 context allocated again for an unchanged image shape: "
                f"first={first_context_stats}, second={second_context_stats}"
            )
        print(f"PASS fused_401_2 persistent context reuse: {second_context_stats}")
    else:
        print(f"SKIP fused_401_2_bgr: {runtime.fused_unavailable_reason}")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for operation, cv_operation in (
        ("open", cv2.MORPH_OPEN),
        ("close", cv2.MORPH_CLOSE),
        ("dilate", cv2.MORPH_DILATE),
        ("erode", cv2.MORPH_ERODE),
    ):
        expected = (
            cv2.morphologyEx(binary, cv_operation, kernel, iterations=1)
            if operation in {"open", "close"}
            else cv2.dilate(binary, kernel, iterations=1)
            if operation == "dilate"
            else cv2.erode(binary, kernel, iterations=1)
        )
        metrics.append(compare(
            f"morphology_{operation}", runtime.morphology(binary, operation, 3, 1), expected,
            "preprocess.morphology_u8",
        ))
    # The 5x5 path uses a shared-memory separable kernel. Cover partial CUDA
    # blocks, neutral borders, BGR channels, and repeated open/close passes.
    morphology_rng = np.random.default_rng(20260914)
    for shape in ((1, 1), (17, 19), (65, 67), (65, 67, 3)):
        source = morphology_rng.integers(0, 256, shape, dtype=np.uint8)
        for operation, cv_operation, iterations in (
            ("open", cv2.MORPH_OPEN, 10),
            ("close", cv2.MORPH_CLOSE, 2),
            ("erode", cv2.MORPH_ERODE, 1),
            ("dilate", cv2.MORPH_DILATE, 1),
        ):
            expected = cv2.morphologyEx(
                source, cv_operation, np.ones((5, 5), dtype=np.uint8),
                iterations=iterations,
            )
            label = f"morphology_k5_{operation}_i{iterations}_{'x'.join(map(str, shape))}"
            metrics.append(compare(
                label, runtime.morphology(source, operation, 5, iterations), expected,
                "preprocess.morphology_u8",
            ))
            if runtime.supports_native_plan:
                plan = PreprocessPlan(
                    (Morphology(operation, 5, iterations),), name=label,
                )
                metrics.append(compare(
                    f"native_{label}", runtime.execute_plan(source, plan), expected,
                    "preprocess.morphology_u8",
                ))
    strided_morphology = morphology_rng.integers(
        0, 256, (67, 139, 3), dtype=np.uint8,
    )[1:-1, 1:-1:2]
    strided_expected = cv2.morphologyEx(
        strided_morphology, cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8), iterations=10,
    )
    metrics.append(compare(
        "morphology_k5_strided_bgr", runtime.morphology(
            strided_morphology, "open", 5, 10,
        ), strided_expected, "preprocess.morphology_u8",
    ))
    if runtime.supports_native_plan:
        strided_plan = PreprocessPlan(
            (Morphology("open", 5, 10),), name="native_k5_strided_bgr",
        )
        metrics.append(compare(
            "native_morphology_k5_strided_bgr",
            runtime.execute_plan(strided_morphology, strided_plan), strided_expected,
            "preprocess.morphology_u8",
        ))
    if runtime.supports_native_plan:
        native_plans = (
            PreprocessPlan(
                (Gray(), Threshold(128, 255, False)),
                name="native_gray_threshold",
            ),
            PreprocessPlan(
                (Gaussian(5), Morphology("open", 3, 2), Gray(), AdaptiveMean(11, -2.0, 255, True)),
                name="native_401_style",
            ),
            PreprocessPlan(
                (Gray(), Resize(96, 64, "area"), Gaussian(5), AdaptiveMean(11, -2.0, 255, True)),
                name="native_401_1_area_resize",
            ),
        )
        cpu_executor = CpuPreprocessExecutor()
        for plan in native_plans:
            expected = cpu_executor.execute(bgr, plan)
            before_calls = runtime.performance_stats()["call_count"]
            actual = runtime.execute_plan(bgr, plan)
            after_first = runtime.performance_stats()
            metrics.append(
                compare(
                    plan.name,
                    actual,
                    expected,
                    "preprocess.adaptive_mean_u8" if "401" in plan.name else "preprocess.gaussian_u8",
                )
            )
            if after_first["call_count"] != before_calls + 1:
                raise AssertionError(f"{plan.name} did not execute as one native CUDA round trip")
            first_context_stats = after_first["persistent_context"]
            repeated = runtime.execute_plan(bgr, plan)
            metrics.append(
                compare(
                    f"{plan.name}_reused",
                    repeated,
                    expected,
                    "preprocess.adaptive_mean_u8" if "401" in plan.name else "preprocess.gaussian_u8",
                )
            )
            second_context_stats = runtime.performance_stats()["persistent_context"]
            if first_context_stats.get("allocation_count") != second_context_stats.get("allocation_count"):
                raise AssertionError(
                    f"{plan.name} allocated again for an unchanged shape: "
                    f"first={first_context_stats}, second={second_context_stats}"
                )
            print(f"PASS {plan.name} native plan reuse: {second_context_stats}")
    else:
        print(f"SKIP generic native plan: {runtime.native_plan_unavailable_reason}")
    if runtime.supports_native_dag_plan:
        morphology_dag = PreprocessDagPlan(
            name="native_k5_bgr_morphology_dag",
            nodes=(
                PreprocessDagNode("morph", "root", Morphology("open", 5, 10)),
                PreprocessDagNode("gray", "root", Gray()),
            ),
            outputs=("morph", "gray"),
        )
        morphology_source = np.random.default_rng(20260915).integers(
            0, 256, (65, 67, 3), dtype=np.uint8,
        )
        morphology_expected = CpuPreprocessDagExecutor().execute(
            morphology_source, morphology_dag,
        )
        morphology_actual = runtime.execute_dag_plan(morphology_source, morphology_dag)
        for name in morphology_dag.outputs:
            metrics.append(compare(
                f"{morphology_dag.name}_{name}",
                morphology_actual[name], morphology_expected[name],
                "preprocess.morphology_u8" if name == "morph" else "preprocess.gray_u8",
            ))
        dag_plan = PreprocessDagPlan(
            name="native_900_shared_gray",
            nodes=(
                PreprocessDagNode("gray", "root", Gray()),
                PreprocessDagNode("outer_mask", "gray", Threshold(128, 255, True)),
                PreprocessDagNode("inner_mask", "gray", AdaptiveMean(11, -2.0, 255, False)),
            ),
            outputs=("outer_mask", "inner_mask"),
        )
        expected_outputs = CpuPreprocessDagExecutor().execute(bgr, dag_plan)
        before_calls = runtime.performance_stats()["call_count"]
        actual_outputs = runtime.execute_dag_plan(bgr, dag_plan)
        after_first = runtime.performance_stats()
        for name in dag_plan.outputs:
            metrics.append(compare(
                f"{dag_plan.name}_{name}", actual_outputs[name], expected_outputs[name],
                "preprocess.adaptive_mean_u8" if name == "inner_mask" else "preprocess.threshold_u8",
            ))
        if after_first["call_count"] != before_calls + 1:
            raise AssertionError("native 900 DAG did not execute as one CUDA call")
        first_context_stats = after_first["persistent_context"]
        runtime.execute_dag_plan(bgr, dag_plan)
        second_context_stats = runtime.performance_stats()["persistent_context"]
        if first_context_stats.get("allocation_count") != second_context_stats.get("allocation_count"):
            raise AssertionError(
                "native 900 DAG allocated again for an unchanged shape: "
                f"first={first_context_stats}, second={second_context_stats}"
            )
        print(f"PASS native 900 DAG reuse: {second_context_stats}")
    else:
        print(f"SKIP generic native DAG plan: {runtime.native_dag_plan_unavailable_reason}")
    if runtime.supports_resident_roi:
        resident = runtime.upload_image(bgr)
        device_roi = resident.roi(17, 13, 64, 64)
        host_roi = bgr[13:77, 17:81]
        roi_plan = PreprocessPlan((Gray(), Threshold(128, 255, False)), name="resident_roi_plan")
        expected_roi = CpuPreprocessExecutor().execute(host_roi, roi_plan)
        before = runtime.performance_stats()
        actual_roi = runtime.execute_plan(host_roi, roi_plan, device_roi=device_roi)
        after = runtime.performance_stats()
        metrics.append(compare("resident_roi_plan", actual_roi, expected_roi, "preprocess.gray_u8"))
        if after["host_to_device_bytes"] != before["host_to_device_bytes"]:
            raise AssertionError("resident linear ROI unexpectedly uploaded detector input")

        roi_dag = PreprocessDagPlan(
            name="resident_roi_dag",
            nodes=(
                PreprocessDagNode("gray", "root", Gray()),
                PreprocessDagNode("dark", "gray", Threshold(128, 255, True)),
                PreprocessDagNode("light", "gray", Threshold(128, 255, False)),
            ),
            outputs=("dark", "light"),
        )
        expected_dag = CpuPreprocessDagExecutor().execute(host_roi, roi_dag)
        before = runtime.performance_stats()
        actual_dag = runtime.execute_dag_plan(host_roi, roi_dag, device_roi=device_roi)
        after = runtime.performance_stats()
        for name in roi_dag.outputs:
            metrics.append(compare(
                f"resident_roi_dag_{name}", actual_dag[name], expected_dag[name],
                "preprocess.gray_u8",
            ))
        if after["host_to_device_bytes"] != before["host_to_device_bytes"]:
            raise AssertionError("resident DAG ROI unexpectedly uploaded detector input")
        print(f"PASS resident image/ROI routing: generation={resident.generation}")
        if runtime.supports_roi_batch:
            coordinates = [
                (x, y, 16, 16)
                for y in range(0, 128, 16)
                for x in range(0, 128, 16)
            ]
            recommended = runtime.recommended_roi_batch_size(16, 16, 3)
            for batch_size in (8, 16, 32, 64):
                with runtime.create_roi_batch(resident, coordinates[:batch_size]) as batch:
                    metrics.append(compare(
                        f"roi_batch_{batch_size}_first",
                        batch.download(0),
                        bgr[0:16, 0:16],
                        "transfer.u8",
                    ))
                    last_x, last_y, last_width, last_height = coordinates[batch_size - 1]
                    metrics.append(compare(
                        f"roi_batch_{batch_size}_last",
                        batch.download(batch_size - 1),
                        bgr[last_y:last_y + last_height, last_x:last_x + last_width],
                        "transfer.u8",
                    ))
            print(
                f"PASS ROI coordinate batches 8/16/32/64; "
                f"recommended={recommended} memory={runtime.memory_info()}"
            )
        else:
            print("SKIP ROI coordinate batch: optional exports unavailable")
    else:
        print("SKIP resident image/ROI routing: optional exports unavailable")
    metrics.extend(validate_context_reuse_matrix(runtime))
    return metrics


def validate_roi_batch_matrix(runtime: GpuRuntime, repetitions: int = 5) -> dict:
    """Production-sized ROI batches: every pixel, timing, VRAM plateau and OOM downshift."""
    if not runtime.supports_roi_batch:
        print(f"SKIP ROI batch matrix: {runtime.native_plan_unavailable_reason or 'ROI batch exports unavailable'}")
        return {}
    rng = np.random.default_rng(20260916)
    image = rng.integers(0, 256, (13000, 16384, 3), dtype=np.uint8)
    resident = runtime.upload_image(image)
    report = {"image_shape": list(image.shape), "repetitions": repetitions, "cases": []}
    for side in (256, 512, 1024):
        coordinates = [
            (x, y, side, side)
            for y in range(0, 13000 - side + 1, 1200)
            for x in range(0, 16384 - side + 1, 997)
        ][:64]
        recommended = runtime.recommended_roi_batch_size(side, side, 3)
        for batch_size in (8, 16, 32, 64):
            regions = coordinates[:batch_size]
            memory_before = runtime.memory_info()["free_bytes"]
            create_ms, download_ms, free_during = [], [], []
            for repetition in range(repetitions):
                started = time.perf_counter()
                with runtime.create_roi_batch(resident, regions) as batch:
                    created = time.perf_counter()
                    free_during.append(runtime.memory_info()["free_bytes"])
                    for index, (x, y, width, height) in enumerate(regions):
                        roi = batch.download(index)
                        if repetition == 0 and not np.array_equal(roi, image[y:y + height, x:x + width]):
                            raise AssertionError(f"ROI batch {side}px x{batch_size} index {index} differs")
                    download_ms.append((time.perf_counter() - created) * 1000.0)
                create_ms.append((created - started) * 1000.0)
            memory_after = runtime.memory_info()["free_bytes"]
            case = {
                "roi": [side, side], "batch_size": batch_size, "recommended_batch_size": recommended,
                "batch_bytes": side * side * 3 * batch_size,
                "create_median_ms": round(statistics.median(create_ms), 3),
                "download_all_median_ms": round(statistics.median(download_ms), 3),
                "device_bytes_in_use_during_batch": int(memory_before - min(free_during)),
                "device_bytes_retained_after_close": int(memory_before - memory_after),
            }
            if case["device_bytes_retained_after_close"] > 64 * 1024 * 1024:
                raise AssertionError(f"ROI batch memory did not return to its plateau: {case}")
            report["cases"].append(case)
            print(f"PASS roi_batch_matrix: {case}")
    large = [(192 + index * 2800, 500, 2000, 12000) for index in range(6)] * 11  # 66 ROIs, about 4.8 GiB at once
    batches = []
    for batch in runtime.iter_roi_batches(resident, large, candidates=(8, 16, 32, 64)):
        x, y, width, height = large[batch.offset + batch.count - 1]
        if not np.array_equal(batch.download(batch.count - 1), image[y:y + height, x:x + width]):
            raise AssertionError("large ROI batch download differs")
        batches.append(batch.count)
    report["large_roi_batches"] = {
        "roi": [2000, 12000], "roi_count": len(large), "batch_counts": batches,
        "recommended_batch_size": runtime.recommended_roi_batch_size(2000, 12000, 3),
        "registered_batches_after": len(runtime._roi_batches),
    }
    if sum(batches) != len(large) or runtime._roi_batches:
        raise AssertionError(f"large ROI batch iteration lost ROIs or handles: {report['large_roi_batches']}")
    print(f"PASS roi_batch_large: {report['large_roi_batches']}")
    return report


def _timing_summary(operation, repetitions: int, warmup: int) -> dict:
    cold_started = time.perf_counter()
    operation()
    cold_ms = (time.perf_counter() - cold_started) * 1000.0
    for _ in range(max(0, int(warmup))):
        operation()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    samples = []
    for _ in range(max(1, int(repetitions))):
        started = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started) * 1000.0)
    wall_sec = max(time.perf_counter() - wall_started, 1e-12)
    process_cpu_sec = max(time.process_time() - cpu_started, 0.0)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, int(np.ceil(len(ordered) * 0.95)) - 1))
    return {
        "cold_ms": round(cold_ms, 3),
        "average_ms": round(statistics.fmean(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "process_cpu_percent": round(process_cpu_sec / wall_sec * 100.0, 1),
    }


def _p95(samples) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(np.ceil(len(ordered) * 0.95)) - 1))
    return float(ordered[index])


def gpu_telemetry_snapshot() -> dict:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {}
    query = (
        "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,"
        "driver_version,name"
    )
    completed = subprocess.run(
        [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"error": completed.stderr.strip()}
    values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
    keys = (
        "utilization_percent", "memory_used_mib", "memory_total_mib",
        "temperature_c", "power_w", "driver_version", "name",
    )
    numeric = set(keys[:5])
    return {
        key: (float(value) if key in numeric and value not in {"N/A", "[N/A]"} else value)
        for key, value in zip(keys, values)
    }


def environment_snapshot(image_path: str | None = None, recipe_path: str | None = None) -> dict:
    return {
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "ram_total_bytes": _total_ram_bytes(),
        "gpu": gpu_telemetry_snapshot(),
        "image": str(Path(image_path).resolve()) if image_path else "",
        "recipe": str(Path(recipe_path).resolve()) if recipe_path else "",
    }


def _total_ram_bytes() -> int:
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        return int(status.total_physical) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else 0
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError):
            pass
    return 0


def benchmark(runtime: GpuRuntime, repetitions: int, warmup: int = 5) -> dict:
    if repetitions <= 0:
        return {}
    image = np.random.default_rng(7).integers(0, 256, size=(2160, 3840, 3), dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    runtime.bgr_to_gray(image)
    runtime.gaussian_blur(gray, 45)
    runtime.adaptive_threshold(gray, 35, -2.0, 255, False)
    operations = (
        ("bgr_to_gray_4k", lambda: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), lambda: runtime.bgr_to_gray(image)),
        ("gaussian_gray_4k_k45", lambda: cv2.GaussianBlur(gray, (45, 45), 0), lambda: runtime.gaussian_blur(gray, 45)),
        (
            "adaptive_mean_gray_4k_b35",
            lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, -2.0),
            lambda: runtime.adaptive_threshold(gray, 35, -2.0, 255, False),
        ),
    )
    if runtime.supports_fused_401_2:
        operations += (
            (
                "fused_401_2_bgr_4k",
                lambda: cv2.adaptiveThreshold(
                    cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (25, 25), 0),
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY_INV,
                    35,
                    -2.0,
                ),
                lambda: runtime.preprocess_401_2(image, 25, 35, -2.0, 255, True),
            ),
        )
    measurements = []
    for name, cpu_operation, gpu_operation in operations:
        telemetry_before = gpu_telemetry_snapshot()
        cpu = _timing_summary(cpu_operation, repetitions, warmup)
        gpu = _timing_summary(gpu_operation, repetitions, warmup)
        telemetry_after = gpu_telemetry_snapshot()
        cpu_ms = cpu["median_ms"]
        gpu_ms = gpu["median_ms"]
        measurements.append(
            {
                "operation": name,
                "cpu": cpu,
                "gpu_including_transfer": gpu,
                "cpu_average_ms": cpu["average_ms"],
                "gpu_average_ms_including_transfer": gpu["average_ms"],
                "speedup": round(cpu_ms / gpu_ms, 3) if gpu_ms > 0 else None,
                "gpu_telemetry_before": telemetry_before,
                "gpu_telemetry_after": telemetry_after,
            }
        )
    result = {
        "repetitions": repetitions,
        "warmup": warmup,
        "image_shape": list(image.shape),
        "measurements": measurements,
        "gpu_host_metrics": runtime.performance_stats(),
    }
    print(f"BENCHMARK {result}")
    return result


def stress_persistent_plan(runtime: GpuRuntime, checkpoints, warmup: int = 5) -> dict:
    ordered = sorted({int(count) for count in checkpoints if int(count) > 0})
    if not ordered:
        return {}
    if not runtime.supports_native_plan:
        raise AssertionError("Persistent-plan stress requires the generic native plan ABI")
    image = np.random.default_rng(20260717).integers(
        0, 256, size=(512, 512, 3), dtype=np.uint8
    )
    plan = PreprocessPlan(
        (Gaussian(5), Gray(), AdaptiveMean(11, -2.0, 255, True)),
        name="stress_401_style",
    )
    for _ in range(max(0, int(warmup))):
        runtime.execute_plan(image, plan)
    baseline = runtime.performance_stats()["persistent_context"]
    memory_before = runtime.memory_info()
    samples_ms = []
    snapshots = []
    completed = 0
    checksum = 0
    for checkpoint in ordered:
        while completed < checkpoint:
            started = time.perf_counter()
            output = runtime.execute_plan(image, plan)
            samples_ms.append((time.perf_counter() - started) * 1000.0)
            checksum = (checksum + int(output[completed % output.shape[0], completed % output.shape[1]])) % (2**32)
            completed += 1
        context = runtime.performance_stats()["persistent_context"]
        if context.get("allocation_count") != baseline.get("allocation_count"):
            raise AssertionError(
                f"Persistent plan allocated after warm-up at checkpoint {checkpoint}: "
                f"baseline={baseline}, current={context}"
            )
        current_samples = samples_ms[:checkpoint]
        snapshots.append({
            "completed": checkpoint,
            "average_ms": round(statistics.fmean(current_samples), 3),
            "median_ms": round(statistics.median(current_samples), 3),
            "p95_ms": round(_p95(current_samples), 3),
            "persistent_context": context,
            "gpu_memory": runtime.memory_info(),
            "gpu_telemetry": gpu_telemetry_snapshot(),
        })
    result = {
        "warmup": max(0, int(warmup)),
        "checkpoints": ordered,
        "image_shape": list(image.shape),
        "checksum": checksum,
        "gpu_memory_before": memory_before,
        "snapshots": snapshots,
        "gpu_metrics": runtime.performance_stats(),
    }
    print(f"STRESS {result}")
    return result


def benchmark_crossover(
    runtime: GpuRuntime,
    repetitions: int = 20,
    warmup: int = 5,
    sizes=(64, 128, 256, 512, 1024),
) -> dict:
    if not runtime.supports_native_plan:
        raise AssertionError("Crossover benchmark requires the generic native plan ABI")
    plan = PreprocessPlan(
        (Gaussian(5), Morphology("close", 3, 2), Gray(), AdaptiveMean(11, -2.0, 255, True)),
        name="crossover_401_style",
    )
    cpu_executor = CpuPreprocessExecutor()
    measurements = []
    rng = np.random.default_rng(20260717)
    for size in sorted({int(value) for value in sizes if int(value) > 0}):
        image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        cpu = _timing_summary(
            lambda image=image: cpu_executor.execute(image, plan), repetitions, warmup
        )
        gpu = _timing_summary(
            lambda image=image: runtime.execute_plan(image, plan), repetitions, warmup
        )
        speedup = cpu["median_ms"] / gpu["median_ms"] if gpu["median_ms"] > 0 else None
        measurements.append({
            "size": [size, size],
            "pixels": size * size,
            "cpu": cpu,
            "gpu_including_transfer": gpu,
            "speedup": round(speedup, 3) if speedup is not None else None,
        })

    def stable_threshold(target: float):
        for index, item in enumerate(measurements):
            suffix = [entry["speedup"] for entry in measurements[index:]]
            if suffix and all(value is not None and value >= target for value in suffix):
                return item["pixels"]
        return None

    result = {
        "repetitions": max(1, int(repetitions)),
        "warmup": max(0, int(warmup)),
        "measurements": measurements,
        "observed_stable_crossover_pixels": stable_threshold(1.0),
        "observed_stable_1_5x_pixels": stable_threshold(1.5),
        "policy_changed": False,
        "note": "Observed thresholds are evidence only; production routing remains unchanged until RTX acceptance.",
    }
    print(f"CROSSOVER {result}")
    return result


def benchmark_morphology_iterations(
    runtime: GpuRuntime,
    repetitions: int = 20,
    warmup: int = 5,
    iterations=(1, 2, 4, 8),
    size: int = 1024,
) -> dict:
    if not runtime.supports_native_plan:
        raise AssertionError("Morphology profiling requires the generic native plan ABI")
    binary = np.random.default_rng(20260717).integers(
        0, 2, size=(int(size), int(size)), dtype=np.uint8
    ) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    measurements = []
    for count in sorted({int(value) for value in iterations if int(value) > 0}):
        plan = PreprocessPlan(
            (Morphology("close", 3, count),), name=f"morphology_close_i{count}"
        )
        cpu = _timing_summary(
            lambda count=count: cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE, kernel, iterations=count
            ),
            repetitions,
            warmup,
        )
        gpu = _timing_summary(
            lambda plan=plan: runtime.execute_plan(binary, plan), repetitions, warmup
        )
        native = runtime.performance_stats().get("native_timings_ms") or {}
        morphology_ms = native.get("morphology_ms")
        kernel_ms = native.get("kernel_ms")
        share = (
            float(morphology_ms) / float(kernel_ms)
            if isinstance(morphology_ms, (int, float)) and isinstance(kernel_ms, (int, float))
            and kernel_ms > 0 else None
        )
        measurements.append({
            "iterations": count,
            "passes": count * 2,
            "cpu": cpu,
            "gpu_including_transfer": gpu,
            "speedup": round(cpu["median_ms"] / gpu["median_ms"], 3)
            if gpu["median_ms"] > 0 else None,
            "native_timings_ms": native,
            "morphology_kernel_share": round(share, 6) if share is not None else None,
        })
    result = {
        "image_shape": list(binary.shape),
        "kernel_size": 3,
        "operation": "close",
        "repetitions": max(1, int(repetitions)),
        "warmup": max(0, int(warmup)),
        "measurements": measurements,
        "optimization_selected": False,
        "note": "Separable morphology remains gated on RTX correctness and measured benefit.",
    }
    print(f"MORPHOLOGY_PROFILE {result}")
    return result


def normalized_result(result: dict) -> dict:
    normalized = deepcopy(result)
    for key in ("duration_sec", "outputs", "execution"):
        normalized.pop(key, None)
    provenance = normalized.get("provenance")
    if isinstance(provenance, dict):
        # Validation recipes intentionally differ in CPU/GPU routing fields.
        provenance.pop("recipe_source_sha256", None)
        provenance.pop("effective_recipe_sha256", None)
    for tile_result in normalized.get("tiles", []):
        for detector_result in tile_result.get("detectors", []):
            detector_result.pop("execution", None)
    return normalized


def load_production_manifest(path: Path) -> list[dict]:
    manifest_path = Path(path).resolve()
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise AssertionError("Production manifest must be a mapping with schema_version: 1")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise AssertionError("Production manifest cases must be a non-empty list")
    cases = []
    coverage = {name: set() for name in PRODUCTION_RECIPES}
    identifiers = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise AssertionError(f"Production case {index} must be a mapping")
        identifier = str(raw.get("id", "")).strip()
        recipe_value = str(raw.get("recipe", "")).strip()
        image_value = str(raw.get("image", "")).strip()
        expected = str(raw.get("expected_final", "")).upper()
        if not identifier or identifier in identifiers:
            raise AssertionError(f"Production case id is missing or duplicated: {identifier!r}")
        if expected not in {"PASS", "NG"}:
            raise AssertionError(f"Production case {identifier} expected_final must be PASS or NG")
        recipe = (manifest_path.parent / recipe_value).resolve()
        image = (manifest_path.parent / image_value).resolve()
        if recipe.name not in coverage:
            raise AssertionError(f"Production case {identifier} uses unknown recipe: {recipe.name}")
        if not recipe.is_file() or not image.is_file():
            raise AssertionError(
                f"Production case {identifier} file is missing: recipe={recipe}, image={image}"
            )
        identifiers.add(identifier)
        coverage[recipe.name].add(expected)
        cases.append({
            "id": identifier,
            "recipe": recipe,
            "image": image,
            "expected_final": expected,
        })
    incomplete = {
        recipe: sorted({"PASS", "NG"} - labels)
        for recipe, labels in coverage.items()
        if labels != {"PASS", "NG"}
    }
    if incomplete:
        raise AssertionError(f"Production manifest requires PASS and NG for every recipe: {incomplete}")
    return cases


def validate_pipeline(image_path: Path, recipe_path: Path, dll_path: str) -> dict:
    manager = RecipeManager()
    base = manager.load(recipe_path)
    cpu_recipe = deepcopy(base)
    gpu_recipe = deepcopy(base)
    cpu_recipe["gpu"] = {
        "tiling": False,
        "display": False,
        "dll_path": dll_path,
        "fallback_to_cpu": False,
    }
    gpu_recipe["gpu"] = {
        "tiling": True,
        "display": True,
        "dll_path": dll_path,
        "fallback_to_cpu": False,
    }
    for config in cpu_recipe.get("detectors", {}).values():
        config["use_gpu"] = False
    for config in gpu_recipe.get("detectors", {}).values():
        config["use_gpu"] = bool(config.get("enabled", False))
    for recipe in (cpu_recipe, gpu_recipe):
        recipe["output"] = _disabled_report_output(recipe.get("output", {}))

    with tempfile.TemporaryDirectory(prefix="visionflow_cuda_validation_") as temporary:
        temporary_path = Path(temporary)
        cpu_path = temporary_path / "cpu.yaml"
        gpu_path = temporary_path / "gpu.yaml"
        cpu_path.write_text(yaml.safe_dump(cpu_recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
        gpu_path.write_text(yaml.safe_dump(gpu_recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
        cpu_result = AOIPipeline(cpu_path, temporary_path / "cpu_outputs").run(image_path)
        gpu_result = AOIPipeline(gpu_path, temporary_path / "gpu_outputs").run(image_path)

    active = gpu_result.get("execution", {}).get("gpu", {})
    if not active.get("tiling", {}).get("active"):
        raise AssertionError(f"GPU tiling did not activate: {active}")
    inactive_detectors = {
        detector_id: status
        for detector_id, status in active.get("detectors", {}).items()
        if status.get("requested") and not status.get("active")
    }
    if inactive_detectors:
        raise AssertionError(f"GPU detectors did not activate: {inactive_detectors}")

    cpu_normalized = normalized_result(cpu_result)
    gpu_normalized = normalized_result(gpu_result)
    if cpu_normalized != gpu_normalized:
        summary = {
            "cpu_final": cpu_result.get("final_result"),
            "gpu_final": gpu_result.get("final_result"),
            "cpu_summary": cpu_result.get("summary"),
            "gpu_summary": gpu_result.get("summary"),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise AssertionError("Full pipeline CPU/GPU results differ; inspect the printed summaries and report JSON")
    print("PASS full_pipeline: CPU and GPU inspection results are identical")
    return {
        "image": str(image_path.resolve()),
        "recipe": str(recipe_path.resolve()),
        "final_result": cpu_result.get("final_result"),
        "summary": cpu_result.get("summary"),
        "gpu": active,
    }


def validate_production_manifest(path: Path, dll_path: str) -> list[dict]:
    results = []
    for case in load_production_manifest(path):
        result = validate_pipeline(case["image"], case["recipe"], dll_path)
        if result["final_result"] != case["expected_final"]:
            raise AssertionError(
                f"Production case {case['id']} expected {case['expected_final']}, "
                f"got {result['final_result']}"
            )
        result["id"] = case["id"]
        result["expected_final"] = case["expected_final"]
        results.append(result)
    print(f"PASS production manifest: {len(results)} CPU/GPU-equivalent labeled cases")
    return results


RESIZE_AREA_RECIPE = "PRODUCT_A_CIRCLE_401_1_AOI_01.yaml"
RESIZE_AREA_SCALES = (1.0, 0.5, 0.37, 0.25)


def synthetic_circle_image(seed: int, circles: int) -> np.ndarray:
    """1300x1200 BGR background with optional bright circles sized for 401-CS-AP-1 filters."""
    rng = np.random.default_rng(seed)
    gray = cv2.GaussianBlur(rng.integers(90, 120, (1300, 1200), dtype=np.uint8), (0, 0), 3)
    for _ in range(circles):
        center = (int(rng.integers(160, 1040)), int(rng.integers(160, 1140)))
        cv2.circle(gray, center, int(rng.integers(9, 17)), int(rng.integers(190, 240)), -1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def validate_resize_area_recipe_sweep(dll_path: str) -> list[dict]:
    """Detector end-to-end CPU/GPU equivalence whose native plan exercises every Resize(area) branch."""
    recipe_path = ROOT / "recipes" / RESIZE_AREA_RECIPE
    base = RecipeManager().load(recipe_path)
    results = []
    with tempfile.TemporaryDirectory(prefix="visionflow_resize_area_") as temporary:
        folder = Path(temporary)
        images = {}
        for label, circles in (("pass", 0), ("ng", 12)):
            image_path = folder / f"{label}.png"
            if not cv2.imwrite(str(image_path), synthetic_circle_image(20260914 + circles, circles)):
                raise AssertionError(f"Failed to write {image_path}")
            images[label] = image_path
        for scale in RESIZE_AREA_SCALES:
            recipe = deepcopy(base)
            recipe["detectors"]["401-CS-AP-1"]["params"]["process_scale"] = scale
            scaled_recipe = folder / f"circle_scale_{scale}.yaml"
            scaled_recipe.write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
            for label, image_path in images.items():
                result = validate_pipeline(image_path, scaled_recipe, dll_path)
                routes = {
                    detector_id: status.get("fallback_reason", "")
                    for detector_id, status in result["gpu"].get("detectors", {}).items()
                }
                if any(routes.values()):
                    raise AssertionError(f"Resize(area) sweep fell back to CPU: {routes}")
                results.append({
                    "recipe": RESIZE_AREA_RECIPE, "process_scale": scale, "image": label,
                    "final_result": result["final_result"], "summary": result["summary"],
                })
    if not any(item["final_result"] == "NG" for item in results):
        raise AssertionError("Resize(area) sweep never produced an NG decision")
    print(f"PASS resize_area_pipeline: {json.dumps(results, ensure_ascii=False)}")
    return results


def _disabled_report_output(output: dict) -> dict:
    return {
        key: False if isinstance(value, bool) else value
        for key, value in output.items()
    }


def main() -> int:
    args = parse_args()
    runtime = GpuRuntime(args.dll, fallback_to_cpu=False)
    if not runtime.available:
        raise SystemExit(f"CUDA DLL unavailable: {runtime.unavailable_reason}")
    print(
        f"CUDA DLL ready: device={runtime.device_name}, capability={runtime.compute_capability}, "
        f"path={runtime.dll_path}"
    )
    validation = validate_primitives(runtime)
    benchmark_result = benchmark(runtime, args.benchmark, args.warmup)
    crossover_result = benchmark_crossover(runtime, args.benchmark, args.warmup) if args.crossover else {}
    morphology_result = (
        benchmark_morphology_iterations(runtime, args.benchmark, args.warmup)
        if args.morphology_profile else {}
    )
    stress_result = stress_persistent_plan(runtime, args.stress, args.warmup)
    production_results = (
        validate_production_manifest(Path(args.production_manifest), str(runtime.dll_path))
        if args.production_manifest else []
    )
    if args.image and args.recipe:
        validate_pipeline(Path(args.image), Path(args.recipe), str(runtime.dll_path))
    roi_batch_matrix = validate_roi_batch_matrix(runtime) if args.roi_batch_matrix else {}
    resize_area_pipeline = (
        validate_resize_area_recipe_sweep(str(runtime.dll_path)) if args.resize_area_pipeline else []
    )
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "commit": os.environ.get("GITHUB_SHA", ""),
                    "environment": environment_snapshot(args.image, args.recipe),
                    "device": runtime.device_name,
                    "compute_capability": runtime.compute_capability,
                    "validation": validation,
                    "benchmark": benchmark_result,
                    "crossover": crossover_result,
                    "morphology_profile": morphology_result,
                    "stress": stress_result,
                    "production": production_results,
                    "resize_area_pipeline": resize_area_pipeline,
                    "roi_batch_matrix": roi_batch_matrix,
                    "gpu_metrics": runtime.performance_stats(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print("All requested CUDA validations passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
