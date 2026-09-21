from __future__ import annotations

import json
import unittest
from pathlib import Path

from gpu.preflight_cuda_build import (
    OPTIONAL_GENERIC_PLAN_EXPORTS,
    OPTIONAL_RESIDENT_ROI_EXPORTS,
    OPTIONAL_ROI_BATCH_EXPORTS,
    OPTIONAL_TIMING_EXPORTS,
    REQUIRED_ABI_V1_EXPORTS,
    inspect_contract,
)


class CudaSourceContractTests(unittest.TestCase):
    def test_header_source_runtime_smoke_and_build_manifest_are_synchronized(self):
        result = inspect_contract()

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["abi_version"], 1)
        self.assertTrue(REQUIRED_ABI_V1_EXPORTS.issubset(result["exports"]))
        self.assertEqual(set(result["optional_generic_plan_exports"]), OPTIONAL_GENERIC_PLAN_EXPORTS)
        self.assertEqual(set(result["optional_resident_roi_exports"]), OPTIONAL_RESIDENT_ROI_EXPORTS)
        self.assertEqual(set(result["optional_roi_batch_exports"]), OPTIONAL_ROI_BATCH_EXPORTS)
        self.assertEqual(set(result["optional_timing_exports"]), OPTIONAL_TIMING_EXPORTS)
        self.assertEqual(result["dll_sources"], ["gpu/visionflow_cuda.cu"])
        self.assertEqual(result["smoke_sources"], ["gpu/test_cuda_api.cu"])

    def test_generic_plan_execute_has_one_upload_one_download_and_no_allocation(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        execute = source.split("VF_CUDA_API int vf_plan_execute(", 1)[1].split(
            "VF_CUDA_API int vf_plan_destroy(", 1
        )[0]
        device_execute = source.split("static int execute_linear_plan_device(", 1)[1].split(
            "static int execute_dag_plan_device(", 1
        )[0]
        header = (root / "gpu" / "include" / "visionflow_cuda.h").read_text(encoding="utf-8")
        descriptor = header.split("typedef struct VfPlanOperatorV1", 1)[1].split(
            "VF_CUDA_API int vf_gpu_abi_version", 1
        )[0]

        self.assertEqual(execute.count("cudaMemcpyHostToDevice"), 1)
        self.assertEqual(device_execute.count("cudaMemcpyDeviceToHost"), 1)
        self.assertEqual(execute.count("cudaStreamSynchronize"), 0)
        self.assertEqual(device_execute.count("stream_result"), 1)
        self.assertNotIn("cudaMalloc", execute)
        self.assertNotIn("reserve_plan_buffers", execute)
        self.assertIn("context->stream", execute)
        self.assertIn("context->u8[4]", device_execute)
        self.assertNotIn("detector", descriptor.lower())

    def test_linear_native_plan_resizes_on_device_and_tracks_output_shape(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        header = (root / "gpu" / "include" / "visionflow_cuda.h").read_text(encoding="utf-8")
        execute = source.split("static int execute_linear_plan_device(", 1)[1].split(
            "static int execute_dag_plan_device(", 1
        )[0]
        validation = source.split("int validate_plan_desc(", 1)[1].split(
            "int validate_dag_plan_desc(", 1
        )[0]

        self.assertIn("VF_PLAN_RESIZE_AREA = 6", header)
        self.assertIn("case VF_PLAN_RESIZE_AREA", validation)
        self.assertIn("target_width", execute)
        self.assertIn("launch_area_resize(", execute)
        self.assertNotIn("resize_gray_kernel<<<", execute)
        self.assertIn("compiled->output_width", source)
        self.assertIn("compiled->output_height", source)

    def test_area_resize_mirrors_opencv_tables_and_disables_fused_multiply_add(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        manifest = json.loads((root / "gpu" / "cuda_project.json").read_text(encoding="utf-8"))
        build = (root / "gpu" / "build_cuda_dll.ps1").read_text(encoding="utf-8")
        axis = source.split("void append_area_axis(", 1)[1].split("int prepare_area_resize(", 1)[0]
        prepare = source.split("int prepare_area_resize(", 1)[1].split("void launch_area_resize(", 1)[0]
        kernel = source.split("__global__ void resize_area_kernel(", 1)[1].split(
            "__global__ void resize_gray_kernel(", 1
        )[0]
        create = source.split("VF_CUDA_API int vf_plan_create(", 1)[1].split(
            "VF_CUDA_API int vf_plan_execute(", 1
        )[0]

        self.assertIs(manifest["nvcc"]["fmad"], False)
        self.assertEqual(build.count('"--fmad=$fmad"'), 2)
        self.assertIn("start - first > 1e-3", axis)
        self.assertIn("std::min(std::min(last - end, 1.0), cell_width) / cell_width", axis)
        self.assertIn("1.0 / (static_cast<double>(target_width) / source_width)", prepare)
        self.assertIn("AREA_RESIZE_FAST_2X2", prepare)
        self.assertIn("+ 2) >> 2", kernel)
        self.assertIn("nearbyintf(total)", kernel)
        self.assertIn("prepare_area_resize(", create)

    def test_persistent_context_owns_stream_and_fused_path_uses_it(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        context = source.split("struct PersistentContext", 1)[1].split("struct NativePlan", 1)[0]
        fused = source.split("VF_CUDA_API int vf_preprocess_401_2_u8(", 1)[1].split(
            "VF_CUDA_API int vf_morphology_rect_u8(", 1
        )[0]

        self.assertIn("cudaStreamCreateWithFlags", context)
        self.assertIn("cudaStreamDestroy", context)
        self.assertIn("persistent->stream", fused)
        self.assertEqual(fused.count("cudaMemcpy2DAsync("), 2)

    def test_gray_and_gaussian_match_opencv_fixed_point_contract(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        weights = source.split("int prepare_gaussian_weights(", 1)[1].split(
            "int adaptive_layout(", 1
        )[0]
        gray = source.split("__global__ void bgr_gray_kernel(", 1)[1].split(
            "__global__ void bgr_rgb_kernel(", 1
        )[0]

        self.assertIn("weights = {64, 128, 64}", weights)
        self.assertIn("weights = {16, 64, 96, 64, 16}", weights)
        self.assertIn("weights = {8, 28, 56, 72, 56, 28, 8}", weights)
        self.assertIn("weights = {4, 13, 30, 51, 60, 51, 30, 13, 4}", weights)
        self.assertIn("std::nearbyint(adjusted)", weights)
        self.assertIn("GAUSSIAN_FIXED_SCALE - side_sum * 2", weights)
        self.assertIn("__constant__ uint16_t gaussian_weights", source)
        self.assertIn("GAUSSIAN_FINAL_ROUND", source)
        self.assertIn("gray_shift = 15", gray)
        self.assertIn("blue_to_gray = 3735", gray)
        self.assertIn("green_to_gray = 19235", gray)
        self.assertIn("red_to_gray = 9798", gray)

    def test_gaussian_interior_fast_path_keeps_reflect101_borders(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        horizontal = source.split("__global__ void gaussian_horizontal_kernel(", 1)[1].split(
            "__global__ void gaussian_vertical_kernel(", 1
        )[0]
        vertical = source.split("__global__ void gaussian_vertical_kernel(", 1)[1].split(
            "__global__ void threshold_kernel(", 1
        )[0]
        launcher = source.split("void launch_gaussian(", 1)[1].split("__global__ void gather_roi_batch_kernel(", 1)[0]

        self.assertIn("x >= radius && x + radius < width", horizontal)
        self.assertIn("reflect101(x + kx, width)", horizontal)
        self.assertIn("y >= radius && y + radius < height", vertical)
        self.assertIn("reflect101(y + ky, height)", vertical)
        self.assertNotIn("__shared__", horizontal + vertical + launcher)
        self.assertEqual(source.count("launch_gaussian("), 4)

    def test_native_dag_uploads_root_once_and_downloads_requested_outputs(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        execute = source.split("VF_CUDA_API int vf_dag_plan_execute(", 1)[1].split(
            "VF_CUDA_API int vf_dag_plan_destroy(", 1
        )[0]
        device_execute = source.split("static int execute_dag_plan_device(", 1)[1].split(
            "VF_CUDA_API int vf_gpu_abi_version", 1
        )[0]

        self.assertEqual(execute.count("cudaMemcpyHostToDevice"), 1)
        self.assertEqual(device_execute.count("cudaMemcpyDeviceToHost"), 1)
        self.assertIn("for (int index = 0; index < output_count; ++index)", device_execute)
        self.assertEqual(device_execute.count("stream_result"), 1)
        self.assertNotIn("cudaMalloc", execute)
        self.assertIn("values[op.input_node]", device_execute)

    def test_resident_roi_execution_uses_device_copy_without_host_upload(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        linear = source.split("VF_CUDA_API int vf_plan_execute_roi(", 1)[1].split(
            "VF_CUDA_API int vf_dag_plan_query(", 1
        )[0]
        dag = source.split("VF_CUDA_API int vf_dag_plan_execute_roi(", 1)[1].split(
            "VF_CUDA_API int vf_bgr_to_gray_u8(", 1
        )[0]

        for execute in (linear, dag):
            self.assertIn("cudaMemcpyDeviceToDevice", execute)
            self.assertNotIn("cudaMemcpyHostToDevice", execute)
            self.assertNotIn("cudaMalloc", execute)

    def test_roi_batch_uses_one_coordinate_array_and_contiguous_device_buffer(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        create = source.split("VF_CUDA_API int vf_roi_batch_create(", 1)[1].split(
            "VF_CUDA_API int vf_roi_batch_info(", 1
        )[0]

        self.assertIn("gather_roi_batch_kernel<<<", create)
        self.assertIn("created->device_rois", create)
        self.assertIn("created->data", create)
        self.assertEqual(create.count("cudaMemcpyAsync("), 1)
        self.assertNotIn("cudaMemcpyDeviceToHost", create)

    def test_persistent_plan_records_cuda_event_phase_timings(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        header = (root / "gpu" / "include" / "visionflow_cuda.h").read_text(encoding="utf-8")

        self.assertIn("typedef struct VfCudaTimingsV1", header)
        self.assertIn("vf_context_set_timing_enabled", header)
        self.assertIn("vf_context_last_timings", header)
        self.assertIn("typedef struct VfCudaContextMemoryStatsV1", header)
        self.assertIn("vf_context_memory_stats_v1", header)
        self.assertIn("typedef struct VfCudaContextMemoryStatsV2", header)
        self.assertIn("vf_context_memory_stats_v2", header)
        self.assertIn("vf_context_trim_analysis_scratch", header)
        self.assertIn("context_memory_breakdown", source)
        self.assertIn("cnr_candidate_bytes", source)
        self.assertIn("record_timing_event", source)
        self.assertIn("if (!context->timing_enabled) return", source)
        self.assertIn("cudaEventElapsedTime", source)
        self.assertIn("TIMING_GAUSSIAN_START", source)
        self.assertIn("TIMING_ADAPTIVE_START", source)
        self.assertIn("TIMING_THRESHOLD_START", source)
        self.assertIn("TIMING_MORPHOLOGY_START", source)

    def test_analysis_scratch_trim_preserves_externally_referenced_lifetimes(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        smoke = (root / "gpu" / "test_cuda_api.cu").read_text(encoding="utf-8")
        trim = self._function_body(
            source, "VF_CUDA_API int vf_context_trim_analysis_scratch("
        )

        self.assertIn("stream_result(persistent->stream)", trim)
        for family in ("median_", "gaussian_f32_", "cnr_mask_", "cand_"):
            self.assertIn(f"&persistent->{family}", trim)
        for protected in (
            "resident_data", "plan_", "match_", "contour_", "find_contours_"
        ):
            self.assertNotIn(f"&persistent->{protected}", trim)
        self.assertIn("peak_memory_bytes", source)
        self.assertIn("vf_context_memory_stats_v2(context", smoke)
        self.assertIn("vf_context_trim_analysis_scratch(context", smoke)
        self.assertIn("median_after_trim", smoke)

    def test_grow_only_reserve_keeps_previous_pointer_when_allocation_fails(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        reserve = source.split("int reserve_device(", 1)[1].split(
            "int prepare_gaussian_weights", 1
        )[0]

        self.assertLess(reserve.index("cudaMalloc(&replacement"), reserve.index("free_device(*pointer)"))
        self.assertLess(reserve.index("if (error != cudaSuccess) return"), reserve.index("free_device(*pointer)"))

    def test_reported_runtime_failure_consumes_stale_last_error_and_smoke_covers_oom_recovery(self):
        root = Path(__file__).resolve().parents[1]
        internal = (root / "gpu" / "include" / "visionflow_cuda_internal.cuh").read_text(encoding="utf-8")
        runtime_error = internal.split("inline int runtime_error(", 1)[1].split("inline bool valid_image(", 1)[0]
        smoke = (root / "gpu" / "test_cuda_api.cu").read_text(encoding="utf-8")

        self.assertLess(runtime_error.index("cudaSuccess) return VF_CUDA_OK"), runtime_error.index("cudaGetLastError()"))
        self.assertLess(runtime_error.index("cudaGetLastError()"), runtime_error.index("VF_CUDA_RUNTIME_ERROR_BASE"))
        self.assertIn("oom_recovery_result = vf_roi_batch_create(", smoke)
        self.assertIn("std::vector<VfRoiV1> oom_rois(65535", smoke)

    @staticmethod
    def _function_body(source: str, signature: str) -> str:
        """Text of one top-level C++ function: from its signature to the first closing brace line."""
        body = source.split(signature, 1)[1]
        return body[: body.index("\n}\n") + 3]

    def test_resident_cnr_export_has_no_host_input_copy_and_is_in_native_smoke(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        smoke = (root / "gpu" / "test_cuda_api.cu").read_text(encoding="utf-8")
        chain = self._function_body(source, "int resident_cnr_mask_device(")
        execute = self._function_body(source, "VF_CUDA_API int vf_cnr_mask_u8_roi(")

        self.assertIn("persistent->resident_u8", chain)
        self.assertIn("resident_gray_f32_kernel<<<", chain)
        self.assertNotIn("cudaMemcpyHostToDevice", chain + execute)
        self.assertNotIn("cudaMemcpyDeviceToHost", chain)
        self.assertIn("resident_cnr_mask_device(", execute)
        self.assertEqual(execute.count("cudaMemcpyDeviceToHost"), 1)
        self.assertIn("vf_cnr_mask_u8_roi(", smoke)

    def test_resident_cnr_candidates_download_only_records_and_counts(self):
        """The candidate export must never upload pixels or download the gray image, mask or labels."""
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        header = (root / "gpu" / "include" / "visionflow_cuda.h").read_text(encoding="utf-8")
        smoke = (root / "gpu" / "test_cuda_api.cu").read_text(encoding="utf-8")
        execute = self._function_body(source, "VF_CUDA_API int vf_cnr_candidates_u8_roi(")
        grouping = self._function_body(source, "int cand_group_components(")
        read_word = self._function_body(source, "int cand_read_word(")
        device_side = execute + grouping

        self.assertIn("resident_cnr_mask_device(", execute)
        self.assertNotIn("cudaMemcpyHostToDevice", device_side + read_word)
        # Only int32 count words, two gather-size scalars and the candidate records cross PCIe.
        self.assertEqual(read_word.count("cudaMemcpyDeviceToHost"), 1)
        self.assertIn("sizeof(int32_t)", read_word)
        downloads = device_side.count("cudaMemcpyDeviceToHost")
        self.assertEqual(downloads, 4)
        self.assertIn("out_candidate_ints, persistent->cand_out_ints", execute)
        self.assertIn("out_candidate_floats, persistent->cand_out_floats", execute)
        for forbidden in ("cnr_mask_image, ", "ccl_parent, sizeof", "cnr_mask_mask, sizeof"):
            self.assertNotIn(forbidden, device_side)
        # The parallel ring statistics only read back long long totals, never pixels or values.
        ring = self._function_body(source, "int cand_ring_statistics(")
        read_total = self._function_body(source, "int cand_read_total(")
        self.assertNotIn("cudaMemcpyHostToDevice", ring + read_total)
        self.assertNotIn("cudaMemcpyDeviceToHost", ring)
        self.assertEqual(read_total.count("sizeof(long long)"), 2)
        self.assertIn("cand_ring_statistics(", execute)
        # Ring statistics follow NumPy's float32 pairwise order and float64 division.
        split = self._function_body(source, "__device__ __forceinline__ long long cand_split(")
        self.assertIn("half - half % 8", split)
        leaf = self._function_body(source, "__device__ float cand_pairwise_leaf(")
        self.assertIn("((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))", leaf)
        self.assertIn("float result = -0.0f;", leaf)
        combine = self._function_body(source, "__device__ float cand_combine_leaves(")
        self.assertIn("count <= CAND_LEAF_SIZE", combine)
        self.assertIn("left + right", combine)
        means = self._function_body(source, "__global__ void cand_combine_kernel(")
        self.assertIn("static_cast<double>(total) / static_cast<double>(n)", means)
        self.assertIn("sqrtf(static_cast<float>(static_cast<double>(squares) / static_cast<double>(n)))", means)
        self.assertIn("constexpr long long CAND_LEAF_SIZE = 128;", source)
        self.assertIn("VF_CUDA_API int vf_cnr_candidates_u8_roi(", header)
        self.assertIn("vf_cnr_candidates_u8_roi(", smoke)


if __name__ == "__main__":
    unittest.main()
