from __future__ import annotations

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
        self.assertIn("resize_gray_kernel<<<", execute)
        self.assertIn("compiled->output_width", source)
        self.assertIn("compiled->output_height", source)

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
        self.assertIn("vf_context_last_timings", header)
        self.assertIn("cudaEventRecord", source)
        self.assertIn("cudaEventElapsedTime", source)
        self.assertIn("TIMING_GAUSSIAN_START", source)
        self.assertIn("TIMING_ADAPTIVE_START", source)
        self.assertIn("TIMING_THRESHOLD_START", source)
        self.assertIn("TIMING_MORPHOLOGY_START", source)

    def test_grow_only_reserve_keeps_previous_pointer_when_allocation_fails(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        reserve = source.split("int reserve_device(", 1)[1].split(
            "int prepare_gaussian_weights", 1
        )[0]

        self.assertLess(reserve.index("cudaMalloc(&replacement"), reserve.index("free_device(*pointer)"))
        self.assertLess(reserve.index("if (error != cudaSuccess) return"), reserve.index("free_device(*pointer)"))


if __name__ == "__main__":
    unittest.main()
