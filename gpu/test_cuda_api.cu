#include <iostream>
#include <vector>
#include "visionflow_cuda.h"

int main() {
    std::cout << "VisionFlow CUDA ABI: " << vf_gpu_abi_version() << "\n";
    if (vf_gpu_abi_version() != VF_CUDA_ABI_VERSION) {
        std::cerr << "ABI mismatch\n";
        return 2;
    }

    int count = vf_gpu_device_count();
    std::cout << "CUDA device count: " << count << "\n";
    if (count <= 0) {
        std::cerr << "No CUDA device\n";
        return 3;
    }

    char name[256]{};
    int result = vf_gpu_device_name(name, static_cast<int>(sizeof(name)));
    if (result != VF_CUDA_OK) {
        char message[256]{};
        vf_gpu_error_message(result, message, static_cast<int>(sizeof(message)));
        std::cerr << "Device query failed: " << message << "\n";
        return 4;
    }

    int capability = vf_gpu_compute_capability();
    std::cout << "Device: " << name << "\n";
    std::cout << "Compute capability: " << capability / 10 << "." << capability % 10 << "\n";

    const int width = 8;
    const int height = 8;
    std::vector<uint8_t> bgr(width * height * 3, 128);
    std::vector<uint8_t> gray(width * height, 0);
    result = vf_bgr_to_gray_u8(
        bgr.data(), width, height, width * 3, 3,
        gray.data(), width, 1);
    if (result != VF_CUDA_OK) {
        char message[256]{};
        vf_gpu_error_message(result, message, static_cast<int>(sizeof(message)));
        std::cerr << "Grayscale smoke failed: " << message << "\n";
        return 5;
    }

    void* context = nullptr;
    result = vf_context_create(&context);
    if (result != VF_CUDA_OK || context == nullptr) {
        std::cerr << "Persistent context creation failed\n";
        return 6;
    }
    result = vf_context_set_timing_enabled(context, 0);
    if (result != VF_CUDA_OK) {
        std::cerr << "Persistent context timing disable failed\n";
        return 16;
    }
    result = vf_context_set_timing_enabled(context, 1);
    if (result != VF_CUDA_OK) {
        std::cerr << "Persistent context timing enable failed\n";
        return 17;
    }
    std::vector<uint8_t> fused_binary(width * height, 0);
    result = vf_preprocess_401_2_u8(
        context,
        bgr.data(), width, height, width * 3, 3,
        fused_binary.data(), width,
        3, 3, -2.0f, 255, 1);
    uint64_t reserved_bytes = 0;
    uint64_t allocation_count = 0;
    int stats_result = vf_context_stats(context, &reserved_bytes, &allocation_count);
    VfCudaContextMemoryStatsV1 memory_stats{};
    memory_stats.struct_size = sizeof(VfCudaContextMemoryStatsV1);
    memory_stats.version = 1;
    int memory_stats_result = vf_context_memory_stats_v1(context, &memory_stats);
    if (result != VF_CUDA_OK || stats_result != VF_CUDA_OK ||
        memory_stats_result != VF_CUDA_OK || reserved_bytes == 0 || allocation_count == 0 ||
        memory_stats.reserved_bytes != reserved_bytes ||
        memory_stats.peak_reserved_bytes < memory_stats.reserved_bytes ||
        memory_stats.allocation_count != allocation_count ||
        memory_stats.plan_bytes + memory_stats.resident_bytes +
                memory_stats.template_match_bytes + memory_stats.contour_bytes +
                memory_stats.median_bytes + memory_stats.gaussian_f32_bytes +
                memory_stats.cnr_mask_bytes + memory_stats.cnr_candidate_bytes !=
            memory_stats.reserved_bytes) {
        char message[256]{};
        int failed = result != VF_CUDA_OK ? result :
            (stats_result != VF_CUDA_OK ? stats_result : memory_stats_result);
        vf_gpu_error_message(failed, message, static_cast<int>(sizeof(message)));
        std::cerr << "Fused 401-2 smoke failed: " << message << "\n";
        return 7;
    }

    const int resized_width = 4;
    const int resized_height = 4;
    VfPlanOperatorV1 operators[4]{};
    operators[0].struct_size = sizeof(VfPlanOperatorV1);
    operators[0].kind = VF_PLAN_GRAY;
    operators[0].input_node = VF_PLAN_INPUT_NODE;
    operators[0].output_node = 0;
    operators[1].struct_size = sizeof(VfPlanOperatorV1);
    operators[1].kind = VF_PLAN_RESIZE_AREA;
    operators[1].input_node = 0;
    operators[1].output_node = 1;
    operators[1].int_params[0] = resized_width;
    operators[1].int_params[1] = resized_height;
    operators[2].struct_size = sizeof(VfPlanOperatorV1);
    operators[2].kind = VF_PLAN_GAUSSIAN;
    operators[2].input_node = 1;
    operators[2].output_node = 2;
    operators[2].int_params[0] = 3;
    operators[3].struct_size = sizeof(VfPlanOperatorV1);
    operators[3].kind = VF_PLAN_ADAPTIVE_MEAN;
    operators[3].input_node = 2;
    operators[3].output_node = 3;
    operators[3].int_params[0] = 3;
    operators[3].int_params[1] = 255;
    operators[3].int_params[2] = 1;
    operators[3].float_params[0] = -2.0f;
    VfPlanDescV1 descriptor{};
    descriptor.struct_size = sizeof(VfPlanDescV1);
    descriptor.version = VF_CUDA_PLAN_VERSION;
    descriptor.input_channels = 3;
    descriptor.operator_count = 4;
    descriptor.operators = operators;
    descriptor.output_node = 3;
    char plan_reason[256]{};
    result = vf_plan_query(&descriptor, width, height, plan_reason, sizeof(plan_reason));
    void* plan = nullptr;
    if (result == VF_CUDA_OK) result = vf_plan_create(context, &descriptor, width, height, &plan);
    std::vector<uint8_t> plan_binary(resized_width * resized_height, 0);
    if (result == VF_CUDA_OK) {
        result = vf_plan_execute(
            plan, bgr.data(), width, height, width * 3, 3,
            plan_binary.data(), resized_width, 1);
    }
    uint64_t resident_generation = 0;
    if (result == VF_CUDA_OK) {
        result = vf_context_upload_u8(
            context, bgr.data(), width, height, width * 3, 3, &resident_generation);
    }
    if (result == VF_CUDA_OK) {
        const int row_bytes = width * 3;
        std::vector<uint8_t> bottom_up(bgr.size(), 0);
        for (int row = 0; row < height; ++row) {
            std::memcpy(
                bottom_up.data() + static_cast<size_t>(height - 1 - row) * row_bytes,
                bgr.data() + static_cast<size_t>(row) * row_bytes,
                static_cast<size_t>(row_bytes));
        }
        uint64_t bottom_up_generation = 0;
        result = vf_host_register_u8(context, bottom_up.data(), bottom_up.size());
        if (result == VF_CUDA_OK) {
            result = vf_context_upload_u8_file_order(
                context,
                bottom_up.data() + static_cast<size_t>(height - 1) * row_bytes,
                width, height, -row_bytes, 3, &bottom_up_generation);
        }
        int unregister_result = vf_host_unregister_u8(context, bottom_up.data());
        if (result == VF_CUDA_OK && unregister_result != VF_CUDA_OK) {
            std::cerr << "pinned host buffer unregister failed\n";
            return 11;
        }
        VfRoiV1 full_roi{sizeof(VfRoiV1), 0, 0, width, height};
        void* full_batch = nullptr;
        if (result == VF_CUDA_OK) {
            result = vf_roi_batch_create(
                context, bottom_up_generation, &full_roi, 1, &full_batch);
        }
        std::vector<uint8_t> bottom_up_download(bgr.size(), 0);
        if (result == VF_CUDA_OK) {
            result = vf_roi_batch_download_u8(
                full_batch, 0, bottom_up_download.data(), row_bytes, 3);
        }
        int full_destroy_result = vf_roi_batch_destroy(full_batch);
        if (result == VF_CUDA_OK &&
            (full_destroy_result != VF_CUDA_OK || bottom_up_download != bgr)) {
            std::cerr << "negative-stride resident upload changed row order\n";
            return 11;
        }
        if (result == VF_CUDA_OK) {
            result = vf_context_upload_u8(
                context, bgr.data(), width, height, row_bytes, 3, &resident_generation);
        }
    }
    if (result == VF_CUDA_OK) {
        result = vf_plan_execute_roi(
            plan, resident_generation, 0, 0, plan_binary.data(), resized_width, 1);
    }
    uint64_t plan_allocation_count = 0;
    if (result == VF_CUDA_OK) {
        result = vf_context_stats(context, &reserved_bytes, &plan_allocation_count);
    }
    if (result == VF_CUDA_OK) {
        result = vf_plan_execute(
            plan, bgr.data(), width, height, width * 3, 3,
            plan_binary.data(), resized_width, 1);
    }
    uint64_t repeated_allocation_count = 0;
    if (result == VF_CUDA_OK) {
        result = vf_context_stats(context, &reserved_bytes, &repeated_allocation_count);
    }
    VfPlanOperatorV1 dag_operators[3]{};
    dag_operators[0] = operators[0];
    dag_operators[1].struct_size = sizeof(VfPlanOperatorV1);
    dag_operators[1].kind = VF_PLAN_THRESHOLD;
    dag_operators[1].input_node = 0;
    dag_operators[1].output_node = 1;
    dag_operators[1].int_params[0] = 127;
    dag_operators[1].int_params[1] = 255;
    dag_operators[1].int_params[2] = 1;
    dag_operators[2] = operators[3];
    dag_operators[2].input_node = 0;
    dag_operators[2].output_node = 2;
    int32_t dag_output_nodes[2]{1, 2};
    VfDagPlanDescV1 dag_descriptor{};
    dag_descriptor.struct_size = sizeof(VfDagPlanDescV1);
    dag_descriptor.version = VF_CUDA_PLAN_VERSION;
    dag_descriptor.input_channels = 3;
    dag_descriptor.operator_count = 3;
    dag_descriptor.operators = dag_operators;
    dag_descriptor.output_count = 2;
    dag_descriptor.output_nodes = dag_output_nodes;
    char dag_reason[256]{};
    void* dag_plan = nullptr;
    if (result == VF_CUDA_OK) {
        result = vf_dag_plan_query(&dag_descriptor, width, height, dag_reason, sizeof(dag_reason));
    }
    if (result == VF_CUDA_OK) {
        result = vf_dag_plan_create(context, &dag_descriptor, width, height, &dag_plan);
    }
    std::vector<uint8_t> outer_mask(width * height, 0);
    std::vector<uint8_t> inner_mask(width * height, 0);
    VfDagOutputV1 dag_outputs[2]{};
    dag_outputs[0] = {sizeof(VfDagOutputV1), 1, outer_mask.data(), width, 1};
    dag_outputs[1] = {sizeof(VfDagOutputV1), 2, inner_mask.data(), width, 1};
    if (result == VF_CUDA_OK) {
        result = vf_dag_plan_execute(
            dag_plan, bgr.data(), width, height, width * 3, 3, dag_outputs, 2);
    }
    if (result == VF_CUDA_OK) {
        result = vf_dag_plan_execute_roi(
            dag_plan, resident_generation, 0, 0, dag_outputs, 2);
    }
    uint64_t free_bytes = 0;
    uint64_t total_bytes = 0;
    if (result == VF_CUDA_OK) result = vf_gpu_memory_info(&free_bytes, &total_bytes);
    VfRoiV1 batch_rois[2] = {
        {sizeof(VfRoiV1), 0, 0, 4, 4},
        {sizeof(VfRoiV1), 4, 4, 4, 4},
    };
    void* roi_batch = nullptr;
    if (result == VF_CUDA_OK) {
        result = vf_roi_batch_create(context, resident_generation, batch_rois, 2, &roi_batch);
    }
    int batch_count = 0, batch_width = 0, batch_height = 0, batch_channels = 0;
    if (result == VF_CUDA_OK) {
        result = vf_roi_batch_info(
            roi_batch, &batch_count, &batch_width, &batch_height, &batch_channels);
    }
    std::vector<uint8_t> downloaded_roi(4 * 4 * 3, 0);
    if (result == VF_CUDA_OK) {
        result = vf_roi_batch_download_u8(roi_batch, 1, downloaded_roi.data(), 4 * 3, 3);
    }
    int batch_destroy_result = vf_roi_batch_destroy(roi_batch);
    std::vector<uint8_t> resident_cnr_mask(width * height, 0);
    float resident_cnr_median = 0.0f;
    float resident_cnr_mad = 0.0f;
    double resident_cnr_threshold = 0.0;
    if (result == VF_CUDA_OK) {
        result = vf_cnr_mask_u8_roi(
            context, resident_generation, 0, 0, width, height,
            3, 0.0, 3.0, 8.0, 0.000001, 1.4826, 255,
            &resident_cnr_median, &resident_cnr_mad, &resident_cnr_threshold,
            resident_cnr_mask.data(), static_cast<long long>(resident_cnr_mask.size()));
    }
    std::vector<int32_t> candidate_ints(1024 * 7, 0);
    std::vector<float> candidate_floats(1024 * 3, 0.0f);
    int candidate_count = -1;
    int candidate_components = -1;
    int candidate_status = -1;
    if (result == VF_CUDA_OK) {
        const int32_t candidate_int_params[22] = {
            3, 255, VF_MORPH_OPEN, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 1, 0, 0, 0, 1, 4, 0};
        const double candidate_real_params[6] = {0.0, 3.0, 8.0, 0.000001, 1.4826, 1.5};
        float candidate_median = 0.0f;
        float candidate_mad = 0.0f;
        double candidate_threshold = 0.0;
        result = vf_cnr_candidates_u8_roi(
            context, resident_generation, 0, 0, width, height,
            candidate_int_params, 22, candidate_real_params, 6,
            &candidate_median, &candidate_mad, &candidate_threshold,
            candidate_ints.data(), candidate_floats.data(), 1024,
            &candidate_count, &candidate_components, &candidate_status);
        if (result == VF_CUDA_OK && (candidate_count < 0 || candidate_status != 0 ||
                                     candidate_median != resident_cnr_median ||
                                     candidate_mad != resident_cnr_mad ||
                                     candidate_threshold != resident_cnr_threshold)) {
            std::cerr << "resident CNR candidates disagree with the resident CNR mask chain\n";
            return 10;
        }
    }
    VfCudaContextMemoryStatsV1 exercised_memory_stats{};
    exercised_memory_stats.struct_size = sizeof(VfCudaContextMemoryStatsV1);
    exercised_memory_stats.version = 1;
    int exercised_memory_result = result == VF_CUDA_OK
        ? vf_context_memory_stats_v1(context, &exercised_memory_stats)
        : result;
    if (result == VF_CUDA_OK &&
        (exercised_memory_result != VF_CUDA_OK ||
         exercised_memory_stats.plan_bytes == 0 ||
         exercised_memory_stats.resident_bytes == 0 ||
         exercised_memory_stats.median_bytes == 0 ||
         exercised_memory_stats.gaussian_f32_bytes == 0 ||
         exercised_memory_stats.cnr_mask_bytes == 0 ||
         exercised_memory_stats.cnr_candidate_bytes == 0)) {
        std::cerr << "Detailed context memory accounting missed an exercised buffer family\n";
        return 11;
    }
    VfCudaContextMemoryStatsV2 memory_before_trim{};
    memory_before_trim.struct_size = sizeof(VfCudaContextMemoryStatsV2);
    memory_before_trim.version = 2;
    int memory_v2_result = result == VF_CUDA_OK
        ? vf_context_memory_stats_v2(context, &memory_before_trim)
        : result;
    const uint64_t protected_before_trim =
        memory_before_trim.plan_bytes + memory_before_trim.resident_bytes +
        memory_before_trim.template_match_bytes + memory_before_trim.contour_bytes;
    const uint64_t analysis_before_trim =
        memory_before_trim.median_bytes + memory_before_trim.gaussian_f32_bytes +
        memory_before_trim.cnr_mask_bytes + memory_before_trim.cnr_candidate_bytes;
    uint64_t released_analysis_bytes = 0;
    int trim_result = result == VF_CUDA_OK
        ? vf_context_trim_analysis_scratch(context, &released_analysis_bytes)
        : result;
    VfCudaContextMemoryStatsV2 memory_after_trim{};
    memory_after_trim.struct_size = sizeof(VfCudaContextMemoryStatsV2);
    memory_after_trim.version = 2;
    int memory_after_trim_result = result == VF_CUDA_OK
        ? vf_context_memory_stats_v2(context, &memory_after_trim)
        : result;
    const uint64_t protected_after_trim =
        memory_after_trim.plan_bytes + memory_after_trim.resident_bytes +
        memory_after_trim.template_match_bytes + memory_after_trim.contour_bytes;
    const uint64_t analysis_after_trim =
        memory_after_trim.median_bytes + memory_after_trim.gaussian_f32_bytes +
        memory_after_trim.cnr_mask_bytes + memory_after_trim.cnr_candidate_bytes;
    if (result == VF_CUDA_OK &&
        (memory_v2_result != VF_CUDA_OK || trim_result != VF_CUDA_OK ||
         memory_after_trim_result != VF_CUDA_OK || analysis_before_trim == 0 ||
         released_analysis_bytes != analysis_before_trim || analysis_after_trim != 0 ||
         protected_after_trim != protected_before_trim ||
         memory_after_trim.peak_median_bytes < memory_before_trim.median_bytes ||
         memory_after_trim.peak_gaussian_f32_bytes < memory_before_trim.gaussian_f32_bytes ||
         memory_after_trim.peak_cnr_mask_bytes < memory_before_trim.cnr_mask_bytes ||
         memory_after_trim.peak_cnr_candidate_bytes < memory_before_trim.cnr_candidate_bytes)) {
        std::cerr << "Analysis scratch trim changed a protected lifetime or lost its high-water mark\n";
        return 11;
    }
    const float median_input[5]{9.0f, -1.0f, 5.0f, 3.0f, 7.0f};
    float median_after_trim = 0.0f;
    if (result == VF_CUDA_OK) {
        result = vf_median_f32(context, median_input, 5, &median_after_trim);
    }
    if (result == VF_CUDA_OK && median_after_trim != 5.0f) {
        std::cerr << "Analysis scratch failed to reallocate after trim\n";
        return 11;
    }
    // A real device OOM must not leave a stale error for the next ROI batch.
    const int oom_side = 4096;
    std::vector<uint8_t> oom_source(static_cast<size_t>(oom_side) * oom_side, 7);
    uint64_t oom_generation = 0;
    int oom_result = VF_CUDA_OK;
    int oom_recovery_result = VF_CUDA_OK;
    if (result == VF_CUDA_OK) {
        result = vf_context_upload_u8(
            context, oom_source.data(), oom_side, oom_side, oom_side, 1, &oom_generation);
    }
    if (result == VF_CUDA_OK) {
        std::vector<VfRoiV1> oom_rois(65535, VfRoiV1{sizeof(VfRoiV1), 0, 0, oom_side, oom_side});
        void* oom_batch = nullptr;
        oom_result = vf_roi_batch_create(
            context, oom_generation, oom_rois.data(), static_cast<int>(oom_rois.size()), &oom_batch);
        if (oom_result == VF_CUDA_OK) vf_roi_batch_destroy(oom_batch);
        void* recovery_batch = nullptr;
        oom_recovery_result = vf_roi_batch_create(
            context, oom_generation, batch_rois, 2, &recovery_batch);
        if (oom_recovery_result == VF_CUDA_OK) vf_roi_batch_destroy(recovery_batch);
    }
    if (result == VF_CUDA_OK && (oom_result == VF_CUDA_OK || oom_recovery_result != VF_CUDA_OK)) {
        std::cerr << "ROI batch OOM recovery failed: oom_result=" << oom_result
                  << " recovery_result=" << oom_recovery_result << "\n";
        return 9;
    }
    VfCudaTimingsV1 timings{};
    timings.struct_size = sizeof(VfCudaTimingsV1);
    timings.version = 1;
    int timings_result = vf_context_last_timings(context, &timings);
    int dag_destroy_result = vf_dag_plan_destroy(dag_plan);
    int plan_destroy_result = vf_plan_destroy(plan);
    int destroy_result = vf_context_destroy(context);
    if (result != VF_CUDA_OK || plan_destroy_result != VF_CUDA_OK ||
        dag_destroy_result != VF_CUDA_OK || batch_destroy_result != VF_CUDA_OK ||
        timings_result != VF_CUDA_OK || timings.context_create_ms < 0.0f ||
        timings.allocation_ms < 0.0f || timings.h2d_ms < 0.0f ||
        timings.device_copy_ms < 0.0f || timings.kernel_ms < 0.0f ||
        timings.d2h_ms < 0.0f || timings.synchronize_ms < 0.0f ||
        timings.free_ms < 0.0f || timings.morphology_ms < 0.0f ||
        timings.total_device_ms < 0.0f ||
        destroy_result != VF_CUDA_OK || free_bytes == 0 || total_bytes < free_bytes ||
        batch_count != 2 || batch_width != 4 || batch_height != 4 || batch_channels != 3 ||
        plan_allocation_count != repeated_allocation_count) {
        char message[256]{};
        int failed = result != VF_CUDA_OK ? result :
            plan_destroy_result != VF_CUDA_OK ? plan_destroy_result : destroy_result;
        vf_gpu_error_message(failed, message, static_cast<int>(sizeof(message)));
        std::cerr << "Generic native plan/DAG smoke failed: " << message
                  << " plan_reason=" << plan_reason << " dag_reason=" << dag_reason << "\n";
        return 8;
    }

    std::cout << "C ABI, plans, resident ROI and coordinate batch smoke passed\n";
    return 0;
}
