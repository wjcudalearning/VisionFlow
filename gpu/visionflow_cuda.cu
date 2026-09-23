#define VISIONFLOW_CUDA_EXPORTS
#include "visionflow_cuda.h"
#include "visionflow_cuda_internal.cuh"
// The exact-median export radix-sorts its order keys with the CCCL CUB device algorithm that ships
// with the CUDA Toolkit. CUB refuses to compile under the traditional MSVC preprocessor, so
// gpu/cuda_project.json declares /Zc:preprocessor for both native targets and
// gpu/build_cuda_dll.ps1 passes it through -Xcompiler.
#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
// which gpu/cuda_project.json now opts into for the whole project.
// vf_cnr_candidates_u8_roi compacts foreground pixels and components, groups pixels by component
// with a stable radix sort, and prefix-sums per-component offsets with the same CUB toolkit copy.
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_select.cuh>
#if defined(_WIN32)
#include <windows.h>
#endif
#include <algorithm>
#include <cfloat>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;
constexpr int ADAPTIVE_SCAN_THREADS = 256;
constexpr int MAX_GAUSSIAN_KERNEL = 127;
constexpr int TIMING_EVENT_COUNT = 12;
enum TimingEventIndex {
    TIMING_START = 0,
    TIMING_AFTER_INPUT = 1,
    TIMING_AFTER_KERNEL = 2,
    TIMING_AFTER_OUTPUT = 3,
    TIMING_GAUSSIAN_START = 4,
    TIMING_GAUSSIAN_END = 5,
    TIMING_ADAPTIVE_START = 6,
    TIMING_ADAPTIVE_END = 7,
    TIMING_THRESHOLD_START = 8,
    TIMING_THRESHOLD_END = 9,
    TIMING_MORPHOLOGY_START = 10,
    TIMING_MORPHOLOGY_END = 11,
};

constexpr unsigned int GAUSSIAN_FIXED_SHIFT = 8;
constexpr unsigned int GAUSSIAN_FIXED_SCALE = 1U << GAUSSIAN_FIXED_SHIFT;
constexpr unsigned int GAUSSIAN_FINAL_ROUND =
    1U << (GAUSSIAN_FIXED_SHIFT * 2 - 1);
__constant__ uint16_t gaussian_weights[MAX_GAUSSIAN_KERNEL];
// float32 coefficients for the optional vf_gaussian_blur_f32 export. Separate symbol from the
// uint16 fixed-point table above because that one serves the u8 export's rounding contract.
__constant__ float gaussian_f32_weights[MAX_GAUSSIAN_KERNEL];
// Odd kernel sizes the float32 export has been verified against cv2.GaussianBlur. The verified
// range is every odd size in [3, MAX_GAUSSIAN_KERNEL]; anything else is reported as
// VF_CUDA_UNSUPPORTED instead of being computed unvalidated. Evidence:
// outputs_validation/cnr_profile/gaussian_f32_equivalence.txt (tools/gaussian_f32_equivalence.py).
constexpr int GAUSSIAN_F32_MIN_KERNEL = 3;

// Template Anchor Grid scratch planes owned by the persistent context. Declared here because the
// context struct sizes its plane arrays from them.
constexpr int MATCH_SUM_PREFIX_PLANE = 0;      // int64: vertical prefix of the horizontal window sum
constexpr int MATCH_SQUARE_PREFIX_PLANE = 1;   // int64: vertical prefix of its sum of squares
constexpr int MATCH_PLANE_COUNT = 2;

struct PersistentContext {
    uint8_t* u8[5]{};
    size_t u8_capacity[5]{};
    // Shared uint32 plan scratch. Gaussian uses it for its fixed-point horizontal pass; Adaptive
    // Mean reuses it for exact uint32 row prefixes after Gaussian has finished.
    uint32_t* gaussian_buffer = nullptr;
    size_t gaussian_capacity = 0;
    std::vector<uint8_t*> dag_u8;
    std::vector<size_t> dag_u8_capacity;
    uint8_t* resident_u8 = nullptr;
    size_t resident_capacity = 0;
    int resident_width = 0;
    int resident_height = 0;
    int resident_channels = 0;
    uint64_t resident_generation = 0;
    // Template Anchor Grid scratch: three grow-only int64 planes of output_width x output_height,
    // one int64 plane of template column sums, and the candidate/result slots. Kept separate from
    // plan scratch because those allocations are grow-only.
    long long* match_plane[MATCH_PLANE_COUNT]{};
    size_t match_plane_capacity[MATCH_PLANE_COUNT]{};
    long long* match_candidates = nullptr;
    size_t match_candidate_capacity = 0;
    int match_candidate_output_width = 0;
    // Full Pattern Match scratch. Unlike Template Anchor Grid, this path keeps every response on
    // device, selects local peaks, sorts them, applies NMS and downloads only final descriptors.
    float* pattern_scores = nullptr;
    size_t pattern_score_capacity = 0;
    unsigned long long* pattern_keys = nullptr;
    size_t pattern_key_capacity = 0;
    unsigned long long* pattern_sorted_keys = nullptr;
    size_t pattern_sorted_key_capacity = 0;
    unsigned long long* pattern_selected_keys = nullptr;
    size_t pattern_selected_key_capacity = 0;
    uint8_t* pattern_sort_scratch = nullptr;
    size_t pattern_sort_scratch_capacity = 0;
    int32_t* pattern_out_xy = nullptr;
    size_t pattern_out_xy_capacity = 0;
    float* pattern_out_scores = nullptr;
    size_t pattern_out_score_capacity = 0;
    int* pattern_out_count = nullptr;
    size_t pattern_out_count_capacity = 0;
    // Large-template Pattern Match scratch (FFT path). The brute-force response kernel costs
    // output_elements x template_pixels, which a production 2000x12000 template makes impossible,
    // so the numerator comes from an FFT cross correlation while the window statistics stay exact
    // in int64 summed-area tables. The transforms are the Stockham kernels in this file, so the
    // library keeps no external FFT dependency. Three complex planes are needed at once: the
    // template spectrum, the image spectrum and one ping-pong scratch. All grow-only.
    float2* pattern_fft_a = nullptr;
    size_t pattern_fft_a_capacity = 0;
    float2* pattern_fft_b = nullptr;
    size_t pattern_fft_b_capacity = 0;
    float2* pattern_fft_c = nullptr;
    size_t pattern_fft_c_capacity = 0;
    long long* pattern_sat_sum = nullptr;         // summed-area table of the gray frame
    size_t pattern_sat_sum_capacity = 0;
    long long* pattern_sat_square = nullptr;      // summed-area table of its squares
    size_t pattern_sat_square_capacity = 0;
    // Contour extension scratch: the padded label image, the discovery-order result, and the
    // OpenCV-order result the download export copies out. All grow-only.
    signed char* contour_label = nullptr;
    size_t contour_label_capacity = 0;
    int32_t* contour_offsets = nullptr;
    size_t contour_offset_capacity = 0;
    int32_t* contour_points = nullptr;
    size_t contour_point_capacity = 0;
    int32_t* contour_out_offsets = nullptr;
    size_t contour_out_offset_capacity = 0;
    int32_t* contour_out_points = nullptr;
    size_t contour_out_point_capacity = 0;
    int* contour_counts = nullptr;
    size_t contour_count_capacity = 0;
    // RETR_LIST transition list: one exact row count, the row segment table, and the raster-ordered
    // padded label indices the fast scan walks.
    int* contour_row_counts = nullptr;
    size_t contour_row_count_capacity = 0;
    int* contour_row_start = nullptr;
    size_t contour_row_start_capacity = 0;
    int32_t* contour_transitions = nullptr;
    size_t contour_transition_capacity = 0;
    int contour_count = 0;
    int contour_point_count = 0;
    uint64_t contour_generation = 0;
    bool contour_result_valid = false;
    // Exact-median scratch: the uploaded float values, their monotone-orderable uint32 order keys,
    // the radix-sorted keys, the cub temporary storage and a one-word NaN-presence flag. All
    // grow-only and deliberately separate from the shared plan scratch.
    float* median_values = nullptr;
    size_t median_value_capacity = 0;
    uint32_t* median_keys = nullptr;
    size_t median_key_capacity = 0;
    uint32_t* median_sorted_keys = nullptr;
    size_t median_sorted_key_capacity = 0;
    uint8_t* median_sort_scratch = nullptr;
    size_t median_sort_scratch_capacity = 0;
    int* median_nan_flag = nullptr;
    size_t median_nan_flag_capacity = 0;
    // float32 Gaussian scratch for the optional vf_gaussian_blur_f32 export: the uploaded source
    // rectangle, the horizontal intermediate and the packed result that is copied back. Grow-only
    // and deliberately separate from the plan scratch that reserve_device only
    // grows, so sharing them would mix a plan's buffer contents with this operator's input.
    float* gaussian_f32_input = nullptr;
    size_t gaussian_f32_input_capacity = 0;
    float* gaussian_f32_intermediate = nullptr;
    size_t gaussian_f32_intermediate_capacity = 0;
    float* gaussian_f32_output = nullptr;
    size_t gaussian_f32_output_capacity = 0;
    // CNR mask scratch for the optional vf_cnr_mask_f32 export: the uploaded float32 image and
    // background, the residual, its absolute deviation, and the device mask that is copied back.
    // Grow-only and separate from the median and Gaussian scratch for the same reason.
    float* cnr_mask_image = nullptr;
    size_t cnr_mask_image_capacity = 0;
    float* cnr_mask_background = nullptr;
    size_t cnr_mask_background_capacity = 0;
    float* cnr_mask_residual = nullptr;
    size_t cnr_mask_residual_capacity = 0;
    float* cnr_mask_absdev = nullptr;
    size_t cnr_mask_absdev_capacity = 0;
    unsigned char* cnr_mask_mask = nullptr;
    size_t cnr_mask_mask_capacity = 0;
    // vf_cnr_candidates_u8_roi scratch, all grow-only: the morphology ping-pong mask, the union-find
    // parent plane and its change word, the pixel index ramp, the foreground/component compaction
    // and grouping arrays, per-component boxes and flags, per-candidate windows, the gathered ring
    // background values, the downloadable candidate records, and one CUB temporary storage block.
    unsigned char* cand_mask_scratch = nullptr;
    size_t cand_mask_scratch_capacity = 0;
    int32_t* ccl_parent = nullptr;
    size_t ccl_parent_capacity = 0;
    int32_t* cand_words = nullptr;
    size_t cand_words_capacity = 0;
    int32_t* cand_ramp = nullptr;
    size_t cand_ramp_capacity = 0;
    int32_t* cand_foreground = nullptr;
    size_t cand_foreground_capacity = 0;
    int32_t* cand_keys = nullptr;
    size_t cand_keys_capacity = 0;
    int32_t* cand_sorted_keys = nullptr;
    size_t cand_sorted_keys_capacity = 0;
    int32_t* cand_sorted_pixels = nullptr;
    size_t cand_sorted_pixels_capacity = 0;
    int32_t* cand_roots = nullptr;
    size_t cand_roots_capacity = 0;
    int32_t* cand_areas = nullptr;
    size_t cand_areas_capacity = 0;
    int32_t* cand_offsets = nullptr;
    size_t cand_offsets_capacity = 0;
    int32_t* cand_boxes = nullptr;
    size_t cand_boxes_capacity = 0;
    unsigned char* cand_keep = nullptr;
    size_t cand_keep_capacity = 0;
    int32_t* cand_kept = nullptr;
    size_t cand_kept_capacity = 0;
    int32_t* cand_windows = nullptr;
    size_t cand_windows_capacity = 0;
    long long* cand_window_sizes = nullptr;
    size_t cand_window_sizes_capacity = 0;
    long long* cand_gather_offsets = nullptr;
    size_t cand_gather_offsets_capacity = 0;
    float* cand_gather = nullptr;
    size_t cand_gather_capacity = 0;
    int32_t* cand_out_ints = nullptr;
    size_t cand_out_ints_capacity = 0;
    float* cand_out_floats = nullptr;
    size_t cand_out_floats_capacity = 0;
    uint8_t* cand_cub_scratch = nullptr;
    size_t cand_cub_scratch_capacity = 0;
    // Parallel ring statistics: per-slot background flags, the value sequences (component values then
    // compacted background values), per-candidate counts/offsets, the pairwise leaf table and results.
    unsigned char* cand_flags = nullptr;
    size_t cand_flags_capacity = 0;
    float* cand_values = nullptr;
    size_t cand_values_capacity = 0;
    long long* cand_value_offsets = nullptr;
    size_t cand_value_offsets_capacity = 0;
    long long* cand_background_counts = nullptr;
    size_t cand_background_counts_capacity = 0;
    long long* cand_background_offsets = nullptr;
    size_t cand_background_offsets_capacity = 0;
    long long* cand_segment_ends = nullptr;
    size_t cand_segment_ends_capacity = 0;
    long long* cand_seq_start = nullptr;
    size_t cand_seq_start_capacity = 0;
    long long* cand_seq_length = nullptr;
    size_t cand_seq_length_capacity = 0;
    long long* cand_leaf_counts = nullptr;
    size_t cand_leaf_counts_capacity = 0;
    long long* cand_leaf_offsets = nullptr;
    size_t cand_leaf_offsets_capacity = 0;
    long long* cand_leaf_start = nullptr;
    size_t cand_leaf_start_capacity = 0;
    int32_t* cand_leaf_length = nullptr;
    size_t cand_leaf_length_capacity = 0;
    int32_t* cand_leaf_sequence = nullptr;
    size_t cand_leaf_sequence_capacity = 0;
    float* cand_leaf_values = nullptr;
    size_t cand_leaf_values_capacity = 0;
    float* cand_seq_mean = nullptr;
    size_t cand_seq_mean_capacity = 0;
    float* cand_seq_std = nullptr;
    size_t cand_seq_std_capacity = 0;
    unsigned long long allocation_count = 0;
    uint64_t peak_reserved_bytes = 0;
    uint64_t peak_memory_bytes[8]{};
    cudaStream_t stream = nullptr;
    cudaError_t initialization_error = cudaSuccess;
    cudaEvent_t timing_events[TIMING_EVENT_COUNT]{};
    VfCudaTimingsV1 last_timings{};
    float pending_allocation_ms = 0.0f;
    bool timing_enabled = true;
    bool timing_input_is_host = true;
    bool timing_has_gaussian = false;
    bool timing_has_adaptive = false;
    bool timing_has_threshold = false;
    bool timing_has_morphology = false;

    PersistentContext() {
        initialization_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        last_timings.struct_size = sizeof(VfCudaTimingsV1);
        last_timings.version = 1;
        if (initialization_error == cudaSuccess) {
            for (cudaEvent_t& event : timing_events) {
                initialization_error = cudaEventCreate(&event);
                if (initialization_error != cudaSuccess) break;
            }
        }
    }

    ~PersistentContext() {
        for (void* pointer : u8) visionflow_cuda::free_device(pointer);
        visionflow_cuda::free_device(gaussian_buffer);
        for (void* pointer : dag_u8) visionflow_cuda::free_device(pointer);
        for (long long* plane : match_plane) visionflow_cuda::free_device(plane);
        visionflow_cuda::free_device(match_candidates);
        visionflow_cuda::free_device(contour_label);
        visionflow_cuda::free_device(contour_offsets);
        visionflow_cuda::free_device(contour_points);
        visionflow_cuda::free_device(contour_out_offsets);
        visionflow_cuda::free_device(contour_out_points);
        visionflow_cuda::free_device(contour_counts);
        visionflow_cuda::free_device(contour_row_counts);
        visionflow_cuda::free_device(contour_row_start);
        visionflow_cuda::free_device(contour_transitions);
        visionflow_cuda::free_device(median_values);
        visionflow_cuda::free_device(median_keys);
        visionflow_cuda::free_device(median_sorted_keys);
        visionflow_cuda::free_device(median_sort_scratch);
        visionflow_cuda::free_device(median_nan_flag);
        visionflow_cuda::free_device(gaussian_f32_input);
        visionflow_cuda::free_device(gaussian_f32_intermediate);
        visionflow_cuda::free_device(gaussian_f32_output);
        visionflow_cuda::free_device(pattern_scores);
        visionflow_cuda::free_device(pattern_keys);
        visionflow_cuda::free_device(pattern_sorted_keys);
        visionflow_cuda::free_device(pattern_selected_keys);
        visionflow_cuda::free_device(pattern_sort_scratch);
        visionflow_cuda::free_device(pattern_out_xy);
        visionflow_cuda::free_device(pattern_out_scores);
        visionflow_cuda::free_device(pattern_out_count);
        visionflow_cuda::free_device(pattern_fft_a);
        visionflow_cuda::free_device(pattern_fft_b);
        visionflow_cuda::free_device(pattern_fft_c);
        visionflow_cuda::free_device(pattern_sat_sum);
        visionflow_cuda::free_device(pattern_sat_square);
        visionflow_cuda::free_device(cnr_mask_image);
        visionflow_cuda::free_device(cnr_mask_background);
        visionflow_cuda::free_device(cnr_mask_residual);
        visionflow_cuda::free_device(cnr_mask_absdev);
        visionflow_cuda::free_device(cnr_mask_mask);
        for (void* pointer : {
                 static_cast<void*>(cand_mask_scratch), static_cast<void*>(ccl_parent),
                 static_cast<void*>(cand_words), static_cast<void*>(cand_ramp),
                 static_cast<void*>(cand_foreground), static_cast<void*>(cand_keys),
                 static_cast<void*>(cand_sorted_keys), static_cast<void*>(cand_sorted_pixels),
                 static_cast<void*>(cand_roots), static_cast<void*>(cand_areas),
                 static_cast<void*>(cand_offsets), static_cast<void*>(cand_boxes),
                 static_cast<void*>(cand_keep), static_cast<void*>(cand_kept),
                 static_cast<void*>(cand_windows), static_cast<void*>(cand_window_sizes),
                 static_cast<void*>(cand_gather_offsets), static_cast<void*>(cand_gather),
                 static_cast<void*>(cand_out_ints), static_cast<void*>(cand_out_floats),
                 static_cast<void*>(cand_cub_scratch), static_cast<void*>(cand_flags),
                 static_cast<void*>(cand_values), static_cast<void*>(cand_value_offsets),
                 static_cast<void*>(cand_background_counts), static_cast<void*>(cand_background_offsets),
                 static_cast<void*>(cand_segment_ends), static_cast<void*>(cand_seq_start),
                 static_cast<void*>(cand_seq_length), static_cast<void*>(cand_leaf_counts),
                 static_cast<void*>(cand_leaf_offsets), static_cast<void*>(cand_leaf_start),
                 static_cast<void*>(cand_leaf_length), static_cast<void*>(cand_leaf_sequence),
                 static_cast<void*>(cand_leaf_values), static_cast<void*>(cand_seq_mean),
                 static_cast<void*>(cand_seq_std)}) {
            visionflow_cuda::free_device(pointer);
        }
        visionflow_cuda::free_device(resident_u8);
        for (cudaEvent_t event : timing_events) {
            if (event != nullptr) cudaEventDestroy(event);
        }
        if (stream != nullptr) cudaStreamDestroy(stream);
    }
};

struct ContextMemoryBreakdown {
    uint64_t plan_bytes = 0;
    uint64_t resident_bytes = 0;
    uint64_t template_match_bytes = 0;
    uint64_t contour_bytes = 0;
    uint64_t median_bytes = 0;
    uint64_t gaussian_f32_bytes = 0;
    uint64_t cnr_mask_bytes = 0;
    uint64_t cnr_candidate_bytes = 0;

    uint64_t total() const {
        return plan_bytes + resident_bytes + template_match_bytes + contour_bytes + median_bytes +
               gaussian_f32_bytes + cnr_mask_bytes + cnr_candidate_bytes;
    }
};

uint64_t capacity_bytes(size_t capacity, size_t item_size) {
    return static_cast<uint64_t>(capacity) * static_cast<uint64_t>(item_size);
}

ContextMemoryBreakdown context_memory_breakdown(const PersistentContext* context) {
    ContextMemoryBreakdown memory{};
    for (size_t capacity : context->u8_capacity) memory.plan_bytes += capacity_bytes(capacity, 1);
    memory.plan_bytes += capacity_bytes(context->gaussian_capacity, sizeof(uint32_t));
    for (size_t capacity : context->dag_u8_capacity) memory.plan_bytes += capacity_bytes(capacity, 1);
    memory.resident_bytes = capacity_bytes(context->resident_capacity, 1);

    for (size_t capacity : context->match_plane_capacity) {
        memory.template_match_bytes += capacity_bytes(capacity, sizeof(long long));
    }
    memory.template_match_bytes +=
        capacity_bytes(context->match_candidate_capacity, sizeof(long long));
    memory.template_match_bytes +=
        capacity_bytes(context->pattern_score_capacity, sizeof(float)) +
        capacity_bytes(context->pattern_key_capacity, sizeof(unsigned long long)) +
        capacity_bytes(context->pattern_sorted_key_capacity, sizeof(unsigned long long)) +
        capacity_bytes(context->pattern_selected_key_capacity, sizeof(unsigned long long)) +
        capacity_bytes(context->pattern_sort_scratch_capacity, 1) +
        capacity_bytes(context->pattern_out_xy_capacity, sizeof(int32_t)) +
        capacity_bytes(context->pattern_out_score_capacity, sizeof(float)) +
        capacity_bytes(context->pattern_out_count_capacity, sizeof(int)) +
        capacity_bytes(context->pattern_fft_a_capacity, sizeof(float2)) +
        capacity_bytes(context->pattern_fft_b_capacity, sizeof(float2)) +
        capacity_bytes(context->pattern_fft_c_capacity, sizeof(float2)) +
        capacity_bytes(context->pattern_sat_sum_capacity, sizeof(long long)) +
        capacity_bytes(context->pattern_sat_square_capacity, sizeof(long long));

    memory.contour_bytes =
        capacity_bytes(context->contour_label_capacity, sizeof(signed char)) +
        capacity_bytes(context->contour_offset_capacity, sizeof(int32_t)) +
        capacity_bytes(context->contour_point_capacity, sizeof(int32_t)) +
        capacity_bytes(context->contour_out_offset_capacity, sizeof(int32_t)) +
        capacity_bytes(context->contour_out_point_capacity, sizeof(int32_t)) +
        capacity_bytes(context->contour_count_capacity, sizeof(int)) +
        capacity_bytes(context->contour_row_count_capacity, sizeof(int)) +
        capacity_bytes(context->contour_row_start_capacity, sizeof(int)) +
        capacity_bytes(context->contour_transition_capacity, sizeof(int32_t));

    memory.median_bytes =
        capacity_bytes(context->median_value_capacity, sizeof(float)) +
        capacity_bytes(context->median_key_capacity, sizeof(uint32_t)) +
        capacity_bytes(context->median_sorted_key_capacity, sizeof(uint32_t)) +
        capacity_bytes(context->median_sort_scratch_capacity, 1) +
        capacity_bytes(context->median_nan_flag_capacity, sizeof(int));

    memory.gaussian_f32_bytes =
        capacity_bytes(context->gaussian_f32_input_capacity, sizeof(float)) +
        capacity_bytes(context->gaussian_f32_intermediate_capacity, sizeof(float)) +
        capacity_bytes(context->gaussian_f32_output_capacity, sizeof(float));

    memory.cnr_mask_bytes =
        capacity_bytes(context->cnr_mask_image_capacity, sizeof(float)) +
        capacity_bytes(context->cnr_mask_background_capacity, sizeof(float)) +
        capacity_bytes(context->cnr_mask_residual_capacity, sizeof(float)) +
        capacity_bytes(context->cnr_mask_absdev_capacity, sizeof(float)) +
        capacity_bytes(context->cnr_mask_mask_capacity, sizeof(unsigned char));

    memory.cnr_candidate_bytes =
        capacity_bytes(context->cand_mask_scratch_capacity, sizeof(unsigned char)) +
        capacity_bytes(context->ccl_parent_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_words_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_ramp_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_foreground_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_keys_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_sorted_keys_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_sorted_pixels_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_roots_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_areas_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_offsets_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_boxes_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_keep_capacity, sizeof(unsigned char)) +
        capacity_bytes(context->cand_kept_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_windows_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_window_sizes_capacity, sizeof(long long)) +
        capacity_bytes(context->cand_gather_offsets_capacity, sizeof(long long)) +
        capacity_bytes(context->cand_gather_capacity, sizeof(float)) +
        capacity_bytes(context->cand_out_ints_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_out_floats_capacity, sizeof(float)) +
        capacity_bytes(context->cand_cub_scratch_capacity, 1) +
        capacity_bytes(context->cand_flags_capacity, sizeof(unsigned char)) +
        capacity_bytes(context->cand_values_capacity, sizeof(float)) +
        capacity_bytes(context->cand_value_offsets_capacity, sizeof(long long)) +
        capacity_bytes(context->cand_background_counts_capacity, sizeof(long long)) +
        capacity_bytes(context->cand_background_offsets_capacity, sizeof(long long)) +
        capacity_bytes(context->cand_segment_ends_capacity, sizeof(long long)) +
        capacity_bytes(context->cand_seq_start_capacity, sizeof(long long)) +
        capacity_bytes(context->cand_seq_length_capacity, sizeof(long long)) +
        capacity_bytes(context->cand_leaf_counts_capacity, sizeof(long long)) +
        capacity_bytes(context->cand_leaf_offsets_capacity, sizeof(long long)) +
        capacity_bytes(context->cand_leaf_start_capacity, sizeof(long long)) +
        capacity_bytes(context->cand_leaf_length_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_leaf_sequence_capacity, sizeof(int32_t)) +
        capacity_bytes(context->cand_leaf_values_capacity, sizeof(float)) +
        capacity_bytes(context->cand_seq_mean_capacity, sizeof(float)) +
        capacity_bytes(context->cand_seq_std_capacity, sizeof(float));
    return memory;
}

ContextMemoryBreakdown update_context_memory_peak(PersistentContext* context) {
    ContextMemoryBreakdown memory = context_memory_breakdown(context);
    context->peak_reserved_bytes = std::max(context->peak_reserved_bytes, memory.total());
    const uint64_t current[8] = {
        memory.plan_bytes, memory.resident_bytes, memory.template_match_bytes,
        memory.contour_bytes, memory.median_bytes, memory.gaussian_f32_bytes,
        memory.cnr_mask_bytes, memory.cnr_candidate_bytes,
    };
    for (int index = 0; index < 8; ++index) {
        context->peak_memory_bytes[index] =
            std::max(context->peak_memory_bytes[index], current[index]);
    }
    return memory;
}

float elapsed_host_ms(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - started).count();
}

void record_timing_event(PersistentContext* context, int event_index) {
    if (context->timing_enabled) {
        cudaEventRecord(context->timing_events[event_index], context->stream);
    }
}

void reset_timing(PersistentContext* context, bool input_is_host) {
    if (!context->timing_enabled) return;
    float context_create_ms = context->last_timings.context_create_ms;
    float allocation_ms = context->pending_allocation_ms;
    context->pending_allocation_ms = 0.0f;
    context->last_timings = {};
    context->last_timings.struct_size = sizeof(VfCudaTimingsV1);
    context->last_timings.version = 1;
    context->last_timings.context_create_ms = context_create_ms;
    context->last_timings.allocation_ms = allocation_ms;
    context->timing_input_is_host = input_is_host;
    context->timing_has_gaussian = false;
    context->timing_has_adaptive = false;
    context->timing_has_threshold = false;
    context->timing_has_morphology = false;
    record_timing_event(context, TIMING_START);
}

void finalize_timing(PersistentContext* context) {
    if (!context->timing_enabled) return;
    float input_ms = 0.0f;
    cudaEventElapsedTime(
        &input_ms, context->timing_events[TIMING_START],
        context->timing_events[TIMING_AFTER_INPUT]);
    if (context->timing_input_is_host) context->last_timings.h2d_ms = input_ms;
    else context->last_timings.device_copy_ms = input_ms;
    cudaEventElapsedTime(
        &context->last_timings.kernel_ms,
        context->timing_events[TIMING_AFTER_INPUT],
        context->timing_events[TIMING_AFTER_KERNEL]);
    cudaEventElapsedTime(
        &context->last_timings.d2h_ms,
        context->timing_events[TIMING_AFTER_KERNEL],
        context->timing_events[TIMING_AFTER_OUTPUT]);
    cudaEventElapsedTime(
        &context->last_timings.total_device_ms,
        context->timing_events[TIMING_START],
        context->timing_events[TIMING_AFTER_OUTPUT]);
    if (context->timing_has_gaussian) {
        cudaEventElapsedTime(
            &context->last_timings.gaussian_ms,
            context->timing_events[TIMING_GAUSSIAN_START],
            context->timing_events[TIMING_GAUSSIAN_END]);
    }
    if (context->timing_has_adaptive) {
        cudaEventElapsedTime(
            &context->last_timings.adaptive_integral_ms,
            context->timing_events[TIMING_ADAPTIVE_START],
            context->timing_events[TIMING_ADAPTIVE_END]);
    }
    if (context->timing_has_threshold) {
        cudaEventElapsedTime(
            &context->last_timings.threshold_ms,
            context->timing_events[TIMING_THRESHOLD_START],
            context->timing_events[TIMING_THRESHOLD_END]);
    }
    if (context->timing_has_morphology) {
        cudaEventElapsedTime(
            &context->last_timings.morphology_ms,
            context->timing_events[TIMING_MORPHOLOGY_START],
            context->timing_events[TIMING_MORPHOLOGY_END]);
    }
}

enum AreaResizeMode {
    AREA_RESIZE_COPY = 0,
    AREA_RESIZE_FAST_2X2 = 1,
    AREA_RESIZE_FAST_INTEGER = 2,
    AREA_RESIZE_GENERAL = 3,
};

// Device tables reproducing OpenCV INTER_AREA downscale for one source/target shape.
// indices: [x offsets (dw+1)] [y offsets (dh+1)] [x sources] [y sources]
// alphas:  [x weights] [y weights]
struct AreaResizeTables {
    int mode = AREA_RESIZE_COPY;
    int scale_x = 1;
    int scale_y = 1;
    float inverse_area = 1.0f;
    int x_entries = 0;
    int* indices = nullptr;
    float* alphas = nullptr;

    AreaResizeTables() = default;
    AreaResizeTables(const AreaResizeTables&) = delete;
    AreaResizeTables& operator=(const AreaResizeTables&) = delete;
    ~AreaResizeTables() {
        visionflow_cuda::free_device(indices);
        visionflow_cuda::free_device(alphas);
    }
};

struct NativePlan {
    PersistentContext* context = nullptr;
    int width = 0;
    int height = 0;
    int output_width = 0;
    int output_height = 0;
    int input_channels = 0;
    int output_channels = 0;
    std::vector<VfPlanOperatorV1> operators;
    std::vector<std::unique_ptr<AreaResizeTables>> area_resizes;
};

struct NativeDagPlan {
    PersistentContext* context = nullptr;
    int width = 0;
    int height = 0;
    int input_channels = 0;
    std::vector<VfPlanOperatorV1> operators;
    std::vector<int> node_channels;
    std::vector<int> output_nodes;
};

struct NativeRoiBatch {
    PersistentContext* context = nullptr;
    uint8_t* data = nullptr;
    VfRoiV1* device_rois = nullptr;
    int count = 0;
    int width = 0;
    int height = 0;
    int channels = 0;

    ~NativeRoiBatch() {
        visionflow_cuda::free_device(data);
        visionflow_cuda::free_device(device_rois);
    }
};

template <typename T>
int reserve_device(
    T** pointer,
    size_t* capacity,
    size_t count,
    unsigned long long* allocation_count = nullptr) {
    if (pointer == nullptr || capacity == nullptr || count == 0 || count > SIZE_MAX / sizeof(T)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (*pointer != nullptr && *capacity >= count) return VF_CUDA_OK;
    T* replacement = nullptr;
    cudaError_t error = cudaMalloc(&replacement, count * sizeof(T));
    if (error != cudaSuccess) return visionflow_cuda::runtime_error(error);
    visionflow_cuda::free_device(*pointer);
    *pointer = replacement;
    *capacity = count;
    if (allocation_count != nullptr) ++(*allocation_count);
    return VF_CUDA_OK;
}

template <typename T>
int reserve_exact(
    T** pointer,
    size_t* capacity,
    size_t count,
    unsigned long long* allocation_count = nullptr) {
    if (pointer == nullptr || capacity == nullptr || count == 0 || count > SIZE_MAX / sizeof(T)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (*pointer != nullptr && *capacity == count) return VF_CUDA_OK;
    T* replacement = nullptr;
    cudaError_t error = cudaMalloc(&replacement, count * sizeof(T));
    if (error != cudaSuccess) return visionflow_cuda::runtime_error(error);
    visionflow_cuda::free_device(*pointer);
    *pointer = replacement;
    *capacity = count;
    if (allocation_count != nullptr) ++(*allocation_count);
    return VF_CUDA_OK;
}

int prepare_gaussian_weights(
    int kernel,
    int* radius_out,
    cudaStream_t stream = nullptr) {
    if (radius_out == nullptr || kernel < 3 || kernel % 2 == 0 || kernel > MAX_GAUSSIAN_KERNEL) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    std::vector<uint16_t> weights(kernel);
    int radius = kernel / 2;
    if (kernel == 3) {
        weights = {64, 128, 64};
    } else if (kernel == 5) {
        weights = {16, 64, 96, 64, 16};
    } else if (kernel == 7) {
        weights = {8, 28, 56, 72, 56, 28, 8};
    } else if (kernel == 9) {
        weights = {4, 13, 30, 51, 60, 51, 30, 13, 4};
    } else {
        double sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8;
        std::vector<double> normalized(kernel);
        double total = 0.0;
        for (int i = -radius; i <= radius; ++i) {
            double value = std::exp(
                -(static_cast<double>(i) * i) / (2.0 * sigma * sigma));
            normalized[i + radius] = value;
            total += value;
        }
        for (double& value : normalized) value /= total;

        double error = 0.0;
        unsigned int side_sum = 0;
        for (int index = 0; index < radius; ++index) {
            double adjusted = normalized[index] * GAUSSIAN_FIXED_SCALE + error;
            unsigned int value = static_cast<unsigned int>(std::nearbyint(adjusted));
            error = adjusted - value;
            weights[index] = static_cast<uint16_t>(value);
            weights[kernel - 1 - index] = static_cast<uint16_t>(value);
            side_sum += value;
        }
        weights[radius] = static_cast<uint16_t>(GAUSSIAN_FIXED_SCALE - side_sum * 2);
    }
    const size_t weight_bytes = static_cast<size_t>(kernel) * sizeof(uint16_t);
    cudaError_t error = stream == nullptr
        ? cudaMemcpyToSymbol(gaussian_weights, weights.data(), weight_bytes)
        : cudaMemcpyToSymbolAsync(
            gaussian_weights,
            weights.data(),
            weight_bytes,
            0,
            cudaMemcpyHostToDevice,
            stream);
    if (error != cudaSuccess) return visionflow_cuda::runtime_error(error);
    *radius_out = radius;
    return VF_CUDA_OK;
}

// Exact cv::getGaussianKernel(ksize, sigma, CV_32F) coefficients for the float32 export.
//
// cv2.GaussianBlur(src, (k, k), sigma) calls getGaussianKernel(k, sigma, CV_32F) for both axes
// (sigma2 defaults to sigma1), so this function has to reproduce both of its branches:
//   - sigma > 0: sigmaX is that sigma and the coefficients are exp/sum/normalize in float64 with a
//     single cast to float32, for every ksize.
//   - sigma <= 0: sigmaX is the automatic rule 0.3*((ksize-1)*0.5-1)+0.8, except that OpenCV
//     substitutes its fixed small-kernel table for odd ksize <= SMALL_GAUSSIAN_SIZE, which is 9 in
//     OpenCV 5.x.
// Both branches were compared bit-for-bit against cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F)
// by tools/gaussian_f32_equivalence.py, so the coefficients - and therefore the kernel sums - are
// identical to OpenCV's; only the accumulation order of the convolution differs.
bool gaussian_f32_kernel_supported(int kernel) {
    return kernel >= GAUSSIAN_F32_MIN_KERNEL && kernel <= MAX_GAUSSIAN_KERNEL && kernel % 2 == 1;
}

int prepare_gaussian_f32_weights(
    int kernel,
    double sigma,
    int* radius_out,
    cudaStream_t stream = nullptr) {
    if (radius_out == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    if (!gaussian_f32_kernel_supported(kernel)) return VF_CUDA_UNSUPPORTED;
    // Rows are indexed by ksize / 2: 1, 3, 5, 7 and 9. The values are the exact float32 constants
    // OpenCV stores in small_gaussian_tab, already summing to 1.
    static const float small_kernels[5][9] = {
        {1.0f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f},
        {0.015625f, 0.05078125f, 0.1171875f, 0.19921875f, 0.234375f,
         0.19921875f, 0.1171875f, 0.05078125f, 0.015625f},
    };
    // A NaN or infinite sigma has no OpenCV coefficient rule to reproduce, so it is refused instead
    // of being folded into the automatic branch.
    if (std::isnan(sigma) || std::isinf(sigma)) return VF_CUDA_INVALID_ARGUMENT;
    const bool automatic = !(sigma > 0.0);
    std::vector<float> values(static_cast<size_t>(kernel));
    if (automatic && kernel <= 9) {
        const float* fixed = small_kernels[kernel / 2];
        for (int i = 0; i < kernel; ++i) values[static_cast<size_t>(i)] = fixed[i];
    } else {
        const double sigma_x = automatic ? 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8 : sigma;
        const double scale = -0.5 / (sigma_x * sigma_x);
        std::vector<double> raw(static_cast<size_t>(kernel));
        double total = 0.0;
        for (int i = 0; i < kernel; ++i) {
            const double offset = static_cast<double>(i) - (kernel - 1) * 0.5;
            raw[static_cast<size_t>(i)] = std::exp(scale * offset * offset);
            total += raw[static_cast<size_t>(i)];
        }
        const double inverse = 1.0 / total;
        for (int i = 0; i < kernel; ++i) {
            values[static_cast<size_t>(i)] =
                static_cast<float>(raw[static_cast<size_t>(i)] * inverse);
        }
    }
    const size_t weight_bytes = static_cast<size_t>(kernel) * sizeof(float);
    cudaError_t error = stream == nullptr
        ? cudaMemcpyToSymbol(gaussian_f32_weights, values.data(), weight_bytes)
        : cudaMemcpyToSymbolAsync(
            gaussian_f32_weights,
            values.data(),
            weight_bytes,
            0,
            cudaMemcpyHostToDevice,
            stream);
    if (error != cudaSuccess) return visionflow_cuda::runtime_error(error);
    *radius_out = kernel / 2;
    return VF_CUDA_OK;
}

int adaptive_layout(
    int width,
    int height,
    int block,
    size_t* scratch_count_out) {
    if (width <= 0 || height <= 0 || block < 3 || block % 2 == 0 ||
        scratch_count_out == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (static_cast<unsigned long long>(width) > UINT_MAX / 255ULL ||
        static_cast<unsigned long long>(block) > UINT_MAX / 255ULL ||
        static_cast<unsigned long long>(block) * block > ULLONG_MAX / 255ULL ||
        static_cast<size_t>(width) > SIZE_MAX / static_cast<size_t>(height)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    size_t scratch_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (scratch_count > SIZE_MAX / sizeof(uint32_t)) return VF_CUDA_INVALID_ARGUMENT;
    *scratch_count_out = scratch_count;
    return VF_CUDA_OK;
}

void write_reason(char* reason, int capacity, const char* message) {
    if (reason != nullptr && capacity > 0) strncpy_s(reason, capacity, message, _TRUNCATE);
}

int validate_plan_desc(
    const VfPlanDescV1* desc,
    int width,
    int height,
    int* output_channels,
    int* output_width,
    int* output_height,
    char* reason,
    int reason_capacity) {
    if (desc == nullptr || desc->struct_size != sizeof(VfPlanDescV1) ||
        desc->version != VF_CUDA_PLAN_VERSION || width <= 0 || height <= 0 ||
        (desc->input_channels != 1 && desc->input_channels != 3) ||
        desc->operator_count <= 0 || desc->operator_count > 64 || desc->operators == nullptr) {
        write_reason(reason, reason_capacity, "Invalid plan descriptor, version, shape or input channels");
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (static_cast<size_t>(width) > SIZE_MAX / static_cast<size_t>(height) ||
        static_cast<size_t>(width) * static_cast<size_t>(height) >
            SIZE_MAX / static_cast<size_t>(desc->input_channels)) {
        write_reason(reason, reason_capacity, "Plan image shape overflows addressable memory");
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (static_cast<size_t>(width) * static_cast<size_t>(height) > INT_MAX) {
        write_reason(reason, reason_capacity, "Plan image contains too many pixels for ABI v1 indexing");
        return VF_CUDA_UNSUPPORTED;
    }

    int channels = desc->input_channels;
    int current_width = width;
    int current_height = height;
    int previous_node = VF_PLAN_INPUT_NODE;
    for (int index = 0; index < desc->operator_count; ++index) {
        const VfPlanOperatorV1& op = desc->operators[index];
        if (op.struct_size != sizeof(VfPlanOperatorV1) || op.input_node != previous_node ||
            op.output_node <= previous_node) {
            write_reason(reason, reason_capacity, "Plan nodes must form one validated linear chain");
            return VF_CUDA_INVALID_ARGUMENT;
        }
        switch (op.kind) {
            case VF_PLAN_GRAY:
                channels = 1;
                break;
            case VF_PLAN_GAUSSIAN:
                if (op.int_params[0] < 3 || op.int_params[0] % 2 == 0 ||
                    op.int_params[0] > MAX_GAUSSIAN_KERNEL) {
                    write_reason(reason, reason_capacity, "Gaussian kernel is unsupported");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            case VF_PLAN_THRESHOLD:
                if (channels != 1 || op.int_params[0] < 0 || op.int_params[0] > 255 ||
                    op.int_params[1] < 0 || op.int_params[1] > 255 ||
                    (op.int_params[2] != 0 && op.int_params[2] != 1)) {
                    write_reason(reason, reason_capacity, "Threshold requires one channel and valid uint8 parameters");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            case VF_PLAN_ADAPTIVE_MEAN: {
                size_t scratch_count = 0;
                if (channels != 1 || op.int_params[1] < 0 || op.int_params[1] > 255 ||
                    (op.int_params[2] != 0 && op.int_params[2] != 1) ||
                    !std::isfinite(op.float_params[0]) ||
                    adaptive_layout(
                        current_width, current_height, op.int_params[0], &scratch_count) !=
                        VF_CUDA_OK) {
                    write_reason(reason, reason_capacity, "AdaptiveMean shape or parameters are unsupported");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            }
            case VF_PLAN_MORPHOLOGY:
                if (op.int_params[0] < VF_MORPH_OPEN || op.int_params[0] > VF_MORPH_ERODE ||
                    op.int_params[1] < 3 || op.int_params[1] % 2 == 0 ||
                    op.int_params[2] < 1 || op.int_params[2] > INT_MAX / 2) {
                    write_reason(reason, reason_capacity, "Morphology parameters are unsupported");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            case VF_PLAN_RESIZE_AREA:
                if (channels != 1 || op.int_params[0] <= 0 || op.int_params[1] <= 0 ||
                    op.int_params[0] > current_width || op.int_params[1] > current_height) {
                    write_reason(
                        reason, reason_capacity,
                        "Resize(area) requires one channel and non-expanding target dimensions");
                    return VF_CUDA_UNSUPPORTED;
                }
                current_width = op.int_params[0];
                current_height = op.int_params[1];
                break;
            default:
                write_reason(reason, reason_capacity, "Plan contains an unsupported operator kind");
                return VF_CUDA_UNSUPPORTED;
        }
        previous_node = op.output_node;
    }
    if (desc->output_node != previous_node) {
        write_reason(reason, reason_capacity, "Plan output node does not match the final operator");
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (output_channels != nullptr) *output_channels = channels;
    if (output_width != nullptr) *output_width = current_width;
    if (output_height != nullptr) *output_height = current_height;
    write_reason(reason, reason_capacity, "Supported generic native linear plan");
    return VF_CUDA_OK;
}

int validate_dag_plan_desc(
    const VfDagPlanDescV1* desc,
    int width,
    int height,
    std::vector<int>* node_channels,
    char* reason,
    int reason_capacity) {
    if (desc == nullptr || desc->struct_size != sizeof(VfDagPlanDescV1) ||
        desc->version != VF_CUDA_PLAN_VERSION || width <= 0 || height <= 0 ||
        (desc->input_channels != 1 && desc->input_channels != 3) ||
        desc->operator_count <= 0 || desc->operator_count > 64 || desc->operators == nullptr ||
        desc->output_count <= 0 || desc->output_count > desc->operator_count ||
        desc->output_nodes == nullptr) {
        write_reason(reason, reason_capacity, "Invalid DAG descriptor, version, shape or counts");
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (static_cast<size_t>(width) > SIZE_MAX / static_cast<size_t>(height) ||
        static_cast<size_t>(width) * static_cast<size_t>(height) > INT_MAX) {
        write_reason(reason, reason_capacity, "DAG image shape is unsupported");
        return VF_CUDA_UNSUPPORTED;
    }
    std::vector<int> channels;
    try {
        channels.resize(desc->operator_count);
    } catch (const std::bad_alloc&) {
        return VF_CUDA_ALLOCATION_FAILED;
    }
    for (int index = 0; index < desc->operator_count; ++index) {
        const VfPlanOperatorV1& op = desc->operators[index];
        if (op.struct_size != sizeof(VfPlanOperatorV1) || op.output_node != index ||
            op.input_node < VF_PLAN_INPUT_NODE || op.input_node >= index) {
            write_reason(reason, reason_capacity, "DAG nodes must be topologically ordered");
            return VF_CUDA_INVALID_ARGUMENT;
        }
        int input_channels = op.input_node == VF_PLAN_INPUT_NODE
            ? desc->input_channels : channels[op.input_node];
        int output_channels = input_channels;
        switch (op.kind) {
            case VF_PLAN_GRAY:
                output_channels = 1;
                break;
            case VF_PLAN_GAUSSIAN:
                if (op.int_params[0] < 3 || op.int_params[0] % 2 == 0 ||
                    op.int_params[0] > MAX_GAUSSIAN_KERNEL) {
                    write_reason(reason, reason_capacity, "DAG Gaussian kernel is unsupported");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            case VF_PLAN_THRESHOLD:
                if (input_channels != 1 || op.int_params[0] < 0 || op.int_params[0] > 255 ||
                    op.int_params[1] < 0 || op.int_params[1] > 255 ||
                    (op.int_params[2] != 0 && op.int_params[2] != 1)) {
                    write_reason(reason, reason_capacity, "DAG Threshold parameters are unsupported");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            case VF_PLAN_ADAPTIVE_MEAN: {
                size_t scratch_count = 0;
                if (input_channels != 1 || op.int_params[1] < 0 || op.int_params[1] > 255 ||
                    (op.int_params[2] != 0 && op.int_params[2] != 1) ||
                    !std::isfinite(op.float_params[0]) ||
                    adaptive_layout(
                        width, height, op.int_params[0], &scratch_count) != VF_CUDA_OK) {
                    write_reason(reason, reason_capacity, "DAG AdaptiveMean parameters are unsupported");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            }
            case VF_PLAN_MORPHOLOGY:
                if (op.int_params[0] < VF_MORPH_OPEN || op.int_params[0] > VF_MORPH_ERODE ||
                    op.int_params[1] < 3 || op.int_params[1] % 2 == 0 ||
                    op.int_params[2] < 1 || op.int_params[2] > INT_MAX / 2) {
                    write_reason(reason, reason_capacity, "DAG Morphology parameters are unsupported");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            default:
                write_reason(reason, reason_capacity, "DAG contains an unsupported operator kind");
                return VF_CUDA_UNSUPPORTED;
        }
        channels[index] = output_channels;
    }
    std::vector<bool> seen(desc->operator_count, false);
    for (int index = 0; index < desc->output_count; ++index) {
        int node = desc->output_nodes[index];
        if (node < 0 || node >= desc->operator_count || seen[node]) {
            write_reason(reason, reason_capacity, "DAG outputs must be unique existing nodes");
            return VF_CUDA_INVALID_ARGUMENT;
        }
        seen[node] = true;
    }
    if (node_channels != nullptr) *node_channels = std::move(channels);
    write_reason(reason, reason_capacity, "Supported generic native DAG plan");
    return VF_CUDA_OK;
}

int reserve_dag_plan_buffers(PersistentContext* context, const NativeDagPlan& plan) {
    if (context == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    try {
        if (context->dag_u8.size() < plan.operators.size()) {
            context->dag_u8.resize(plan.operators.size(), nullptr);
            context->dag_u8_capacity.resize(plan.operators.size(), 0);
        }
    } catch (const std::bad_alloc&) {
        return VF_CUDA_ALLOCATION_FAILED;
    }
    const size_t pixels = static_cast<size_t>(plan.width) * plan.height;
    bool needs_morph_scratch = false;
    size_t maximum_u32_count = 0;
    for (size_t index = 0; index < plan.operators.size(); ++index) {
        int result = reserve_device(
            &context->dag_u8[index], &context->dag_u8_capacity[index],
            pixels * static_cast<size_t>(plan.node_channels[index]), &context->allocation_count);
        if (result != VF_CUDA_OK) return result;
        const VfPlanOperatorV1& op = plan.operators[index];
        if (op.kind == VF_PLAN_GAUSSIAN) {
            maximum_u32_count = std::max(
                maximum_u32_count, pixels * static_cast<size_t>(plan.node_channels[index]));
        }
        if (op.kind == VF_PLAN_ADAPTIVE_MEAN) {
            maximum_u32_count = std::max(maximum_u32_count, pixels);
        }
        needs_morph_scratch = needs_morph_scratch || op.kind == VF_PLAN_MORPHOLOGY;
        if (op.kind == VF_PLAN_ADAPTIVE_MEAN) {
            size_t scratch_count = 0;
            result = adaptive_layout(plan.width, plan.height, op.int_params[0], &scratch_count);
            if (result != VF_CUDA_OK) return result;
        }
    }
    int result = reserve_device(&context->u8[0], &context->u8_capacity[0],
                                pixels * static_cast<size_t>(plan.input_channels),
                                &context->allocation_count);
    if (result == VF_CUDA_OK && needs_morph_scratch) result = reserve_device(
        &context->u8[4], &context->u8_capacity[4], pixels * 3, &context->allocation_count);
    if (result == VF_CUDA_OK && maximum_u32_count > 0) result = reserve_device(
        &context->gaussian_buffer, &context->gaussian_capacity,
        maximum_u32_count, &context->allocation_count);
    return result;
}

int reserve_plan_buffers(PersistentContext* context, const NativePlan& plan) {
    if (context == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    const size_t input_pixels = static_cast<size_t>(plan.width) * plan.height;
    size_t maximum_pixels = input_pixels;
    int maximum_channels = plan.input_channels;
    bool needs_morph_scratch = false;
    size_t maximum_u32_count = 0;
    int channels = plan.input_channels;
    int current_width = plan.width;
    int current_height = plan.height;
    for (const VfPlanOperatorV1& op : plan.operators) {
        if (op.kind == VF_PLAN_GRAY) channels = 1;
        if (op.kind == VF_PLAN_RESIZE_AREA) {
            current_width = op.int_params[0];
            current_height = op.int_params[1];
        }
        const size_t current_pixels = static_cast<size_t>(current_width) * current_height;
        maximum_pixels = std::max(maximum_pixels, current_pixels);
        maximum_channels = std::max(maximum_channels, channels);
        if (op.kind == VF_PLAN_GAUSSIAN) {
            maximum_u32_count = std::max(
                maximum_u32_count, current_pixels * static_cast<size_t>(channels));
        }
        if (op.kind == VF_PLAN_ADAPTIVE_MEAN) {
            maximum_u32_count = std::max(maximum_u32_count, current_pixels);
        }
        needs_morph_scratch = needs_morph_scratch || op.kind == VF_PLAN_MORPHOLOGY;
        if (op.kind == VF_PLAN_ADAPTIVE_MEAN) {
            size_t scratch_count = 0;
            int result = adaptive_layout(
                current_width, current_height, op.int_params[0], &scratch_count);
            if (result != VF_CUDA_OK) return result;
        }
    }
    const size_t image_bytes = maximum_pixels * static_cast<size_t>(maximum_channels);
    int result = reserve_device(&context->u8[0], &context->u8_capacity[0], image_bytes,
                                &context->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &context->u8[1], &context->u8_capacity[1], image_bytes, &context->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &context->u8[2], &context->u8_capacity[2], image_bytes, &context->allocation_count);
    if (result == VF_CUDA_OK && needs_morph_scratch) result = reserve_device(
        &context->u8[4], &context->u8_capacity[4], image_bytes, &context->allocation_count);
    if (result == VF_CUDA_OK && maximum_u32_count > 0) result = reserve_device(
        &context->gaussian_buffer, &context->gaussian_capacity,
        maximum_u32_count, &context->allocation_count);
    return result;
}

int cuda_result(cudaError_t error) { return visionflow_cuda::runtime_error(error); }

int alloc_copy(const uint8_t* host, int width, int height, int stride, int channels, uint8_t** device) {
    return visionflow_cuda::allocate_and_upload(host, width, height, stride, channels, device);
}

int copy_back_free(uint8_t* host, int stride, int width, int height, int channels, uint8_t* device) {
    return visionflow_cuda::download_and_free(host, stride, width, height, channels, device);
}

__device__ int reflect101(int value, int length) {
    if (length <= 1) return 0;
    while (value < 0 || value >= length) {
        value = value < 0 ? -value : 2 * length - value - 2;
    }
    return value;
}

__global__ void bgr_gray_kernel(const uint8_t* src, uint8_t* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int index = (y * width + x) * 3;
    constexpr int gray_shift = 15;
    constexpr int blue_to_gray = 3735;
    constexpr int green_to_gray = 19235;
    constexpr int red_to_gray = 9798;
    dst[y * width + x] = static_cast<uint8_t>(
        (blue_to_gray * src[index] +
         green_to_gray * src[index + 1] +
         red_to_gray * src[index + 2] +
         (1 << (gray_shift - 1))) >> gray_shift);
}

// Grays a rectangular region of a wider BGR source into a tightly packed single-channel buffer.
// The plain bgr_gray_kernel above assumes a packed source, which a resident-image ROI is not.
__global__ void bgr_gray_roi_kernel(
    const uint8_t* src, int src_width, int offset_x, int offset_y,
    uint8_t* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int index = ((y + offset_y) * src_width + x + offset_x) * 3;
    constexpr int gray_shift = 15;
    constexpr int blue_to_gray = 3735;
    constexpr int green_to_gray = 19235;
    constexpr int red_to_gray = 9798;
    dst[y * width + x] = static_cast<uint8_t>(
        (blue_to_gray * src[index] +
         green_to_gray * src[index + 1] +
         red_to_gray * src[index + 2] +
         (1 << (gray_shift - 1))) >> gray_shift);
}

__global__ void bgr_rgb_kernel(const uint8_t* src, uint8_t* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int i = (y * width + x) * 3;
    dst[i] = src[i + 2]; dst[i + 1] = src[i + 1]; dst[i + 2] = src[i];
}

__global__ void crop_kernel(const uint8_t* src, uint8_t* dst, int src_width, int x0, int y0, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    for (int c = 0; c < channels; ++c) dst[(y * width + x) * channels + c] = src[((y + y0) * src_width + x + x0) * channels + c];
}

// Exact OpenCV INTER_AREA downscale for CV_8UC1. Float accumulation order matches
// ResizeArea_Invoker; the DLL is built with --fmad=false so products are never fused.
__global__ void resize_area_kernel(
    const uint8_t* src, uint8_t* dst, int sw, int dw, int dh, int mode,
    int scale_x, int scale_y, float inverse_area, int x_entries,
    const int* indices, const float* alphas) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    int value = 0;
    if (mode == AREA_RESIZE_COPY) {
        value = src[static_cast<size_t>(y) * sw + x];
    } else if (mode == AREA_RESIZE_FAST_2X2) {
        const uint8_t* row0 = src + static_cast<size_t>(y) * 2 * sw;
        const uint8_t* row1 = row0 + sw;
        const int sx = x * 2;
        value = (row0[sx] + row0[sx + 1] + row1[sx] + row1[sx + 1] + 2) >> 2;
    } else if (mode == AREA_RESIZE_FAST_INTEGER) {
        uint32_t sum = 0;
        for (int sy = 0; sy < scale_y; ++sy) {
            const uint8_t* row = src + (static_cast<size_t>(y) * scale_y + sy) * sw;
            for (int sx = 0; sx < scale_x; ++sx) sum += row[x * scale_x + sx];
        }
        float scaled = static_cast<float>(static_cast<int>(sum)) * inverse_area;
        value = static_cast<int>(nearbyintf(scaled));
    } else {
        const int y_offsets = dw + 1;
        const int x_sources = dw + dh + 2;
        const int y_sources = x_sources + x_entries;
        float total = 0.0f;
        for (int j = indices[y_offsets + y]; j < indices[y_offsets + y + 1]; ++j) {
            const uint8_t* row = src + static_cast<size_t>(indices[y_sources + j]) * sw;
            float row_sum = 0.0f;
            for (int k = indices[x]; k < indices[x + 1]; ++k) {
                float product = static_cast<float>(row[indices[x_sources + k]]) * alphas[k];
                row_sum = row_sum + product;
            }
            float weighted = alphas[x_entries + j] * row_sum;
            total = total + weighted;
        }
        value = static_cast<int>(nearbyintf(total));
    }
    dst[static_cast<size_t>(y) * dw + x] = static_cast<uint8_t>(max(0, min(255, value)));
}

__global__ void resize_gray_kernel(const uint8_t* src, uint8_t* dst, int sw, int sh, int dw, int dh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    // Upscale only; non-expanding targets use resize_area_kernel.
    float sx = (x + 0.5f) * sw / dw - 0.5f, sy = (y + 0.5f) * sh / dh - 0.5f;
    int raw_x0 = static_cast<int>(floorf(sx));
    int raw_y0 = static_cast<int>(floorf(sy));
    int x0 = max(0, min(sw - 1, raw_x0));
    int y0 = max(0, min(sh - 1, raw_y0));
    int x1 = max(0, min(sw - 1, raw_x0 + 1));
    int y1 = max(0, min(sh - 1, raw_y0 + 1));
    float ax = sx - floorf(sx), ay = sy - floorf(sy);
    float value = (1 - ay) * ((1 - ax) * src[y0 * sw + x0] + ax * src[y0 * sw + x1]) + ay * ((1 - ax) * src[y1 * sw + x0] + ax * src[y1 * sw + x1]);
    dst[y * dw + x] = static_cast<uint8_t>(value + 0.5f);
}

__global__ void gaussian_horizontal_kernel(
    const uint8_t* src,
    uint32_t* intermediate,
    int width,
    int height,
    int channels,
    int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const uint8_t* row = src + static_cast<size_t>(y) * width * channels;
    const bool interior = x >= radius && x + radius < width;
    for (int c = 0; c < channels; ++c) {
        uint32_t sum = 0;
        if (interior) {
            const uint8_t* window = row + (x - radius) * channels + c;
            for (int k = 0; k <= 2 * radius; ++k) {
                sum += static_cast<uint32_t>(window[k * channels]) * gaussian_weights[k];
            }
        } else {
            for (int kx = -radius; kx <= radius; ++kx) {
                int sx = reflect101(x + kx, width);
                sum += static_cast<uint32_t>(row[sx * channels + c]) * gaussian_weights[kx + radius];
            }
        }
        intermediate[(static_cast<size_t>(y) * width + x) * channels + c] = sum;
    }
}

__global__ void gaussian_vertical_kernel(
    const uint32_t* intermediate,
    uint8_t* dst,
    int width,
    int height,
    int channels,
    int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const size_t row_step = static_cast<size_t>(width) * channels;
    const bool interior = y >= radius && y + radius < height;
    for (int c = 0; c < channels; ++c) {
        unsigned long long sum = 0;
        if (interior) {
            const uint32_t* window = intermediate + (y - radius) * row_step + x * channels + c;
            for (int k = 0; k <= 2 * radius; ++k) {
                sum += static_cast<unsigned long long>(window[k * row_step]) * gaussian_weights[k];
            }
        } else {
            for (int ky = -radius; ky <= radius; ++ky) {
                int sy = reflect101(y + ky, height);
                sum += static_cast<unsigned long long>(
                    intermediate[sy * row_step + x * channels + c]) * gaussian_weights[ky + radius];
            }
        }
        dst[(static_cast<size_t>(y) * width + x) * channels + c] = static_cast<uint8_t>(
            (sum + GAUSSIAN_FINAL_ROUND) >>
            (GAUSSIAN_FIXED_SHIFT * 2));
    }
}

__global__ void threshold_kernel(const uint8_t* src, uint8_t* dst, int count, int threshold, int max_value, int invert) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    bool high = src[i] > threshold;
    dst[i] = static_cast<uint8_t>((invert ? !high : high) ? max_value : 0);
}

// Exact separable box mean with BORDER_REPLICATE. A block scans each source row into a uint32
// prefix plane; one thread per output pixel accumulates its vertical window from those prefixes.
// Adjacent threads read adjacent row-prefix values, preserving coalescing without a second plane.
// The row prefix fits uint32 under adaptive_layout's width limit and the box sum remains uint64 so
// block*block*255 cannot overflow. This replaces padded pixels plus two padded uint64 integral
// planes with one width*height uint32 plane while retaining parallel horizontal work.
__global__ void adaptive_row_prefix_u32_kernel(
    const uint8_t* src,
    uint32_t* prefix,
    int width,
    int height) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    if (row >= height) return;
    using BlockScan = cub::BlockScan<uint32_t, ADAPTIVE_SCAN_THREADS>;
    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ uint32_t carry;
    __shared__ uint32_t chunk_carry;
    if (lane == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < width; base += ADAPTIVE_SCAN_THREADS) {
        int column = base + lane;
        uint32_t value = column < width ? src[static_cast<size_t>(row) * width + column] : 0U;
        uint32_t scanned = 0;
        uint32_t aggregate = 0;
        BlockScan(scan_storage).InclusiveSum(value, scanned, aggregate);
        if (lane == 0) chunk_carry = carry;
        __syncthreads();
        if (column < width) {
            prefix[static_cast<size_t>(row) * width + column] = scanned + chunk_carry;
        }
        __syncthreads();
        if (lane == 0) carry = chunk_carry + aggregate;
        __syncthreads();
    }
}

__device__ __forceinline__ uint32_t adaptive_horizontal_box_sum(
    const uint8_t* src,
    const uint32_t* prefix,
    int width,
    int row,
    int x,
    int radius) {
    int left = x - radius;
    int right = x + radius;
    int clamped_left = max(0, left);
    int clamped_right = min(width - 1, right);
    size_t row_offset = static_cast<size_t>(row) * width;
    uint32_t sum = prefix[row_offset + clamped_right];
    if (clamped_left > 0) sum -= prefix[row_offset + clamped_left - 1];
    if (left < 0) {
        sum += static_cast<uint32_t>(-left) * src[row_offset];
    }
    if (right >= width) {
        sum += static_cast<uint32_t>(right - width + 1) * src[row_offset + width - 1];
    }
    return sum;
}

__global__ void adaptive_vertical_threshold_kernel(
    const uint8_t* src,
    const uint32_t* prefix,
    uint8_t* dst,
    int width,
    int height,
    int block_size,
    float c,
    int max_value,
    int invert) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int radius = block_size / 2;
    unsigned long long sum = 0;
    for (int offset = -radius; offset <= radius; ++offset) {
        int row = max(0, min(height - 1, y + offset));
        sum += adaptive_horizontal_box_sum(src, prefix, width, row, x, radius);
    }
    unsigned long long area = static_cast<unsigned long long>(block_size) * block_size;
    int mean = static_cast<int>((sum + area / 2ULL) / area);
    bool selected = invert
        ? static_cast<int>(src[static_cast<size_t>(y) * width + x]) <=
            mean - static_cast<int>(floorf(c))
        : static_cast<int>(src[static_cast<size_t>(y) * width + x]) >
            mean - static_cast<int>(ceilf(c));
    dst[static_cast<size_t>(y) * width + x] =
        static_cast<uint8_t>(selected ? max_value : 0);
}

void launch_adaptive_mean(
    const uint8_t* src,
    uint32_t* prefix,
    uint8_t* dst,
    int width,
    int height,
    int block_size,
    float c,
    int max_value,
    int invert,
    cudaStream_t stream = nullptr) {
    constexpr int threads = ADAPTIVE_SCAN_THREADS;
    adaptive_row_prefix_u32_kernel<<<height, threads, 0, stream>>>(
        src, prefix, width, height);
    constexpr int vertical_block_x = 32;
    constexpr int vertical_block_y = 8;
    dim3 block(vertical_block_x, vertical_block_y);
    dim3 grid(
        (width + vertical_block_x - 1) / vertical_block_x,
        (height + vertical_block_y - 1) / vertical_block_y);
    adaptive_vertical_threshold_kernel<<<grid, block, 0, stream>>>(
        src, prefix, dst, width, height, block_size, c, max_value, invert);
}

__global__ void morph_kernel(const uint8_t* src, uint8_t* dst, int width, int height, int channels, int radius, int dilate) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    for (int c = 0; c < channels; ++c) {
        int value = dilate ? 0 : 255;
        for (int ky = -radius; ky <= radius; ++ky) for (int kx = -radius; kx <= radius; ++kx) {
            int sx = x + kx, sy = y + ky;
            int sample = (sx < 0 || sx >= width || sy < 0 || sy >= height) ? (dilate ? 0 : 255) : src[(sy * width + sx) * channels + c];
            value = dilate ? max(value, sample) : min(value, sample);
        }
        dst[(y * width + x) * channels + c] = static_cast<uint8_t>(value);
    }
}

// A rectangular min/max filter is separable. Keep both directions in one
// block so a 5x5 pass needs only one launch and no device scratch image.
__global__ void morph_k5_shared_kernel(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels, int dilate) {
    constexpr int radius = 2;
    constexpr int tile_width = BLOCK_X + 2 * radius;
    constexpr int tile_height = BLOCK_Y + 2 * radius;
    __shared__ uint8_t tile[tile_width * tile_height * 3];
    __shared__ uint8_t horizontal[tile_height * BLOCK_X * 3];
    const int thread_index = threadIdx.y * BLOCK_X + threadIdx.x;
    const int thread_count = BLOCK_X * BLOCK_Y;
    const uint8_t border = static_cast<uint8_t>(dilate ? 0 : 255);

    for (int index = thread_index; index < tile_width * tile_height * channels;
         index += thread_count) {
        const int channel = index % channels;
        const int pixel = index / channels;
        const int x = blockIdx.x * BLOCK_X + pixel % tile_width - radius;
        const int y = blockIdx.y * BLOCK_Y + pixel / tile_width - radius;
        tile[index] = (x < 0 || x >= width || y < 0 || y >= height)
            ? border : src[(y * width + x) * channels + channel];
    }
    __syncthreads();

    for (int index = thread_index; index < tile_height * BLOCK_X * channels;
         index += thread_count) {
        const int channel = index % channels;
        const int pixel = index / channels;
        const int x = pixel % BLOCK_X + radius;
        const int y = pixel / BLOCK_X;
        int value = dilate ? 0 : 255;
        for (int dx = -radius; dx <= radius; ++dx) {
            const int sample = tile[(y * tile_width + x + dx) * channels + channel];
            value = dilate ? max(value, sample) : min(value, sample);
        }
        horizontal[index] = static_cast<uint8_t>(value);
    }
    __syncthreads();

    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= width || y >= height) return;
    for (int channel = 0; channel < channels; ++channel) {
        int value = dilate ? 0 : 255;
        for (int dy = -radius; dy <= radius; ++dy) {
            const int sample = horizontal[
                ((threadIdx.y + dy + radius) * BLOCK_X + threadIdx.x) * channels + channel];
            value = dilate ? max(value, sample) : min(value, sample);
        }
        dst[(y * width + x) * channels + channel] = static_cast<uint8_t>(value);
    }
}

dim3 grid2d(int width, int height);

void launch_morph_pass(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    int radius, int dilate, cudaStream_t stream = nullptr) {
    if (radius == 2) {
        morph_k5_shared_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
            src, dst, width, height, channels, dilate);
    } else {
        morph_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
            src, dst, width, height, channels, radius, dilate);
    }
}

// Separable Gaussian. A block-local shared-memory tile variant produced identical output but
// was about 2x slower on RTX 3090 (2026-09-14), so both passes read global memory directly.
void launch_gaussian(
    const uint8_t* src, uint32_t* intermediate, uint8_t* dst, int width, int height,
    int channels, int radius, cudaStream_t stream = nullptr) {
    gaussian_horizontal_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
        src, intermediate, width, height, channels, radius);
    gaussian_vertical_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
        intermediate, dst, width, height, channels, radius);
}

// float32 separable Gaussian for the optional vf_gaussian_blur_f32 export. Single channel only:
// the caller passes the detector's float32 gray/residual plane. Taps are read from the constant
// coefficient table prepared by prepare_gaussian_f32_weights(), and both passes accumulate in
// float32 with reflect101 borders, which is what cv2.GaussianBlur(single_channel_float32, ksize,
// 0.0) does internally. The build disables FMA contraction (/fmad=false), so the result is a pure
// function of the tap order documented here and of the input bytes.
//
// Horizontal pass: taps are accumulated left to right, the order OpenCV's row filter uses.
__global__ void gaussian_f32_horizontal_kernel(
    const float* src, float* dst, int width, int height, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const float* row = src + static_cast<size_t>(y) * width;
    float sum = 0.0f;
    if (x >= radius && x + radius < width) {
        const float* window = row + (x - radius);
        for (int k = 0; k <= 2 * radius; ++k) {
            sum += window[k] * gaussian_f32_weights[k];
        }
    } else {
        for (int kx = -radius; kx <= radius; ++kx) {
            sum += row[reflect101(x + kx, width)] * gaussian_f32_weights[kx + radius];
        }
    }
    dst[static_cast<size_t>(y) * width + x] = sum;
}

// Vertical pass: symmetric taps are added as a pair before the multiply, the order OpenCV's
// symmetric column filter uses. Measured against cv2.GaussianBlur this is closer than a plain
// left-to-right sum for every kernel size in the verified range (see the equivalence tool).
__global__ void gaussian_f32_vertical_kernel(
    const float* src, float* dst, int width, int height, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const std::ptrdiff_t step = static_cast<std::ptrdiff_t>(width);
    float sum = 0.0f;
    if (y >= radius && y + radius < height) {
        const float* centre = src + static_cast<size_t>(y) * width + x;
        sum = centre[0] * gaussian_f32_weights[radius];
        for (int k = 1; k <= radius; ++k) {
            const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(k) * step;
            sum += (centre[-offset] + centre[offset]) * gaussian_f32_weights[radius + k];
        }
    } else {
        sum = src[static_cast<size_t>(y) * width + x] * gaussian_f32_weights[radius];
        for (int k = 1; k <= radius; ++k) {
            const float top = src[static_cast<size_t>(reflect101(y - k, height)) * width + x];
            const float bottom = src[static_cast<size_t>(reflect101(y + k, height)) * width + x];
            sum += (top + bottom) * gaussian_f32_weights[radius + k];
        }
    }
    dst[static_cast<size_t>(y) * width + x] = sum;
}

void launch_gaussian_f32(
    const float* src, float* intermediate, float* dst, int width, int height,
    int radius, cudaStream_t stream) {
    gaussian_f32_horizontal_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
        src, intermediate, width, height, radius);
    gaussian_f32_vertical_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
        intermediate, dst, width, height, radius);
}

__global__ void gather_roi_batch_kernel(
    const uint8_t* source,
    int source_width,
    int channels,
    const VfRoiV1* rois,
    uint8_t* batch,
    int roi_width,
    int roi_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int roi_index = blockIdx.z;
    if (x >= roi_width || y >= roi_height) return;
    const VfRoiV1 roi = rois[roi_index];
    size_t source_pixel =
        (static_cast<size_t>(roi.y + y) * source_width + roi.x + x) * channels;
    size_t batch_pixel =
        ((static_cast<size_t>(roi_index) * roi_height + y) * roi_width + x) * channels;
    for (int channel = 0; channel < channels; ++channel) {
        batch[batch_pixel + channel] = source[source_pixel + channel];
    }
}

dim3 grid2d(int width, int height) { return dim3((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y); }

__global__ void flip_vertical_u8_in_place_kernel(
    uint8_t* image, int row_bytes, int height) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int top = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= row_bytes || top >= height / 2) return;
    int bottom = height - 1 - top;
    uint8_t value = image[static_cast<size_t>(top) * row_bytes + column];
    image[static_cast<size_t>(top) * row_bytes + column] =
        image[static_cast<size_t>(bottom) * row_bytes + column];
    image[static_cast<size_t>(bottom) * row_bytes + column] = value;
}

// --- Template Anchor Grid localization (TM_CCOEFF_NORMED) -------------------------------
// Reproduces core/tiler.py::Tiler._find_grid_anchor: the same correlation formula and the same
// "topmost then leftmost wins a tie" order as the CPU argmax over the result map. Only the
// match rectangle and its score cross PCIe.
//
// Two vertical prefixes per output column give the window pixel sum and the window square sum in
// O(1) each. The template-weighted window sum is accumulated directly by the score kernel:
//   num = sum(w*t) - N*mean_w*mean_t
//   den = sqrt((sum(w^2) - N*mean_w^2) * (sum(t^2) - N*mean_t^2))
constexpr int MATCH_TEMPLATE_BUFFER = 2;    // context->u8[2]: template gray on device
constexpr int MATCH_ROI_BUFFER = 3;         // context->u8[3]: search ROI gray on device
constexpr int MATCH_REDUCE_BLOCK = 256;
// Coordinates are packed into 20 bits, so a search wider than this cannot be reported at all.
constexpr int MATCH_MAX_OUTPUT_WIDTH = (1 << 20) - 1;
// Output tile of the tiled score kernel and its block shape. Shared memory per block is
// (tile_rows + template_height - 1) * (tile_cols + template_width - 1) bytes, so the caller
// shrinks the tile when a large template would exceed the device limit.
constexpr int MATCH_TILE_COLS = 64;
constexpr int MATCH_TILE_ROWS = 32;
constexpr int MATCH_BLOCK_X = 64;
constexpr int MATCH_BLOCK_Y = 4;
// sm_86 allows up to 100 KiB of dynamic shared memory per block once opted in, and the export
// opts in for every launch it makes. The tile shrink loop stays within this budget.
constexpr size_t MATCH_SHARED_LIMIT_BYTES = 99 * 1024;
// Candidate slots scale with the search width, not the block size: the score kernel writes one
// entry per output column, so a fixed small count would overflow as soon as output_width exceeds
// it. Slots are int64-sized so the double scores stay aligned, and the result fields follow them.
// Layout, all indexed by column count:
//   [0, W)      packed best keys (one per output column, int64, atomic target)
//   [W, 2W)     candidate score (double, one per output column)
//   [2W, 3W)    candidate row (int, one per output column, padded to int64 stride)
//   then        the two coordinate ints, the float score and the global best key
constexpr int MATCH_CANDIDATE_SLOT_STRIDE = 3;
constexpr int MATCH_RESULT_SLOTS = 3;
constexpr int MATCH_FIXED_SLOTS = MATCH_RESULT_SLOTS + 1;

int match_candidate_slots(int output_width) {
    return output_width > 0 ? output_width : 1;
}

int match_slot_count(int output_width) {
    return match_candidate_slots(output_width) * MATCH_CANDIDATE_SLOT_STRIDE + MATCH_FIXED_SLOTS;
}

int match_score_offset(int output_width) {
    return match_candidate_slots(output_width);
}

int match_row_offset(int output_width) {
    return match_candidate_slots(output_width) * 2;
}

int match_result_offset(int output_width) {
    return match_candidate_slots(output_width) * MATCH_CANDIDATE_SLOT_STRIDE;
}
// Match key layout, ordered so that a larger unsigned key is the better match:
//   bits 62..40 score, bits 39..20 inverted y, bits 19..0 inverted x.
// The score is quantized to 22 bits once, and that same value is what the caller receives, so a
// tie in the packed key is exactly a tie in the reported score. Coordinates must fit 20 bits.
constexpr int MATCH_SCORE_BITS = 22;
constexpr double MATCH_SCORE_MAX = static_cast<double>((1 << MATCH_SCORE_BITS) - 1);
constexpr int MATCH_COORD_BITS = 20;
constexpr int MATCH_COORD_MASK = (1 << MATCH_COORD_BITS) - 1;

__device__ __forceinline__ unsigned long long match_pack_key(double score, int x, int y) {
    double clamped = score;
    if (clamped > 1.0) clamped = 1.0;
    if (clamped < 0.0) clamped = 0.0;
    const unsigned long long quantized =
        static_cast<unsigned long long>(clamped * MATCH_SCORE_MAX + 0.5);
    return (quantized << 40) |
           (static_cast<unsigned long long>(MATCH_COORD_MASK - y) << 20) |
           static_cast<unsigned long long>(MATCH_COORD_MASK - x);
}

__device__ __forceinline__ void match_unpack_key(
    unsigned long long key, double* score, int* x, int* y) {
    *x = MATCH_COORD_MASK - static_cast<int>(key & MATCH_COORD_MASK);
    *y = MATCH_COORD_MASK - static_cast<int>((key >> 20) & MATCH_COORD_MASK);
    const unsigned long long quantized = (key >> 40) & ((1ULL << MATCH_SCORE_BITS) - 1);
    *score = static_cast<double>(quantized) / MATCH_SCORE_MAX;
}

// One thread per output column. Writes the vertical prefixes of the horizontal window sum and of
// its square, so the score kernel obtains the window pixel sum and the window square sum with two
// subtractions each. The horizontal window for output (row, column) starts at image row `row`.
__global__ void match_prefix_kernel(
    const uint8_t* roi, int roi_width, int roi_height,
    int output_width, int template_width,
    long long* sum_prefix, long long* square_prefix) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= output_width) return;
    long long sum_running = 0;
    long long square_running = 0;
    for (int row = threadIdx.y; row < roi_height; row += blockDim.y) {
        const uint8_t* line = roi + static_cast<size_t>(row) * roi_width;
        int sum = 0;
        long long square_sum = 0;
        for (int offset = 0; offset < template_width; ++offset) {
            const int value = line[column + offset];
            sum += value;
            square_sum += static_cast<long long>(value) * value;
        }
        sum_running += sum;
        square_running += square_sum;
        const size_t index = static_cast<size_t>(row) * output_width + column;
        sum_prefix[index] = sum_running;
        square_prefix[index] = square_running;
    }
}

// Offers one candidate for a column through a dedicated packed-key slot. The key is
// (quantized score, inverted row, inverted column), so a larger key is a better match and a tie
// resolves to the smaller row and then the smaller column - the same order as the CPU argmax over
// the match map. Publishing the whole triple with one compare-and-swap on its own array keeps the
// winner independent of block and thread order; the score array is never used as the atomic.
__device__ __forceinline__ void match_offer_candidate(
    double score, int column, int row, unsigned long long* best_keys) {
    double clamped = score;
    if (clamped > 1.0) clamped = 1.0;
    if (clamped < 0.0) clamped = 0.0;
    const unsigned long long quantized =
        static_cast<unsigned long long>(clamped * MATCH_SCORE_MAX + 0.5);
    const int safe_row = row > MATCH_COORD_MASK - 1 ? MATCH_COORD_MASK - 1 : row;
    const int safe_column = column > MATCH_COORD_MASK - 1 ? MATCH_COORD_MASK - 1 : column;
    const unsigned long long key =
        (quantized << 40) |
        (static_cast<unsigned long long>(MATCH_COORD_MASK - safe_row) << 20) |
        static_cast<unsigned long long>(MATCH_COORD_MASK - safe_column);
    unsigned long long* slot = best_keys + column;
    unsigned long long current = *slot;
    while (key > current) {
        const unsigned long long previous = atomicCAS(slot, current, key);
        if (previous == current) break;
        current = previous;
    }
}

// One thread per output row within a column band. The template is staged in shared memory, because
// it is the small, constant, every-candidate operand; the ROI stays in global memory and each row
// the thread reads is reused across all template rows it participates in, so the cost per output
// is one read of a template_width strip plus one multiply-add chain. Consecutive threads handle
// consecutive output rows, so both the ROI loads and the template loads coalesce (the warp shares
// one template row and reads a contiguous ROI patch).
__global__ void match_score_shared_template_kernel(
    const uint8_t* roi, int roi_width,
    int output_width, int output_height,
    int template_width, int template_height, int template_pixels,
    int column_band,
    double template_mean, double template_variance,
    const uint8_t* templ, unsigned long long* best_keys) {
    extern __shared__ unsigned char shared_bytes[];
    uint8_t* shared_template = shared_bytes;

    for (int index = threadIdx.y * blockDim.x + threadIdx.x;
         index < template_width * template_height; index += blockDim.x * blockDim.y) {
        shared_template[index] = templ[index];
    }
    __syncthreads();

    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * column_band + threadIdx.x;
    if (column >= output_width || output_row >= output_height) return;

    long long window_sum = 0;
    long long window_square = 0;
    long long weighted = 0;
    for (int template_row = 0; template_row < template_height; ++template_row) {
        const uint8_t* image_line =
            roi + static_cast<size_t>(output_row + template_row) * roi_width + column;
        const uint8_t* template_line =
            shared_template + static_cast<size_t>(template_row) * template_width;
        long long term = 0;
        for (int offset = 0; offset < template_width; ++offset) {
            const int value = image_line[offset];
            window_sum += value;
            window_square += static_cast<long long>(value) * value;
            term += static_cast<long long>(value) * template_line[offset];
        }
        weighted += term;
    }
    const double mean = static_cast<double>(window_sum) / template_pixels;
    double window_variance = static_cast<double>(window_square) / template_pixels - mean * mean;
    if (window_variance < 0.0) window_variance = 0.0;
    const double denominator = std::sqrt(window_variance * template_variance) * template_pixels;
    double score = -1.0;
    if (denominator > 0.0) {
        score = (static_cast<double>(weighted) - template_mean * window_sum) / denominator;
    }
    if (score > 1.0) score = 1.0;
    if (score < -1.0) score = -1.0;
    // Several output rows of the same column are in flight at once, so publish through the same
    // packed-key compare-and-swap the tiled kernel uses; the ordering is identical, so the winner
    // does not depend on which block ran first.
    match_offer_candidate(score, column, output_row, best_keys);
}

// Pattern Match uses the same TM_CCOEFF_NORMED arithmetic as the anchor path, but retains the
// complete response plane so local-peak selection and NMS can remain on the device.
__global__ void pattern_score_map_kernel(
    const uint8_t* roi, int roi_width,
    int output_width, int output_height,
    int template_width, int template_height, int template_pixels,
    double template_mean, double template_variance,
    const uint8_t* templ, float* scores) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= output_width || row >= output_height) return;
    long long window_sum = 0;
    long long window_square = 0;
    long long weighted = 0;
    for (int template_row = 0; template_row < template_height; ++template_row) {
        const uint8_t* image_line =
            roi + static_cast<size_t>(row + template_row) * roi_width + column;
        const uint8_t* template_line = templ + static_cast<size_t>(template_row) * template_width;
        for (int offset = 0; offset < template_width; ++offset) {
            const int value = image_line[offset];
            window_sum += value;
            window_square += static_cast<long long>(value) * value;
            weighted += static_cast<long long>(value) * template_line[offset];
        }
    }
    const double mean = static_cast<double>(window_sum) / template_pixels;
    double window_variance =
        static_cast<double>(window_square) / template_pixels - mean * mean;
    if (window_variance < 0.0) window_variance = 0.0;
    const double denominator =
        std::sqrt(window_variance * template_variance) * template_pixels;
    double score = -1.0;
    if (denominator > 0.0) {
        score = (static_cast<double>(weighted) - template_mean * window_sum) / denominator;
    }
    score = score > 1.0 ? 1.0 : (score < -1.0 ? -1.0 : score);
    scores[static_cast<size_t>(row) * output_width + column] = static_cast<float>(score);
}

// ---- Large-template Pattern Match ------------------------------------------------------------
// pattern_score_map_kernel costs output_elements x template_pixels, which a production
// 2000x12000 template on a 16384x13000 frame turns into 3.5e14 multiply-adds. The kernels below
// produce the same TM_CCOEFF_NORMED response from an FFT cross correlation (the numerator) and
// exact int64 summed-area tables (the window statistics). The tables are what keep this path
// faithful: OpenCV's own CUDA TemplateMatching normalizes in float32 and, measured at this size,
// disagrees with its CPU reference by up to 0.9, while this split stays near 1e-5.
constexpr int PATTERN_SAT_SCAN_THREADS = 256;
constexpr float FFT_PI = 3.14159265358979323846f;
// The FFT response costs what the frame costs, while the brute-force kernel costs what the
// template costs, so the crossover belongs per frame pixel. RTX 3090 sweep in
// gpu/validate_pattern_match_fft.py: brute force sustains about 4.3e11 multiply-adds per second
// and the FFT path costs about 1.5 ns per frame pixel, which breaks even near 650 multiply-adds
// per frame pixel. Six times that keeps small templates on the brute-force path, which needs no
// transforms or summed-area tables, until the FFT is clearly the faster route.
constexpr double PATTERN_FFT_CROSSOVER_WORK_PER_PIXEL = 4000.0;

// One block per frame row: inclusive row prefixes of the pixel values and of their squares.
__global__ void pattern_sat_rows_kernel(
    const uint8_t* gray, int width, int height,
    long long* sum_plane, long long* square_plane) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= height) return;
    using BlockScan = cub::BlockScan<long long, PATTERN_SAT_SCAN_THREADS>;
    __shared__ typename BlockScan::TempStorage sum_storage;
    __shared__ typename BlockScan::TempStorage square_storage;
    __shared__ long long sum_carry;
    __shared__ long long square_carry;
    __shared__ long long sum_chunk_carry;
    __shared__ long long square_chunk_carry;
    if (lane == 0) {
        sum_carry = 0;
        square_carry = 0;
    }
    __syncthreads();
    for (int base = 0; base < width; base += PATTERN_SAT_SCAN_THREADS) {
        const int column = base + lane;
        const long long value = column < width
            ? static_cast<long long>(gray[static_cast<size_t>(row) * width + column])
            : 0LL;
        long long sum_scanned = 0;
        long long sum_aggregate = 0;
        long long square_scanned = 0;
        long long square_aggregate = 0;
        BlockScan(sum_storage).InclusiveSum(value, sum_scanned, sum_aggregate);
        BlockScan(square_storage).InclusiveSum(value * value, square_scanned, square_aggregate);
        if (lane == 0) {
            sum_chunk_carry = sum_carry;
            square_chunk_carry = square_carry;
        }
        __syncthreads();
        if (column < width) {
            const size_t index = static_cast<size_t>(row) * width + column;
            sum_plane[index] = sum_scanned + sum_chunk_carry;
            square_plane[index] = square_scanned + square_chunk_carry;
        }
        __syncthreads();
        if (lane == 0) {
            sum_carry = sum_chunk_carry + sum_aggregate;
            square_carry = square_chunk_carry + square_aggregate;
        }
        __syncthreads();
    }
}

// One thread per column turns the row prefixes into a full summed-area table in place. Adjacent
// threads walk adjacent columns, so every row of the sweep is coalesced.
__global__ void pattern_sat_columns_kernel(
    long long* sum_plane, long long* square_plane, int width, int height) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= width) return;
    long long sum_running = 0;
    long long square_running = 0;
    for (int row = 0; row < height; ++row) {
        const size_t index = static_cast<size_t>(row) * width + column;
        sum_running += sum_plane[index];
        square_running += square_plane[index];
        sum_plane[index] = sum_running;
        square_plane[index] = square_running;
    }
}

// Copies a u8 plane into the zero-padded complex buffer the forward transform consumes. The
// caller zeroes the buffer first, which is what makes the circular correlation linear on the
// valid region.
__global__ void pattern_pad_u8_kernel(
    const uint8_t* src, int src_width, int src_height, int src_pitch,
    float2* dst, int pad_width) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= src_width || row >= src_height) return;
    dst[static_cast<size_t>(row) * pad_width + column] =
        make_float2(static_cast<float>(src[static_cast<size_t>(row) * src_pitch + column]), 0.0f);
}

// ---- Stockham autosort FFT --------------------------------------------------------------------
// Batched power-of-two transforms over contiguous rows, written so the column direction is served
// by transposing and reusing the same kernels. One thread owns one butterfly: it reads with a
// fixed stride and scatters its outputs, which is what makes the transform self-sorting (no
// separate bit-reversal pass). `ns` is the size of the sub-transforms already completed.
// `sign` is -1 for the forward transform and +1 for the inverse; the 1/N scaling is folded into
// the correlation kernel. The index arithmetic is the formulation verified against numpy.fft
// before this was written; see the completion record for 2026-09-23.
__device__ __forceinline__ float2 complex_multiply(float2 left, float2 right) {
    return make_float2(
        left.x * right.x - left.y * right.y,
        left.x * right.y + left.y * right.x);
}

__device__ __forceinline__ float2 complex_twiddle(float angle) {
    // The accurate sincosf, not the __sincosf intrinsic: twiddle error feeds every later stage.
    float sine = 0.0f;
    float cosine = 0.0f;
    sincosf(angle, &sine, &cosine);
    return make_float2(cosine, sine);
}

__global__ void fft_stage_radix2_kernel(
    const float2* src, float2* dst, int n, int ns, int batch, float sign) {
    const int half = n >> 1;
    const long long thread = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (thread >= static_cast<long long>(half) * batch) return;
    const int index = static_cast<int>(thread % half);
    const size_t offset = static_cast<size_t>(thread / half) * n;
    const int j = index & (ns - 1);
    const int out = ((index - j) << 1) + j;
    const float2 twiddle = complex_twiddle(sign * FFT_PI * static_cast<float>(j) / ns);
    const float2 a = src[offset + index];
    const float2 b = complex_multiply(src[offset + index + half], twiddle);
    dst[offset + out] = make_float2(a.x + b.x, a.y + b.y);
    dst[offset + out + ns] = make_float2(a.x - b.x, a.y - b.y);
}

__global__ void fft_stage_radix4_kernel(
    const float2* src, float2* dst, int n, int ns, int batch, float sign) {
    const int quarter = n >> 2;
    const long long thread = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (thread >= static_cast<long long>(quarter) * batch) return;
    const int index = static_cast<int>(thread % quarter);
    const size_t offset = static_cast<size_t>(thread / quarter) * n;
    const int j = index & (ns - 1);
    const int out = ((index - j) << 2) + j;
    const float base = sign * 2.0f * FFT_PI * static_cast<float>(j) / (4.0f * ns);
    const float2 a0 = src[offset + index];
    const float2 a1 = complex_multiply(src[offset + index + quarter], complex_twiddle(base));
    const float2 a2 = complex_multiply(src[offset + index + 2 * quarter], complex_twiddle(2.0f * base));
    const float2 a3 = complex_multiply(src[offset + index + 3 * quarter], complex_twiddle(3.0f * base));
    const float2 t0 = make_float2(a0.x + a2.x, a0.y + a2.y);
    const float2 t1 = make_float2(a0.x - a2.x, a0.y - a2.y);
    const float2 t2 = make_float2(a1.x + a3.x, a1.y + a3.y);
    float2 t3 = make_float2(a1.x - a3.x, a1.y - a3.y);
    // Multiply by -i for the forward transform and by +i for the inverse.
    t3 = make_float2(-sign * t3.y, sign * t3.x);
    dst[offset + out] = make_float2(t0.x + t2.x, t0.y + t2.y);
    dst[offset + out + ns] = make_float2(t1.x + t3.x, t1.y + t3.y);
    dst[offset + out + 2 * ns] = make_float2(t0.x - t2.x, t0.y - t2.y);
    dst[offset + out + 3 * ns] = make_float2(t1.x - t3.x, t1.y - t3.y);
}

// Tiled transpose so the column transform can reuse the row kernels with coalesced access.
constexpr int FFT_TRANSPOSE_TILE = 32;
constexpr int FFT_TRANSPOSE_ROWS = 8;

__global__ void fft_transpose_kernel(const float2* src, float2* dst, int width, int height) {
    __shared__ float2 tile[FFT_TRANSPOSE_TILE][FFT_TRANSPOSE_TILE + 1];
    const int x = blockIdx.x * FFT_TRANSPOSE_TILE + threadIdx.x;
    const int y = blockIdx.y * FFT_TRANSPOSE_TILE + threadIdx.y;
    for (int offset = 0; offset < FFT_TRANSPOSE_TILE; offset += FFT_TRANSPOSE_ROWS) {
        if (x < width && y + offset < height) {
            tile[threadIdx.y + offset][threadIdx.x] =
                src[static_cast<size_t>(y + offset) * width + x];
        }
    }
    __syncthreads();
    const int transposed_x = blockIdx.y * FFT_TRANSPOSE_TILE + threadIdx.x;
    const int transposed_y = blockIdx.x * FFT_TRANSPOSE_TILE + threadIdx.y;
    for (int offset = 0; offset < FFT_TRANSPOSE_TILE; offset += FFT_TRANSPOSE_ROWS) {
        if (transposed_x < height && transposed_y + offset < width) {
            dst[static_cast<size_t>(transposed_y + offset) * height + transposed_x] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

// image = image * conj(template) * scale. The inverse transform turns this into the correlation
// plane, with the scale folding the unnormalized round trip into the same pass.
__global__ void pattern_spectrum_correlate_kernel(
    float2* image_spectrum, const float2* template_spectrum, size_t count, float scale) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float2 image = image_spectrum[index];
    const float2 templ = template_spectrum[index];
    float2 product;
    product.x = (image.x * templ.x + image.y * templ.y) * scale;
    product.y = (image.y * templ.x - image.x * templ.y) * scale;
    image_spectrum[index] = product;
}

// The TM_CCOEFF_NORMED definition of pattern_score_map_kernel, with the numerator read from the
// correlation plane and the window sums taken from the summed-area tables in four lookups.
__global__ void pattern_fft_score_kernel(
    const float2* correlation, int pad_width,
    const long long* sat_sum, const long long* sat_square, int frame_width,
    int output_width, int output_height,
    int template_width, int template_height, int template_pixels,
    double template_mean, double template_variance, float* scores) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column >= output_width || row >= output_height) return;
    const size_t bottom_row = static_cast<size_t>(row + template_height - 1) * frame_width;
    const int right = column + template_width - 1;
    long long window_sum = sat_sum[bottom_row + right];
    long long window_square = sat_square[bottom_row + right];
    if (row > 0) {
        const size_t top_row = static_cast<size_t>(row - 1) * frame_width;
        window_sum -= sat_sum[top_row + right];
        window_square -= sat_square[top_row + right];
        if (column > 0) {
            window_sum += sat_sum[top_row + column - 1];
            window_square += sat_square[top_row + column - 1];
        }
    }
    if (column > 0) {
        window_sum -= sat_sum[bottom_row + column - 1];
        window_square -= sat_square[bottom_row + column - 1];
    }
    const double mean = static_cast<double>(window_sum) / template_pixels;
    double window_variance = static_cast<double>(window_square) / template_pixels - mean * mean;
    if (window_variance < 0.0) window_variance = 0.0;
    const double denominator = std::sqrt(window_variance * template_variance) * template_pixels;
    double score = -1.0;
    if (denominator > 0.0) {
        const double weighted =
            static_cast<double>(correlation[static_cast<size_t>(row) * pad_width + column].x);
        score = (weighted - template_mean * static_cast<double>(window_sum)) / denominator;
    }
    score = score > 1.0 ? 1.0 : (score < -1.0 ? -1.0 : score);
    scores[static_cast<size_t>(row) * output_width + column] = static_cast<float>(score);
}

__device__ __forceinline__ uint32_t pattern_float_order_key(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
}

__device__ __forceinline__ float pattern_float_from_order_key(uint32_t ordered) {
    const uint32_t bits = (ordered & 0x80000000U) ? (ordered ^ 0x80000000U) : ~ordered;
    return __uint_as_float(bits);
}

__device__ __forceinline__ void pattern_unpack_key(
    unsigned long long key, float* score, int* x, int* y) {
    *score = pattern_float_from_order_key(static_cast<uint32_t>(key >> 32));
    *y = 0xffff - static_cast<int>((key >> 16) & 0xffffULL);
    *x = 0xffff - static_cast<int>(key & 0xffffULL);
}

__global__ void pattern_local_peak_keys_kernel(
    const float* scores, int width, int height,
    int kernel_width, int kernel_height, float threshold,
    unsigned long long* keys) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const size_t index = static_cast<size_t>(y) * width + x;
    const float value = scores[index];
    if (value < threshold) {
        keys[index] = 0;
        return;
    }
    const int anchor_x = kernel_width / 2;
    const int anchor_y = kernel_height / 2;
    const int first_x = max(0, x - anchor_x);
    const int last_x = min(width - 1, x + kernel_width - anchor_x - 1);
    const int first_y = max(0, y - anchor_y);
    const int last_y = min(height - 1, y + kernel_height - anchor_y - 1);
    for (int row = first_y; row <= last_y; ++row) {
        const float* line = scores + static_cast<size_t>(row) * width;
        for (int column = first_x; column <= last_x; ++column) {
            if (line[column] > value) {
                keys[index] = 0;
                return;
            }
        }
    }
    keys[index] =
        (static_cast<unsigned long long>(pattern_float_order_key(value)) << 32) |
        (static_cast<unsigned long long>(0xffff - y) << 16) |
        static_cast<unsigned long long>(0xffff - x);
}

__device__ __forceinline__ bool pattern_iou_passes(
    int ax, int ay, int bx, int by, int width, int height, float threshold) {
    const int intersection_width = max(0, min(ax + width, bx + width) - max(ax, bx));
    const int intersection_height = max(0, min(ay + height, by + height) - max(ay, by));
    const long long intersection =
        static_cast<long long>(intersection_width) * intersection_height;
    const long long area = static_cast<long long>(width) * height;
    const long long union_area = area * 2 - intersection;
    const float iou = union_area > 0 ? static_cast<float>(intersection) / union_area : 0.0f;
    return iou <= threshold;
}

// Candidate counts are intentionally bounded by the Recipe's max_candidates. A single serialized
// selector keeps the Python reference's exact score/y/x order and deterministic NMS semantics;
// the expensive response calculation and local-maximum scan remain fully parallel.
__global__ void pattern_select_nms_kernel(
    const unsigned long long* sorted_keys, long long element_count,
    int template_width, int template_height,
    int max_candidates, float nms_threshold, int max_count, int row_tolerance,
    unsigned long long* selected, int output_capacity,
    int32_t* output_xy, float* output_scores, int* output_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int selected_count = 0;
    int considered = 0;
    const int candidate_limit = max_candidates > 0 ? max_candidates : INT_MAX;
    const int selection_limit = max_count > 0 ? max_count : output_capacity;
    for (long long index = element_count - 1;
         index >= 0 && considered < candidate_limit && selected_count < selection_limit;
         --index) {
        const unsigned long long key = sorted_keys[index];
        if (key == 0) break;
        ++considered;
        float score = 0.0f;
        int x = 0;
        int y = 0;
        pattern_unpack_key(key, &score, &x, &y);
        bool keep = true;
        for (int existing_index = 0; existing_index < selected_count; ++existing_index) {
            float existing_score = 0.0f;
            int existing_x = 0;
            int existing_y = 0;
            pattern_unpack_key(
                selected[existing_index], &existing_score, &existing_x, &existing_y);
            if (!pattern_iou_passes(
                    x, y, existing_x, existing_y,
                    template_width, template_height, nms_threshold)) {
                keep = false;
                break;
            }
        }
        if (keep && selected_count < output_capacity) selected[selected_count++] = key;
    }
    const int tolerance = max(1, row_tolerance);
    for (int index = 1; index < selected_count; ++index) {
        const unsigned long long value = selected[index];
        float value_score = 0.0f;
        int value_x = 0;
        int value_y = 0;
        pattern_unpack_key(value, &value_score, &value_x, &value_y);
        const int value_bucket = __double2int_rn(static_cast<double>(value_y) / tolerance);
        int position = index;
        while (position > 0) {
            float previous_score = 0.0f;
            int previous_x = 0;
            int previous_y = 0;
            pattern_unpack_key(
                selected[position - 1], &previous_score, &previous_x, &previous_y);
            const int previous_bucket =
                __double2int_rn(static_cast<double>(previous_y) / tolerance);
            if (previous_bucket < value_bucket ||
                (previous_bucket == value_bucket && previous_x <= value_x)) break;
            selected[position] = selected[position - 1];
            --position;
        }
        selected[position] = value;
    }
    for (int index = 0; index < selected_count; ++index) {
        float score = 0.0f;
        int x = 0;
        int y = 0;
        pattern_unpack_key(selected[index], &score, &x, &y);
        output_xy[index * 2] = x;
        output_xy[index * 2 + 1] = y;
        output_scores[index] = score;
    }
    *output_count = selected_count;
}

// Tiled score kernel. Each block stages the ROI patch covering its output tile into shared memory
// once, so the window sum, its square and the template-weighted sum are all accumulated from
// shared memory instead of re-reading the ROI for every candidate. The template stays in global
// memory on purpose: it is small, constant per call, and reused by every block, so it stays hot in
// L2 without competing with the ROI patch for shared memory.
//
// Shared bytes: (tile_rows + template_height - 1) * (tile_cols + template_width - 1).
__global__ void match_score_kernel(
    const uint8_t* roi, int roi_width,
    int output_width, int output_height,
    int template_width, int template_height, int template_pixels,
    int tile_cols, int tile_rows,
    double template_mean, double template_variance,
    const uint8_t* templ,
    unsigned long long* best_keys) {
    extern __shared__ unsigned char shared_bytes[];
    uint8_t* tile = shared_bytes;

    const int tiles_x = (output_width + tile_cols - 1) / tile_cols;
    const int tile_x = blockIdx.x % tiles_x;
    const int tile_y = blockIdx.x / tiles_x;
    const int first_column = tile_x * tile_cols;
    const int first_row = tile_y * tile_rows;
    const int columns = min(tile_cols, output_width - first_column);
    const int rows = min(tile_rows, output_height - first_row);
    const int patch_rows = rows + template_height - 1;
    const int patch_cols = columns + template_width - 1;
    const int pitch = patch_cols;

    for (int row = threadIdx.y; row < patch_rows; row += blockDim.y) {
        const uint8_t* source =
            roi + static_cast<size_t>(first_row + row) * roi_width + first_column;
        uint8_t* destination = tile + static_cast<size_t>(row) * pitch;
        for (int column = threadIdx.x; column < patch_cols; column += blockDim.x) {
            destination[column] = source[column];
        }
    }
    __syncthreads();

    for (int output_row = threadIdx.y; output_row < rows; output_row += blockDim.y) {
        for (int output_column = threadIdx.x; output_column < columns; output_column += blockDim.x) {
            long long window_sum = 0;
            long long window_square = 0;
            long long weighted = 0;
            for (int template_row = 0; template_row < template_height; ++template_row) {
                const uint8_t* patch =
                    tile + static_cast<size_t>(output_row + template_row) * pitch + output_column;
                const uint8_t* template_line =
                    templ + static_cast<size_t>(template_row) * template_width;
                for (int offset = 0; offset < template_width; ++offset) {
                    const int value = patch[offset];
                    window_sum += value;
                    window_square += static_cast<long long>(value) * value;
                    weighted += static_cast<long long>(value) * template_line[offset];
                }
            }
            const double mean = static_cast<double>(window_sum) / template_pixels;
            double window_variance =
                static_cast<double>(window_square) / template_pixels - mean * mean;
            if (window_variance < 0.0) window_variance = 0.0;
            const double denominator =
                std::sqrt(window_variance * template_variance) * template_pixels;
            double score = -1.0;
            if (denominator > 0.0) {
                score = (static_cast<double>(weighted) - template_mean * window_sum) / denominator;
            }
            if (score > 1.0) score = 1.0;
            if (score < -1.0) score = -1.0;
            match_offer_candidate(
                score, first_column + output_column, first_row + output_row, best_keys);
        }
    }
}


// One thread per output column. Builds the vertical prefix of the ROI pixel sums and of the ROI
// sum of squares, which is everything the TM_SQDIFF_NORMED score needs:
//   sum((w - t)^2) = sum(w^2) - 2*sum(w*t) + sum(t^2)
__global__ void match_diff_prefix_kernel(
    const uint8_t* roi, int roi_width, int roi_height,
    int output_width, int template_width,
    long long* value_prefix, long long* square_prefix) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= output_width) return;
    long long value_running = 0;
    long long square_running = 0;
    for (int row = threadIdx.y; row < roi_height; row += blockDim.y) {
        const uint8_t* line = roi + static_cast<size_t>(row) * roi_width;
        long long value_sum = 0;
        long long square_sum = 0;
        for (int offset = 0; offset < template_width; ++offset) {
            const long long value = line[column + offset];
            value_sum += value;
            square_sum += value * value;
        }
        value_running += value_sum;
        square_running += square_sum;
        const size_t index = static_cast<size_t>(row) * output_width + column;
        value_prefix[index] = value_running;
        square_prefix[index] = square_running;
    }
}

// One thread per output column minimises the normalized squared difference over the column. The
// packed value is 1 - difference so that the shared "larger key wins" reduction, and therefore the
// topmost-then-leftmost tie order, describes the smallest difference.
__global__ void match_diff_score_kernel(
    const uint8_t* roi, int roi_width, int roi_height,
    int output_width, int output_height,
    int template_width, int template_height,
    const uint8_t* templ,
    const long long* value_prefix, const long long* square_prefix,
    long long template_square_sum,
    double* candidate_scores, int* candidate_ys) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= output_width || template_height > roi_height) return;
    double best_value = -1.0;
    int best_y = -1;
    for (int row = 0; row < output_height; ++row) {
        const size_t bottom = static_cast<size_t>(row + template_height - 1) * output_width + column;
        long long window_value = value_prefix[bottom];
        long long window_square = square_prefix[bottom];
        if (row > 0) {
            const size_t top = static_cast<size_t>(row - 1) * output_width + column;
            window_value -= value_prefix[top];
            window_square -= square_prefix[top];
        }
        long long cross = 0;
        for (int template_row = 0; template_row < template_height; ++template_row) {
            const uint8_t* line =
                roi + static_cast<size_t>(row + template_row) * roi_width + column;
            const uint8_t* template_line =
                templ + static_cast<size_t>(template_row) * template_width;
            long long term = 0;
            for (int offset = 0; offset < template_width; ++offset) {
                term += static_cast<long long>(line[offset]) * template_line[offset];
            }
            cross += term;
        }
        const double numerator =
            static_cast<double>(window_square) - 2.0 * cross + static_cast<double>(template_square_sum);
        const double denominator = std::sqrt(
            static_cast<double>(window_square) * static_cast<double>(template_square_sum));
        const double difference = denominator > 0.0 ? numerator / denominator : 1.0;
        const double clamped_difference = difference < 0.0 ? 0.0 : (difference > 1.0 ? 1.0 : difference);
        const double value = 1.0 - clamped_difference;
        if (value > best_value) {
            best_value = value;
            best_y = row;
        }
    }
    candidate_scores[column] = best_value;
    candidate_ys[column] = best_y;
}

// Reduces the per-column packed keys into one slot, keeping the same ordering the columns used.
__global__ void match_publish_kernel(
    const unsigned long long* column_keys, int output_width, unsigned long long* best_key) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= output_width) return;
    const unsigned long long key = column_keys[column];
    if (key == 0) return;
    unsigned long long* slot = best_key;
    unsigned long long current = *slot;
    while (key > current) {
        const unsigned long long previous = atomicCAS(slot, current, key);
        if (previous == current) break;
        current = previous;
    }
}

// Expands the winning key back into the rectangle origin and score. mode_sign is +1 for
// TM_CCOEFF_NORMED (report the packed value) and -1 for TM_SQDIFF_NORMED (report 1 - value), the
// same conversion core/tiler.py applies when it detects a flat template.
__global__ void match_unpack_kernel(
    const unsigned long long* best_key, float mode_sign, int* out_xy, float* out_score) {
    double score = 0.0;
    int x = 0;
    int y = 0;
    match_unpack_key(*best_key, &score, &x, &y);
    out_xy[0] = x;
    out_xy[1] = y;
    *out_score = static_cast<float>(1.0 + mode_sign * (score - 1.0));
}


// Mirrors OpenCV computeResizeAreaTab: double geometry, 1e-3 edge tolerance, float weights.
void append_area_axis(
    int source_size, int target_size, double scale,
    std::vector<int>* offsets, std::vector<int>* sources, std::vector<float>* weights) {
    offsets->push_back(0);
    for (int target = 0; target < target_size; ++target) {
        const double first = target * scale;
        const double last = first + scale;
        const double cell_width = std::min(scale, source_size - first);
        int start = static_cast<int>(std::ceil(first));
        int end = static_cast<int>(std::floor(last));
        end = std::min(end, source_size - 1);
        start = std::min(start, end);
        if (start - first > 1e-3) {
            sources->push_back(start - 1);
            weights->push_back(static_cast<float>((start - first) / cell_width));
        }
        for (int source = start; source < end; ++source) {
            sources->push_back(source);
            weights->push_back(static_cast<float>(1.0 / cell_width));
        }
        if (last - end > 1e-3) {
            sources->push_back(end);
            weights->push_back(static_cast<float>(
                std::min(std::min(last - end, 1.0), cell_width) / cell_width));
        }
        offsets->push_back(static_cast<int>(sources->size()));
    }
}

int prepare_area_resize(
    int source_width, int source_height, int target_width, int target_height,
    AreaResizeTables* tables, unsigned long long* allocation_count) {
    if (tables == nullptr || source_width <= 0 || source_height <= 0 || target_width <= 0 ||
        target_height <= 0 || target_width > source_width || target_height > source_height) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (source_width == target_width && source_height == target_height) {
        tables->mode = AREA_RESIZE_COPY;
        return VF_CUDA_OK;
    }
    const double scale_x = 1.0 / (static_cast<double>(target_width) / source_width);
    const double scale_y = 1.0 / (static_cast<double>(target_height) / source_height);
    const int integer_x = static_cast<int>(std::lround(scale_x));
    const int integer_y = static_cast<int>(std::lround(scale_y));
    if (std::abs(scale_x - integer_x) < DBL_EPSILON && std::abs(scale_y - integer_y) < DBL_EPSILON) {
        tables->scale_x = integer_x;
        tables->scale_y = integer_y;
        if (integer_x == 2 && integer_y == 2) {
            tables->mode = AREA_RESIZE_FAST_2X2;
        } else {
            tables->mode = AREA_RESIZE_FAST_INTEGER;
            tables->inverse_area = static_cast<float>(1.0 / (integer_x * integer_y));
        }
        return VF_CUDA_OK;
    }
    std::vector<int> x_offsets, y_offsets, x_sources, y_sources;
    std::vector<float> x_weights, y_weights;
    try {
        append_area_axis(source_width, target_width, scale_x, &x_offsets, &x_sources, &x_weights);
        append_area_axis(source_height, target_height, scale_y, &y_offsets, &y_sources, &y_weights);
        std::vector<int> indices;
        indices.reserve(x_offsets.size() + y_offsets.size() + x_sources.size() + y_sources.size());
        indices.insert(indices.end(), x_offsets.begin(), x_offsets.end());
        indices.insert(indices.end(), y_offsets.begin(), y_offsets.end());
        indices.insert(indices.end(), x_sources.begin(), x_sources.end());
        indices.insert(indices.end(), y_sources.begin(), y_sources.end());
        std::vector<float> weights(x_weights);
        weights.insert(weights.end(), y_weights.begin(), y_weights.end());

        tables->mode = AREA_RESIZE_GENERAL;
        tables->x_entries = static_cast<int>(x_sources.size());
        int* device_indices = nullptr;
        cudaError_t error = cudaMalloc(&device_indices, indices.size() * sizeof(int));
        if (error != cudaSuccess) return cuda_result(error);
        tables->indices = device_indices;
        if (allocation_count != nullptr) ++(*allocation_count);
        float* device_alphas = nullptr;
        error = cudaMalloc(&device_alphas, weights.size() * sizeof(float));
        if (error != cudaSuccess) return cuda_result(error);
        tables->alphas = device_alphas;
        if (allocation_count != nullptr) ++(*allocation_count);
        error = cudaMemcpy(tables->indices, indices.data(), indices.size() * sizeof(int),
                           cudaMemcpyHostToDevice);
        if (error == cudaSuccess) {
            error = cudaMemcpy(tables->alphas, weights.data(), weights.size() * sizeof(float),
                               cudaMemcpyHostToDevice);
        }
        return cuda_result(error);
    } catch (const std::bad_alloc&) {
        return VF_CUDA_ALLOCATION_FAILED;
    }
}

void launch_area_resize(
    const AreaResizeTables& tables, const uint8_t* src, uint8_t* dst,
    int source_width, int target_width, int target_height, cudaStream_t stream = nullptr) {
    resize_area_kernel<<<grid2d(target_width, target_height), dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
        src, dst, source_width, target_width, target_height, tables.mode, tables.scale_x,
        tables.scale_y, tables.inverse_area, tables.x_entries, tables.indices, tables.alphas);
}

// ---------------------------------------------------------------------------------------------
// Contour extension: cv2.findContours(RETR_LIST | RETR_EXTERNAL, CHAIN_APPROX_SIMPLE) equivalence.
//
// This is a direct port of tools/contour_reference.py, which was verified point-for-point against
// cv2.findContours. The reference follows OpenCV contours.cpp:
//   cvStartFindContours_Impl -> 1-pixel zero frame, THRESH_BINARY binarization, scanner state
//   cvFindNextContour        -> raster scan, outer/hole classification, lnbd bookkeeping
//   icvFetchContour          -> the 8-neighbour border trace, CHAIN_APPROX_SIMPLE point rule
//
// Two properties keep the port cheap and exact:
//   - CHAIN_APPROX_SIMPLE compression is inherent to the trace (a point is emitted only where the
//     step direction changes), so no separate compression pass exists.
//   - The trace only ever tests whether a neighbour is non-zero, and marking only rewrites 1 into
//     2 or -126 (both still non-zero), so the trace of a border is independent of the marks left
//     by other borders. Only the raster scan is order-dependent, which is why it stays serial.
//
// OpenCV reports the flat contour list in reverse discovery order (icvEndProcessContour prepends to
// frame->v_next), so a final kernel reverses the discovery-order scratch into the output.
// ---------------------------------------------------------------------------------------------
constexpr int CONTOUR_NBD = 2;        // const schar nbd = 2 inside icvFetchContour
constexpr int CONTOUR_MARKED = -126;  // (schar)(nbd | -128)

// CV_INIT_3X3_DELTAS(deltas, step, 1): index 0..7 is E, NE, N, NW, W, SW, S, SE and 8..15 mirrors
// it. `index & 7` reproduces the 16-entry table exactly, including the exhausted search (15).
__device__ __forceinline__ void contour_ring_step(int index, int* dy, int* dx) {
    const int ring_dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    const int ring_dy[8] = {0, -1, -1, -1, 0, 1, 1, 1};
    const int slot = index & 7;
    *dx = ring_dx[slot];
    *dy = ring_dy[slot];
}

// Appends one point; a full buffer sets the overflow flag instead of truncating the contour.
__device__ __forceinline__ void contour_store_point(
    int32_t* points, int point_capacity, int* point_index, int* overflow, int px, int py) {
    const int index = *point_index;
    if (index < point_capacity) {
        points[static_cast<size_t>(index) * 2] = px;
        points[static_cast<size_t>(index) * 2 + 1] = py;
    } else {
        *overflow = 1;
    }
    *point_index = index + 1;
}

// Port of icvFetchContour(ptr, step, pt, contour, CV_CHAIN_APPROX_SIMPLE). The padded label image
// is mutated in place exactly like OpenCV does (marks 2 / -126).
__device__ void contour_fetch(
    signed char* image, int stride, int i0_y, int i0_x, int is_hole, int pt_x, int pt_y,
    int32_t* points, int point_capacity, int* point_index, int* overflow) {
    int s_end = is_hole ? 0 : 4;
    int s = s_end;
    int i1_y = i0_y;
    int i1_x = i0_x;
    for (;;) {
        s = (s - 1) & 7;
        int dy = 0;
        int dx = 0;
        contour_ring_step(s, &dy, &dx);
        i1_y = i0_y + dy;
        i1_x = i0_x + dx;
        if (image[static_cast<size_t>(i1_y) * stride + i1_x] != 0) break;
        if (s == s_end) break;
    }

    if (s == s_end) {
        // Single-pixel domain: mark the pixel and emit exactly one point.
        image[static_cast<size_t>(i0_y) * stride + i0_x] = static_cast<signed char>(CONTOUR_MARKED);
        contour_store_point(points, point_capacity, point_index, overflow, pt_x, pt_y);
        return;
    }

    int i3_y = i0_y;
    int i3_x = i0_x;
    int prev_s = s ^ 4;
    int i4_y = 0;
    int i4_x = 0;
    for (;;) {
        s_end = s;
        // `s` is always in 0..7 here, so C's `s = min(s, MAX_SIZE - 1)` is a no-op.
        while (s < 15) {
            s += 1;
            int dy = 0;
            int dx = 0;
            contour_ring_step(s, &dy, &dx);
            i4_y = i3_y + dy;
            i4_x = i3_x + dx;
            if (image[static_cast<size_t>(i4_y) * stride + i4_x] != 0) break;
        }
        s &= 7;

        // Right-bound marking: (unsigned)(s - 1) < (unsigned)s_end means 1 <= s <= s_end.
        if (s >= 1 && (s - 1) < s_end) {
            image[static_cast<size_t>(i3_y) * stride + i3_x] =
                static_cast<signed char>(CONTOUR_MARKED);
        } else if (image[static_cast<size_t>(i3_y) * stride + i3_x] == 1) {
            image[static_cast<size_t>(i3_y) * stride + i3_x] = static_cast<signed char>(CONTOUR_NBD);
        }

        if (s != prev_s) {
            contour_store_point(points, point_capacity, point_index, overflow, pt_x, pt_y);
            prev_s = s;
        }

        int dy = 0;
        int dx = 0;
        contour_ring_step(s, &dy, &dx);
        pt_y += dy;
        pt_x += dx;

        if (i4_y == i0_y && i4_x == i0_x && i3_y == i1_y && i3_x == i1_x) break;
        i3_y = i4_y;
        i3_x = i4_x;
        s = (s + 4) & 7;
    }
}

// Warp-cooperative form of contour_fetch for RETR_LIST. The border walk is still sequential, but
// its expensive operation is choosing the first non-zero pixel in an eight-neighbour ring. Eight
// lanes load that ring together and a ballot selects the same first neighbour as OpenCV's serial
// loop. Lane zero remains the sole writer, so marking and CHAIN_APPROX_SIMPLE output stay exact.
__device__ void contour_fetch_warp(
    signed char* image, int stride, int i0_y, int i0_x, int is_hole, int pt_x, int pt_y,
    int32_t* points, int point_capacity, int* point_index, int* overflow) {
    constexpr unsigned int warp_mask = 0xffffffffu;
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    int s_end = is_hole ? 0 : 4;

    int search_s = 0;
    int search_y = 0;
    int search_x = 0;
    bool occupied = false;
    if (lane < 8) {
        search_s = (s_end - 1 - lane) & 7;
        int dy = 0;
        int dx = 0;
        contour_ring_step(search_s, &dy, &dx);
        search_y = i0_y + dy;
        search_x = i0_x + dx;
        occupied = image[static_cast<size_t>(search_y) * stride + search_x] != 0;
    }
    unsigned int occupied_lanes = __ballot_sync(warp_mask, occupied) & 0xffu;
    if (occupied_lanes == 0) {
        if (lane == 0) {
            image[static_cast<size_t>(i0_y) * stride + i0_x] =
                static_cast<signed char>(CONTOUR_MARKED);
            contour_store_point(points, point_capacity, point_index, overflow, pt_x, pt_y);
        }
        __syncwarp(warp_mask);
        return;
    }

    int selected_lane = __ffs(static_cast<int>(occupied_lanes)) - 1;
    int s = (s_end - 1 - selected_lane) & 7;
    int dy = 0;
    int dx = 0;
    contour_ring_step(s, &dy, &dx);
    const int i1_y = i0_y + dy;
    const int i1_x = i0_x + dx;
    int i3_y = i0_y;
    int i3_x = i0_x;
    int prev_s = s ^ 4;

    for (;;) {
        s_end = s;
        occupied = false;
        if (lane < 8) {
            search_s = (s_end + lane + 1) & 7;
            contour_ring_step(search_s, &dy, &dx);
            search_y = i3_y + dy;
            search_x = i3_x + dx;
            occupied = image[static_cast<size_t>(search_y) * stride + search_x] != 0;
        }
        occupied_lanes = __ballot_sync(warp_mask, occupied) & 0xffu;
        selected_lane = __ffs(static_cast<int>(occupied_lanes)) - 1;
        s = (s_end + selected_lane + 1) & 7;
        contour_ring_step(s, &dy, &dx);
        const int i4_y = i3_y + dy;
        const int i4_x = i3_x + dx;

        if (lane == 0) {
            if (s >= 1 && (s - 1) < s_end) {
                image[static_cast<size_t>(i3_y) * stride + i3_x] =
                    static_cast<signed char>(CONTOUR_MARKED);
            } else if (image[static_cast<size_t>(i3_y) * stride + i3_x] == 1) {
                image[static_cast<size_t>(i3_y) * stride + i3_x] =
                    static_cast<signed char>(CONTOUR_NBD);
            }
            if (s != prev_s) {
                contour_store_point(points, point_capacity, point_index, overflow, pt_x, pt_y);
            }
        }
        if (s != prev_s) prev_s = s;
        pt_y += dy;
        pt_x += dx;
        const bool complete =
            i4_y == i0_y && i4_x == i0_x && i3_y == i1_y && i3_x == i1_x;
        __syncwarp(warp_mask);
        if (complete) break;
        i3_y = i4_y;
        i3_x = i4_x;
        s = (s + 4) & 7;
    }
}

// Builds the 1-pixel-zero-framed label image of the requested region. Every padded pixel is
// written by exactly one thread, so the buffer is a pure function of the mask.
__global__ void contour_init_label_kernel(
    const uint8_t* mask, int mask_stride, int x, int y, int width, int height,
    signed char* label, int label_stride) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column > width + 1 || row > height + 1) return;
    signed char value = 0;
    if (column >= 1 && column <= width && row >= 1 && row <= height) {
        const uint8_t* source =
            mask + static_cast<size_t>(y + row - 1) * mask_stride + static_cast<size_t>(x + column - 1);
        value = (*source != 0) ? static_cast<signed char>(1) : static_cast<signed char>(0);
    }
    label[static_cast<size_t>(row) * label_stride + column] = value;
}

// Decides what the raster scan must do at a stop position: a stop is any column where the current
// label differs from the column to its left. Returns true when a border must be opened, with
// *is_hole set to 0 (outer) or 1 (hole). The two else-branches mirror cvFindNextContour's
// resume_scan path, including its lnbd bookkeeping.
__device__ __forceinline__ bool contour_stop_starts_border(
    const signed char* image, int stride, int y, int x, int mode,
    int* lnbd_x, int* lnbd_y, int* is_hole_out) {
    const int prev = static_cast<int>(image[static_cast<size_t>(y) * stride + (x - 1)]);
    const int p = static_cast<int>(image[static_cast<size_t>(y) * stride + x]);
    *is_hole_out = 0;
    if (!(prev == 0 && p == 1)) {
        // Not an outer border. `p != 0 || prev < 1` also rejects a hole start where the left pixel
        // carries the -126 right-bound mark (which is < 1).
        if (p != 0 || prev < 1) {
            if (p & -2) *lnbd_x = x;
            return false;
        }
        *is_hole_out = 1;
    }
    // RETR_EXTERNAL skips hole borders and borders whose left neighbour already carries a label.
    if (mode == VF_CONTOURS_RETR_EXTERNAL &&
        (*is_hole_out != 0 ||
         static_cast<int>(image[static_cast<size_t>(*lnbd_y) * stride + *lnbd_x]) > 0)) {
        if (p & -2) *lnbd_x = x;
        return false;
    }
    return true;
}

// Opens one border at (y, x) and traces it into the discovery-order scratch.
__device__ __forceinline__ void contour_open_border(
    signed char* image, int stride, int y, int x, int is_hole,
    int32_t* offsets, int offset_capacity,
    int32_t* points, int point_capacity,
    int* contour_count, int* point_index, int* overflow) {
    const int origin_y = y;
    const int origin_x = x - is_hole;
    if (*contour_count < offset_capacity) {
        offsets[*contour_count] = *point_index;
    } else {
        *overflow = 1;
    }
    contour_fetch(
        image, stride, origin_y, origin_x, is_hole, origin_x - 1, origin_y - 1,
        points, point_capacity, point_index, overflow);
    if (*contour_count + 1 < offset_capacity) {
        offsets[*contour_count + 1] = *point_index;
    } else {
        *overflow = 1;
    }
    *contour_count += 1;
}

// RETR_LIST calls this from one full warp. Lane zero owns the offset table and counters while all
// lanes cooperate in the neighbour reads performed by contour_fetch_warp.
__device__ __forceinline__ void contour_open_border_warp(
    signed char* image, int stride, int y, int x, int is_hole,
    int32_t* offsets, int offset_capacity,
    int32_t* points, int point_capacity,
    int* contour_count, int* point_index, int* overflow) {
    constexpr unsigned int warp_mask = 0xffffffffu;
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    const int origin_y = y;
    const int origin_x = x - is_hole;
    if (lane == 0) {
        if (*contour_count < offset_capacity) {
            offsets[*contour_count] = *point_index;
        } else {
            *overflow = 1;
        }
    }
    __syncwarp(warp_mask);
    contour_fetch_warp(
        image, stride, origin_y, origin_x, is_hole, origin_x - 1, origin_y - 1,
        points, point_capacity, point_index, overflow);
    __syncwarp(warp_mask);
    if (lane == 0) {
        if (*contour_count + 1 < offset_capacity) {
            offsets[*contour_count + 1] = *point_index;
        } else {
            *overflow = 1;
        }
        *contour_count += 1;
    }
    __syncwarp(warp_mask);
}

// Serial port of the cvStartFindContours_Impl / cvFindNextContour raster scan plus the per-border
// trace. A single thread owns the whole scan, so the marking order is the reference order and no
// synchronization or atomic is involved; that is what makes the operator deterministic.
//
// counts[0] = contour count, counts[1] = point count, counts[2] = overflow flag. The counts are
// reported even when the buffers were too small, so the caller can retry with the exact size.
// This literal row walk is the reference form and stays in charge of RETR_EXTERNAL, where the
// scan's lnbd bookkeeping can be updated by stops that this walk sees and a transition list does
// not. RETR_LIST takes contour_scan_list_kernel instead.
__global__ void contour_scan_kernel(
    signed char* image, int stride, int width, int height, int mode,
    int32_t* offsets, int offset_capacity,
    int32_t* points, int point_capacity,
    int* counts) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x != 0 || threadIdx.y != 0) return;
    const int scan_w = width + 1;  // scanner->img_size.width  = W + 2 - 1
    const int scan_h = height + 1; // scanner->img_size.height = H + 2 - 1
    int contour_count = 0;
    int point_index = 0;
    int overflow = 0;

    int x = 1;
    int y = 1;
    int lnbd_x = 0;
    int lnbd_y = 1;
    int prev = static_cast<int>(image[static_cast<size_t>(y) * stride + (x - 1)]);

    while (y < scan_h) {
        int restarted = 0;
        signed char* row = image + static_cast<size_t>(y) * stride;
        while (x < scan_w) {
            while (x < scan_w && static_cast<int>(row[x]) == prev) x += 1;
            if (x >= scan_w) break;
            int is_hole = 0;
            if (contour_stop_starts_border(image, stride, y, x, mode, &lnbd_x, &lnbd_y, &is_hole)) {
                lnbd_x = x - is_hole;
                lnbd_y = y;
                contour_open_border(
                    image, stride, y, x, is_hole, offsets, offset_capacity,
                    points, point_capacity, &contour_count, &point_index, &overflow);
                restarted = 1;
            }
            x += 1;
            prev = static_cast<int>(row[x - 1]);
            if (restarted) break;
        }
        if (restarted) continue;
        lnbd_x = 0;
        lnbd_y = y + 1;
        x = 1;
        prev = 0;
        y += 1;
    }

    counts[0] = contour_count;
    counts[1] = point_index;
    counts[2] = overflow;
}

// Row transition counts of the region's zero-ness: a column where the binary value differs from its
// left neighbour (the padded frame counts as background). Every border this scan can open sits on
// such a column, because both the outer rule (bg -> fg) and the hole rule (fg -> bg) require the
// value change, and marking only rewrites 1 into 2 or -126, which never changes zero-ness.
// One thread per row keeps the pass deterministic; the count is exact, so the list needs no guess.
__global__ void contour_row_transition_counts_kernel(
    const uint8_t* mask, int mask_stride, int x0, int y0, int width, int height,
    int* row_counts) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (row > height) return;
    const uint8_t* source = mask + static_cast<size_t>(y0 + row - 1) * mask_stride + x0;
    int previous = 0;
    int count = 0;
    for (int column = 1; column <= width; ++column) {
        const int value = source[column - 1] != 0 ? 1 : 0;
        if (value != previous) count += 1;
        previous = value;
    }
    row_counts[row - 1] = count;
}

// Fills the raster-ordered list of padded label indices, one thread per row, each row writing its
// own segment in ascending column order.
__global__ void contour_fill_transitions_kernel(
    const uint8_t* mask, int mask_stride, int x0, int y0, int width, int height,
    const int* row_start, int stride, int* transitions) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (row > height) return;
    const uint8_t* source = mask + static_cast<size_t>(y0 + row - 1) * mask_stride + x0;
    int previous = 0;
    int index = row_start[row];
    const int base = row * stride;
    for (int column = 1; column <= width; ++column) {
        const int value = source[column - 1] != 0 ? 1 : 0;
        if (value != previous) {
            transitions[index] = base + column;
            index += 1;
        }
        previous = value;
    }
}

// RETR_LIST scan over the precomputed transition list. Iterating only the value changes is exact:
// a stop whose left neighbour keeps the same zero-ness can only take the harmless resume_scan
// branch (it updates prev, which this kernel re-reads from the image at every stop, and lnbd_x,
// which RETR_LIST never reads). The literal row walk and this walk therefore open the same borders
// in the same order, while this one touches memory only where a decision can happen.
__global__ void contour_scan_list_kernel(
    signed char* image, int stride, int height,
    const int32_t* transitions, const int32_t* row_start,
    int32_t* offsets, int offset_capacity,
    int32_t* points, int point_capacity,
    int* counts) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x >= warpSize || threadIdx.y != 0) return;
    constexpr unsigned int warp_mask = 0xffffffffu;
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    int contour_count = 0;
    int point_index = 0;
    int overflow = 0;
    int lnbd_x = 0;
    int lnbd_y = 1;

    for (int y = 1; y <= height; ++y) {
        const int end = row_start[y + 1];
        for (int index = row_start[y]; index < end; ++index) {
            const int position = transitions[index];
            const int x = position - y * stride;
            int starts_border = 0;
            int is_hole = 0;
            if (lane == 0) {
                starts_border = contour_stop_starts_border(
                    image, stride, y, x, VF_CONTOURS_RETR_LIST, &lnbd_x, &lnbd_y, &is_hole)
                    ? 1 : 0;
                if (starts_border != 0) {
                    lnbd_x = x - is_hole;
                    lnbd_y = y;
                }
            }
            starts_border = __shfl_sync(warp_mask, starts_border, 0);
            is_hole = __shfl_sync(warp_mask, is_hole, 0);
            if (starts_border != 0) {
                contour_open_border_warp(
                    image, stride, y, x, is_hole, offsets, offset_capacity,
                    points, point_capacity, &contour_count, &point_index, &overflow);
            }
        }
    }

    if (lane == 0) {
        counts[0] = contour_count;
        counts[1] = point_index;
        counts[2] = overflow;
    }
}

// Reverses the discovery-order scratch into the OpenCV order. Each contour is copied by one thread
// that also derives its output offset from the monotonic offset table, so the result is a pure
// function of the scratch.
__global__ void contour_reverse_kernel(
    const int32_t* offsets, const int32_t* points, int contour_count, int point_count,
    int32_t* out_offsets, int32_t* out_points) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= contour_count) return;
    const int source = contour_count - 1 - index;
    const int source_start = offsets[source];
    const int source_end = offsets[source + 1];
    const int target_start = point_count - offsets[contour_count - index];
    out_offsets[index] = target_start;
    if (index == 0) out_offsets[contour_count] = point_count;
    for (int point = source_start; point < source_end; ++point) {
        const int target = target_start + (point - source_start);
        out_points[static_cast<size_t>(target) * 2] = points[static_cast<size_t>(point) * 2];
        out_points[static_cast<size_t>(target) * 2 + 1] = points[static_cast<size_t>(point) * 2 + 1];
    }
}
}

static int execute_linear_plan_device(
    NativePlan* compiled,
    uint8_t* current,
    uint8_t* dst,
    int dst_stride,
    int dst_channels,
    uint8_t** device_output = nullptr) {
    PersistentContext* context = compiled->context;
    int width = compiled->width;
    int height = compiled->height;
    int channels = compiled->input_channels;
    size_t area_resize_index = 0;
    for (const VfPlanOperatorV1& op : compiled->operators) {
        uint8_t* next = current == context->u8[1] ? context->u8[2] : context->u8[1];
        switch (op.kind) {
            case VF_PLAN_GRAY:
                if (channels == 3) {
                    bgr_gray_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                        current, next, width, height);
                    current = next;
                    channels = 1;
                }
                break;
            case VF_PLAN_RESIZE_AREA: {
                const int target_width = op.int_params[0];
                const int target_height = op.int_params[1];
                if (area_resize_index >= compiled->area_resizes.size()) return VF_CUDA_INTERNAL_ERROR;
                launch_area_resize(
                    *compiled->area_resizes[area_resize_index++], current, next, width,
                    target_width, target_height, context->stream);
                current = next;
                width = target_width;
                height = target_height;
                break;
            }
            case VF_PLAN_GAUSSIAN: {
                context->timing_has_gaussian = true;
                record_timing_event(context, TIMING_GAUSSIAN_START);
                int radius = 0;
                int result = prepare_gaussian_weights(
                    op.int_params[0], &radius, context->stream);
                if (result != VF_CUDA_OK) return result;
                launch_gaussian(
                    current, context->gaussian_buffer, next, width, height, channels, radius,
                    context->stream);
                record_timing_event(context, TIMING_GAUSSIAN_END);
                current = next;
                break;
            }
            case VF_PLAN_THRESHOLD:
                context->timing_has_threshold = true;
                record_timing_event(context, TIMING_THRESHOLD_START);
                threshold_kernel<<<(width * height + 255) / 256, 256, 0, context->stream>>>(
                    current, next, width * height, op.int_params[0],
                    op.int_params[1], op.int_params[2]);
                record_timing_event(context, TIMING_THRESHOLD_END);
                current = next;
                break;
            case VF_PLAN_ADAPTIVE_MEAN: {
                context->timing_has_adaptive = true;
                record_timing_event(context, TIMING_ADAPTIVE_START);
                size_t scratch_count = 0;
                int result = adaptive_layout(
                    width, height, op.int_params[0], &scratch_count);
                if (result != VF_CUDA_OK) return result;
                launch_adaptive_mean(
                    current, context->gaussian_buffer, next, width, height,
                    op.int_params[0], op.float_params[0], op.int_params[1], op.int_params[2],
                    context->stream);
                record_timing_event(context, TIMING_ADAPTIVE_END);
                current = next;
                break;
            }
            case VF_PLAN_MORPHOLOGY: {
                context->timing_has_morphology = true;
                record_timing_event(context, TIMING_MORPHOLOGY_START);
                const int operation = op.int_params[0];
                const int kernel = op.int_params[1];
                const int iterations = op.int_params[2];
                const int passes = (operation == VF_MORPH_OPEN || operation == VF_MORPH_CLOSE)
                    ? iterations * 2 : iterations;
                uint8_t* source_buffer = current;
                uint8_t* next = current == context->u8[1] ? context->u8[2] : context->u8[1];
                uint8_t* destination = passes % 2 == 1 ? next : context->u8[4];
                for (int pass = 0; pass < passes; ++pass) {
                    int dilate = operation == VF_MORPH_DILATE;
                    if (operation == VF_MORPH_OPEN) dilate = pass >= iterations;
                    if (operation == VF_MORPH_CLOSE) dilate = pass < iterations;
                    launch_morph_pass(
                        source_buffer, destination, width, height, channels,
                        kernel / 2, dilate, context->stream);
                    source_buffer = destination;
                    destination = destination == next ? context->u8[4] : next;
                }
                record_timing_event(context, TIMING_MORPHOLOGY_END);
                current = source_buffer;
                break;
            }
            default:
                return VF_CUDA_UNSUPPORTED;
        }
    }
    int result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    record_timing_event(context, TIMING_AFTER_KERNEL);
    cudaError_t error = cudaSuccess;
    if (device_output != nullptr) {
        *device_output = current;
    } else {
        const size_t output_row_bytes = static_cast<size_t>(width) * dst_channels;
        error = cudaMemcpy2DAsync(
            dst, dst_stride, current, output_row_bytes, output_row_bytes, height,
            cudaMemcpyDeviceToHost, context->stream);
        if (error != cudaSuccess) return cuda_result(error);
    }
    record_timing_event(context, TIMING_AFTER_OUTPUT);
    auto synchronize_started = std::chrono::steady_clock::now();
    result = visionflow_cuda::stream_result(context->stream);
    context->last_timings.synchronize_ms = elapsed_host_ms(synchronize_started);
    if (result == VF_CUDA_OK) finalize_timing(context);
    return result;
}

static int execute_dag_plan_device(
    NativeDagPlan* compiled,
    uint8_t* root,
    const VfDagOutputV1* outputs,
    int output_count) {
    PersistentContext* context = compiled->context;
    const int width = compiled->width;
    const int height = compiled->height;
    const size_t pixels = static_cast<size_t>(width) * height;
    std::vector<uint8_t*> values(compiled->operators.size(), nullptr);
    for (size_t index = 0; index < compiled->operators.size(); ++index) {
        const VfPlanOperatorV1& op = compiled->operators[index];
        uint8_t* input = op.input_node == VF_PLAN_INPUT_NODE ? root : values[op.input_node];
        uint8_t* output = context->dag_u8[index];
        int channels = op.input_node == VF_PLAN_INPUT_NODE
            ? compiled->input_channels : compiled->node_channels[op.input_node];
        switch (op.kind) {
            case VF_PLAN_GRAY:
                if (channels == 3) {
                    bgr_gray_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                        input, output, width, height);
                    values[index] = output;
                } else {
                    values[index] = input;
                }
                break;
            case VF_PLAN_GAUSSIAN: {
                context->timing_has_gaussian = true;
                record_timing_event(context, TIMING_GAUSSIAN_START);
                int radius = 0;
                int result = prepare_gaussian_weights(
                    op.int_params[0], &radius, context->stream);
                if (result != VF_CUDA_OK) return result;
                launch_gaussian(
                    input, context->gaussian_buffer, output, width, height, channels, radius,
                    context->stream);
                record_timing_event(context, TIMING_GAUSSIAN_END);
                values[index] = output;
                break;
            }
            case VF_PLAN_THRESHOLD:
                context->timing_has_threshold = true;
                record_timing_event(context, TIMING_THRESHOLD_START);
                threshold_kernel<<<(static_cast<int>(pixels) + 255) / 256, 256, 0, context->stream>>>(
                    input, output, static_cast<int>(pixels), op.int_params[0],
                    op.int_params[1], op.int_params[2]);
                record_timing_event(context, TIMING_THRESHOLD_END);
                values[index] = output;
                break;
            case VF_PLAN_ADAPTIVE_MEAN: {
                context->timing_has_adaptive = true;
                record_timing_event(context, TIMING_ADAPTIVE_START);
                size_t scratch_count = 0;
                int result = adaptive_layout(
                    width, height, op.int_params[0], &scratch_count);
                if (result != VF_CUDA_OK) return result;
                launch_adaptive_mean(
                    input, context->gaussian_buffer, output, width, height,
                    op.int_params[0], op.float_params[0], op.int_params[1], op.int_params[2],
                    context->stream);
                record_timing_event(context, TIMING_ADAPTIVE_END);
                values[index] = output;
                break;
            }
            case VF_PLAN_MORPHOLOGY: {
                context->timing_has_morphology = true;
                record_timing_event(context, TIMING_MORPHOLOGY_START);
                const int operation = op.int_params[0];
                const int iterations = op.int_params[2];
                const int passes = (operation == VF_MORPH_OPEN || operation == VF_MORPH_CLOSE)
                    ? iterations * 2 : iterations;
                uint8_t* source_buffer = input;
                uint8_t* destination = passes % 2 == 1 ? output : context->u8[4];
                for (int pass = 0; pass < passes; ++pass) {
                    int dilate = operation == VF_MORPH_DILATE;
                    if (operation == VF_MORPH_OPEN) dilate = pass >= iterations;
                    if (operation == VF_MORPH_CLOSE) dilate = pass < iterations;
                    launch_morph_pass(
                        source_buffer, destination, width, height, channels,
                        op.int_params[1] / 2, dilate, context->stream);
                    source_buffer = destination;
                    destination = destination == output ? context->u8[4] : output;
                }
                record_timing_event(context, TIMING_MORPHOLOGY_END);
                values[index] = source_buffer;
                break;
            }
            default:
                return VF_CUDA_UNSUPPORTED;
        }
    }
    int result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    record_timing_event(context, TIMING_AFTER_KERNEL);
    for (int index = 0; index < output_count; ++index) {
        int node = compiled->output_nodes[index];
        size_t row_bytes = static_cast<size_t>(width) * compiled->node_channels[node];
        cudaError_t error = cudaMemcpy2DAsync(
            outputs[index].data, outputs[index].stride, values[node], row_bytes,
            row_bytes, height, cudaMemcpyDeviceToHost, context->stream);
        if (error != cudaSuccess) return cuda_result(error);
    }
    record_timing_event(context, TIMING_AFTER_OUTPUT);
    auto synchronize_started = std::chrono::steady_clock::now();
    result = visionflow_cuda::stream_result(context->stream);
    context->last_timings.synchronize_ms = elapsed_host_ms(synchronize_started);
    if (result == VF_CUDA_OK) finalize_timing(context);
    return result;
}

VF_CUDA_API int vf_gpu_abi_version() { return VF_CUDA_ABI_VERSION; }

VF_CUDA_API int vf_gpu_device_count() { int count = 0; return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0; }

VF_CUDA_API int vf_gpu_compute_capability() {
    cudaDeviceProp prop{};
    return cudaGetDeviceProperties(&prop, 0) == cudaSuccess ? prop.major * 10 + prop.minor : 0;
}

VF_CUDA_API int vf_gpu_device_name(char* output, int capacity) {
    if (!output || capacity <= 0) return 1;
    cudaDeviceProp prop{}; cudaError_t error = cudaGetDeviceProperties(&prop, 0);
    if (error != cudaSuccess) return cuda_result(error);
    strncpy_s(output, capacity, prop.name, _TRUNCATE); return 0;
}

VF_CUDA_API int vf_gpu_error_message(int error_code, char* output, int capacity) {
    if (!output || capacity <= 0) return VF_CUDA_INVALID_ARGUMENT;
    const char* message = "Unknown VisionFlow CUDA error";
    switch (error_code) {
        case VF_CUDA_OK: message = "Success"; break;
        case VF_CUDA_INVALID_ARGUMENT: message = "Invalid argument"; break;
        case VF_CUDA_ALLOCATION_FAILED: message = "Device allocation failed"; break;
        case VF_CUDA_COPY_FAILED: message = "Host/device copy failed"; break;
        case VF_CUDA_KERNEL_FAILED: message = "CUDA kernel failed"; break;
        case VF_CUDA_DEVICE_UNAVAILABLE: message = "CUDA device unavailable"; break;
        case VF_CUDA_ABI_MISMATCH: message = "CUDA DLL ABI mismatch"; break;
        case VF_CUDA_INTERNAL_ERROR: message = "Internal CUDA DLL error"; break;
        case VF_CUDA_UNSUPPORTED: message = "Requested CUDA operation is unsupported"; break;
        default:
            if (error_code >= VF_CUDA_RUNTIME_ERROR_BASE) {
                message = cudaGetErrorString(static_cast<cudaError_t>(error_code - VF_CUDA_RUNTIME_ERROR_BASE));
            }
            break;
    }
    strncpy_s(output, capacity, message, _TRUNCATE);
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_gpu_memory_info(uint64_t* free_bytes, uint64_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    size_t free_value = 0;
    size_t total_value = 0;
    cudaError_t error = cudaMemGetInfo(&free_value, &total_value);
    if (error != cudaSuccess) return cuda_result(error);
    *free_bytes = static_cast<uint64_t>(free_value);
    *total_bytes = static_cast<uint64_t>(total_value);
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_context_create(void** context) {
    if (context == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    *context = nullptr;
    auto started = std::chrono::steady_clock::now();
    PersistentContext* created = new (std::nothrow) PersistentContext();
    if (created == nullptr) return VF_CUDA_ALLOCATION_FAILED;
    if (created->initialization_error != cudaSuccess) {
        int result = cuda_result(created->initialization_error);
        delete created;
        return result;
    }
    created->last_timings.context_create_ms = elapsed_host_ms(started);
    *context = created;
    return VF_CUDA_OK;
}

template <typename T>
void release_device_buffer(T** pointer, size_t* capacity) {
    if (pointer == nullptr || capacity == nullptr) return;
    visionflow_cuda::free_device(*pointer);
    *pointer = nullptr;
    *capacity = 0;
}

VF_CUDA_API int vf_context_set_timing_enabled(void* context, int enabled) {
    if (context == nullptr || (enabled != 0 && enabled != 1)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    persistent->timing_enabled = enabled != 0;
    const float context_create_ms = persistent->last_timings.context_create_ms;
    persistent->last_timings = {};
    persistent->last_timings.struct_size = sizeof(VfCudaTimingsV1);
    persistent->last_timings.version = 1;
    persistent->last_timings.context_create_ms = context_create_ms;
    persistent->pending_allocation_ms = 0.0f;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_context_last_timings(void* context, VfCudaTimingsV1* timings) {
    if (context == nullptr || timings == nullptr ||
        timings->struct_size != sizeof(VfCudaTimingsV1) || timings->version != 1) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    *timings = static_cast<PersistentContext*>(context)->last_timings;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_context_destroy(void* context) {
    delete static_cast<PersistentContext*>(context);
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_context_stats(
    void* context,
    uint64_t* reserved_bytes,
    uint64_t* allocation_count) {
    if (context == nullptr || reserved_bytes == nullptr || allocation_count == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    const ContextMemoryBreakdown memory = update_context_memory_peak(persistent);
    *reserved_bytes = memory.total();
    *allocation_count = persistent->allocation_count;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_context_memory_stats_v1(
    void* context,
    VfCudaContextMemoryStatsV1* stats) {
    if (context == nullptr || stats == nullptr ||
        stats->struct_size != sizeof(VfCudaContextMemoryStatsV1) || stats->version != 1) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    const ContextMemoryBreakdown memory = update_context_memory_peak(persistent);
    VfCudaContextMemoryStatsV1 output{};
    output.struct_size = sizeof(VfCudaContextMemoryStatsV1);
    output.version = 1;
    output.reserved_bytes = memory.total();
    output.peak_reserved_bytes = persistent->peak_reserved_bytes;
    output.allocation_count = persistent->allocation_count;
    output.plan_bytes = memory.plan_bytes;
    output.resident_bytes = memory.resident_bytes;
    output.template_match_bytes = memory.template_match_bytes;
    output.contour_bytes = memory.contour_bytes;
    output.median_bytes = memory.median_bytes;
    output.gaussian_f32_bytes = memory.gaussian_f32_bytes;
    output.cnr_mask_bytes = memory.cnr_mask_bytes;
    output.cnr_candidate_bytes = memory.cnr_candidate_bytes;
    *stats = output;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_context_memory_stats_v2(
    void* context,
    VfCudaContextMemoryStatsV2* stats) {
    if (context == nullptr || stats == nullptr ||
        stats->struct_size != sizeof(VfCudaContextMemoryStatsV2) || stats->version != 2) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    const ContextMemoryBreakdown memory = update_context_memory_peak(persistent);
    VfCudaContextMemoryStatsV2 output{};
    output.struct_size = sizeof(VfCudaContextMemoryStatsV2);
    output.version = 2;
    output.reserved_bytes = memory.total();
    output.peak_reserved_bytes = persistent->peak_reserved_bytes;
    output.allocation_count = persistent->allocation_count;
    output.plan_bytes = memory.plan_bytes;
    output.resident_bytes = memory.resident_bytes;
    output.template_match_bytes = memory.template_match_bytes;
    output.contour_bytes = memory.contour_bytes;
    output.median_bytes = memory.median_bytes;
    output.gaussian_f32_bytes = memory.gaussian_f32_bytes;
    output.cnr_mask_bytes = memory.cnr_mask_bytes;
    output.cnr_candidate_bytes = memory.cnr_candidate_bytes;
    output.peak_plan_bytes = persistent->peak_memory_bytes[0];
    output.peak_resident_bytes = persistent->peak_memory_bytes[1];
    output.peak_template_match_bytes = persistent->peak_memory_bytes[2];
    output.peak_contour_bytes = persistent->peak_memory_bytes[3];
    output.peak_median_bytes = persistent->peak_memory_bytes[4];
    output.peak_gaussian_f32_bytes = persistent->peak_memory_bytes[5];
    output.peak_cnr_mask_bytes = persistent->peak_memory_bytes[6];
    output.peak_cnr_candidate_bytes = persistent->peak_memory_bytes[7];
    *stats = output;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_context_trim_analysis_scratch(
    void* context,
    uint64_t* released_bytes) {
    if (context == nullptr || released_bytes == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    int result = visionflow_cuda::stream_result(persistent->stream);
    if (result != VF_CUDA_OK) return result;
    const ContextMemoryBreakdown before = update_context_memory_peak(persistent);

    release_device_buffer(&persistent->median_values, &persistent->median_value_capacity);
    release_device_buffer(&persistent->median_keys, &persistent->median_key_capacity);
    release_device_buffer(&persistent->median_sorted_keys, &persistent->median_sorted_key_capacity);
    release_device_buffer(&persistent->median_sort_scratch, &persistent->median_sort_scratch_capacity);
    release_device_buffer(&persistent->median_nan_flag, &persistent->median_nan_flag_capacity);
    release_device_buffer(&persistent->gaussian_f32_input, &persistent->gaussian_f32_input_capacity);
    release_device_buffer(
        &persistent->gaussian_f32_intermediate, &persistent->gaussian_f32_intermediate_capacity);
    release_device_buffer(&persistent->gaussian_f32_output, &persistent->gaussian_f32_output_capacity);
    release_device_buffer(&persistent->cnr_mask_image, &persistent->cnr_mask_image_capacity);
    release_device_buffer(&persistent->cnr_mask_background, &persistent->cnr_mask_background_capacity);
    release_device_buffer(&persistent->cnr_mask_residual, &persistent->cnr_mask_residual_capacity);
    release_device_buffer(&persistent->cnr_mask_absdev, &persistent->cnr_mask_absdev_capacity);
    release_device_buffer(&persistent->cnr_mask_mask, &persistent->cnr_mask_mask_capacity);

    release_device_buffer(&persistent->cand_mask_scratch, &persistent->cand_mask_scratch_capacity);
    release_device_buffer(&persistent->ccl_parent, &persistent->ccl_parent_capacity);
    release_device_buffer(&persistent->cand_words, &persistent->cand_words_capacity);
    release_device_buffer(&persistent->cand_ramp, &persistent->cand_ramp_capacity);
    release_device_buffer(&persistent->cand_foreground, &persistent->cand_foreground_capacity);
    release_device_buffer(&persistent->cand_keys, &persistent->cand_keys_capacity);
    release_device_buffer(&persistent->cand_sorted_keys, &persistent->cand_sorted_keys_capacity);
    release_device_buffer(&persistent->cand_sorted_pixels, &persistent->cand_sorted_pixels_capacity);
    release_device_buffer(&persistent->cand_roots, &persistent->cand_roots_capacity);
    release_device_buffer(&persistent->cand_areas, &persistent->cand_areas_capacity);
    release_device_buffer(&persistent->cand_offsets, &persistent->cand_offsets_capacity);
    release_device_buffer(&persistent->cand_boxes, &persistent->cand_boxes_capacity);
    release_device_buffer(&persistent->cand_keep, &persistent->cand_keep_capacity);
    release_device_buffer(&persistent->cand_kept, &persistent->cand_kept_capacity);
    release_device_buffer(&persistent->cand_windows, &persistent->cand_windows_capacity);
    release_device_buffer(&persistent->cand_window_sizes, &persistent->cand_window_sizes_capacity);
    release_device_buffer(&persistent->cand_gather_offsets, &persistent->cand_gather_offsets_capacity);
    release_device_buffer(&persistent->cand_gather, &persistent->cand_gather_capacity);
    release_device_buffer(&persistent->cand_out_ints, &persistent->cand_out_ints_capacity);
    release_device_buffer(&persistent->cand_out_floats, &persistent->cand_out_floats_capacity);
    release_device_buffer(&persistent->cand_cub_scratch, &persistent->cand_cub_scratch_capacity);
    release_device_buffer(&persistent->cand_flags, &persistent->cand_flags_capacity);
    release_device_buffer(&persistent->cand_values, &persistent->cand_values_capacity);
    release_device_buffer(&persistent->cand_value_offsets, &persistent->cand_value_offsets_capacity);
    release_device_buffer(
        &persistent->cand_background_counts, &persistent->cand_background_counts_capacity);
    release_device_buffer(
        &persistent->cand_background_offsets, &persistent->cand_background_offsets_capacity);
    release_device_buffer(&persistent->cand_segment_ends, &persistent->cand_segment_ends_capacity);
    release_device_buffer(&persistent->cand_seq_start, &persistent->cand_seq_start_capacity);
    release_device_buffer(&persistent->cand_seq_length, &persistent->cand_seq_length_capacity);
    release_device_buffer(&persistent->cand_leaf_counts, &persistent->cand_leaf_counts_capacity);
    release_device_buffer(&persistent->cand_leaf_offsets, &persistent->cand_leaf_offsets_capacity);
    release_device_buffer(&persistent->cand_leaf_start, &persistent->cand_leaf_start_capacity);
    release_device_buffer(&persistent->cand_leaf_length, &persistent->cand_leaf_length_capacity);
    release_device_buffer(&persistent->cand_leaf_sequence, &persistent->cand_leaf_sequence_capacity);
    release_device_buffer(&persistent->cand_leaf_values, &persistent->cand_leaf_values_capacity);
    release_device_buffer(&persistent->cand_seq_mean, &persistent->cand_seq_mean_capacity);
    release_device_buffer(&persistent->cand_seq_std, &persistent->cand_seq_std_capacity);

    const ContextMemoryBreakdown after = context_memory_breakdown(persistent);
    *released_bytes = before.total() - after.total();
    return VF_CUDA_OK;
}

static int context_upload_u8_impl(
    void* context,
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    uint64_t* generation,
    bool allow_file_order) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    const long long stride64 = static_cast<long long>(src_stride);
    const long long absolute_stride = stride64 < 0 ? -stride64 : stride64;
    const long long logical_row_bytes = static_cast<long long>(width) * src_channels;
    if (persistent == nullptr || generation == nullptr ||
        (src_channels != 1 && src_channels != 3) ||
        src == nullptr || width <= 0 || height <= 0 ||
        logical_row_bytes <= 0 || logical_row_bytes > INT_MAX ||
        (src_stride < 0 && !allow_file_order) ||
        absolute_stride < logical_row_bytes) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    reset_timing(persistent, true);
    size_t row_bytes = static_cast<size_t>(width) * src_channels;
    auto allocation_started = std::chrono::steady_clock::now();
    int result = reserve_device(
        &persistent->resident_u8, &persistent->resident_capacity,
        row_bytes * static_cast<size_t>(height), &persistent->allocation_count);
    persistent->last_timings.allocation_ms += elapsed_host_ms(allocation_started);
    if (result != VF_CUDA_OK) return result;
    const uint8_t* file_first_row = src;
    size_t source_stride = static_cast<size_t>(absolute_stride);
    if (src_stride < 0 && height > 1) {
        file_first_row = src + static_cast<long long>(height - 1) * src_stride;
    }
    cudaError_t error = cudaMemcpy2DAsync(
        persistent->resident_u8, row_bytes, file_first_row, source_stride, row_bytes, height,
        cudaMemcpyHostToDevice, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    record_timing_event(persistent, TIMING_AFTER_INPUT);
    if (src_stride < 0 && height > 1) {
        flip_vertical_u8_in_place_kernel<<<
            grid2d(static_cast<int>(row_bytes), height / 2),
            dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
                persistent->resident_u8, static_cast<int>(row_bytes), height);
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
    }
    record_timing_event(persistent, TIMING_AFTER_KERNEL);
    record_timing_event(persistent, TIMING_AFTER_OUTPUT);
    auto synchronize_started = std::chrono::steady_clock::now();
    result = visionflow_cuda::stream_result(persistent->stream);
    persistent->last_timings.synchronize_ms = elapsed_host_ms(synchronize_started);
    if (result == VF_CUDA_OK) finalize_timing(persistent);
    if (result != VF_CUDA_OK) return result;
    persistent->resident_width = width;
    persistent->resident_height = height;
    persistent->resident_channels = src_channels;
    ++persistent->resident_generation;
    if (persistent->resident_generation == 0) ++persistent->resident_generation;
    *generation = persistent->resident_generation;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_context_upload_u8(
    void* context,
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    uint64_t* generation) {
    return context_upload_u8_impl(
        context, src, width, height, src_stride, src_channels, generation, false);
}

VF_CUDA_API int vf_context_upload_u8_file_order(
    void* context,
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    uint64_t* generation) {
    return context_upload_u8_impl(
        context, src, width, height, src_stride, src_channels, generation, true);
}

VF_CUDA_API int vf_host_register_u8(void* context, uint8_t* host, uint64_t bytes) {
    // The DLL is built for x64 only, so every uint64_t byte count fits size_t.
    if (context == nullptr || host == nullptr || bytes == 0) return VF_CUDA_INVALID_ARGUMENT;
    cudaError_t error = cudaHostRegister(host, static_cast<size_t>(bytes), cudaHostRegisterDefault);
    return error == cudaSuccess ? VF_CUDA_OK : cuda_result(error);
}

VF_CUDA_API int vf_host_unregister_u8(void* context, uint8_t* host) {
    if (context == nullptr || host == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    cudaError_t error = cudaHostUnregister(host);
    return error == cudaSuccess ? VF_CUDA_OK : cuda_result(error);
}

VF_CUDA_API int vf_roi_batch_create(
    void* context,
    uint64_t generation,
    const VfRoiV1* rois,
    int roi_count,
    void** batch) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || batch == nullptr || rois == nullptr || roi_count <= 0 ||
        roi_count > 65535 || generation == 0 || generation != persistent->resident_generation ||
        persistent->resident_u8 == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    *batch = nullptr;
    const int width = rois[0].width;
    const int height = rois[0].height;
    if (width <= 0 || height <= 0 || width > persistent->resident_width ||
        height > persistent->resident_height) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    for (int index = 0; index < roi_count; ++index) {
        const VfRoiV1& roi = rois[index];
        if (roi.struct_size != sizeof(VfRoiV1) || roi.width != width || roi.height != height ||
            roi.x < 0 || roi.y < 0 || roi.x > persistent->resident_width - width ||
            roi.y > persistent->resident_height - height) {
            return VF_CUDA_INVALID_ARGUMENT;
        }
    }
    size_t roi_bytes = static_cast<size_t>(width) * height * persistent->resident_channels;
    if (roi_bytes == 0 || static_cast<size_t>(roi_count) > SIZE_MAX / roi_bytes ||
        static_cast<size_t>(roi_count) > SIZE_MAX / sizeof(VfRoiV1)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    NativeRoiBatch* created = new (std::nothrow) NativeRoiBatch();
    if (created == nullptr) return VF_CUDA_ALLOCATION_FAILED;
    created->context = persistent;
    created->count = roi_count;
    created->width = width;
    created->height = height;
    created->channels = persistent->resident_channels;
    cudaError_t error = cudaMalloc(&created->data, roi_bytes * roi_count);
    if (error == cudaSuccess) {
        error = cudaMalloc(&created->device_rois, sizeof(VfRoiV1) * roi_count);
    }
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            created->device_rois, rois, sizeof(VfRoiV1) * roi_count,
            cudaMemcpyHostToDevice, persistent->stream);
    }
    if (error != cudaSuccess) {
        delete created;
        return cuda_result(error);
    }
    dim3 grid(
        (width + BLOCK_X - 1) / BLOCK_X,
        (height + BLOCK_Y - 1) / BLOCK_Y,
        roi_count);
    gather_roi_batch_kernel<<<grid, dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        persistent->resident_u8, persistent->resident_width, persistent->resident_channels,
        created->device_rois, created->data, width, height);
    int result = visionflow_cuda::kernel_launch_result();
    if (result == VF_CUDA_OK) result = visionflow_cuda::stream_result(persistent->stream);
    if (result != VF_CUDA_OK) {
        delete created;
        return result;
    }
    *batch = created;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_roi_batch_info(
    void* batch,
    int* roi_count,
    int* width,
    int* height,
    int* channels) {
    NativeRoiBatch* native = static_cast<NativeRoiBatch*>(batch);
    if (native == nullptr || roi_count == nullptr || width == nullptr || height == nullptr ||
        channels == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    *roi_count = native->count;
    *width = native->width;
    *height = native->height;
    *channels = native->channels;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_roi_batch_download_u8(
    void* batch,
    int roi_index,
    uint8_t* dst,
    int dst_stride,
    int dst_channels) {
    NativeRoiBatch* native = static_cast<NativeRoiBatch*>(batch);
    if (native == nullptr || native->context == nullptr || roi_index < 0 ||
        roi_index >= native->count || dst_channels != native->channels ||
        !visionflow_cuda::valid_image(
            dst, native->width, native->height, dst_stride, dst_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    size_t row_bytes = static_cast<size_t>(native->width) * native->channels;
    size_t roi_bytes = row_bytes * native->height;
    cudaError_t error = cudaMemcpy2DAsync(
        dst, dst_stride, native->data + static_cast<size_t>(roi_index) * roi_bytes,
        row_bytes, row_bytes, native->height, cudaMemcpyDeviceToHost,
        native->context->stream);
    if (error != cudaSuccess) return cuda_result(error);
    return visionflow_cuda::stream_result(native->context->stream);
}

VF_CUDA_API int vf_roi_batch_destroy(void* batch) {
    NativeRoiBatch* native = static_cast<NativeRoiBatch*>(batch);
    if (native == nullptr) return VF_CUDA_OK;
    PersistentContext* context = native->context;
    auto started = std::chrono::steady_clock::now();
    delete native;
    if (context != nullptr) context->last_timings.free_ms = elapsed_host_ms(started);
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_plan_query(
    const VfPlanDescV1* desc,
    int width,
    int height,
    char* reason,
    int reason_capacity) {
    return validate_plan_desc(
        desc, width, height, nullptr, nullptr, nullptr, reason, reason_capacity);
}

VF_CUDA_API int vf_plan_create(
    void* context,
    const VfPlanDescV1* desc,
    int width,
    int height,
    void** plan) {
    if (context == nullptr || plan == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    *plan = nullptr;
    int output_channels = 0;
    int output_width = 0;
    int output_height = 0;
    int result = validate_plan_desc(
        desc, width, height, &output_channels, &output_width, &output_height, nullptr, 0);
    if (result != VF_CUDA_OK) return result;

    NativePlan* created = new (std::nothrow) NativePlan();
    if (created == nullptr) return VF_CUDA_ALLOCATION_FAILED;
    created->context = static_cast<PersistentContext*>(context);
    created->width = width;
    created->height = height;
    created->output_width = output_width;
    created->output_height = output_height;
    created->input_channels = desc->input_channels;
    created->output_channels = output_channels;
    try {
        created->operators.assign(desc->operators, desc->operators + desc->operator_count);
    } catch (const std::bad_alloc&) {
        delete created;
        return VF_CUDA_ALLOCATION_FAILED;
    }
    auto allocation_started = std::chrono::steady_clock::now();
    result = reserve_plan_buffers(created->context, *created);
    int current_width = width;
    int current_height = height;
    for (const VfPlanOperatorV1& op : created->operators) {
        if (result != VF_CUDA_OK) break;
        if (op.kind != VF_PLAN_RESIZE_AREA) continue;
        std::unique_ptr<AreaResizeTables> tables(new (std::nothrow) AreaResizeTables());
        if (!tables) {
            result = VF_CUDA_ALLOCATION_FAILED;
            break;
        }
        result = prepare_area_resize(
            current_width, current_height, op.int_params[0], op.int_params[1], tables.get(),
            &created->context->allocation_count);
        current_width = op.int_params[0];
        current_height = op.int_params[1];
        if (result == VF_CUDA_OK) {
            try {
                created->area_resizes.push_back(std::move(tables));
            } catch (const std::bad_alloc&) {
                result = VF_CUDA_ALLOCATION_FAILED;
            }
        }
    }
    created->context->pending_allocation_ms += elapsed_host_ms(allocation_started);
    if (result != VF_CUDA_OK) {
        delete created;
        return result;
    }
    *plan = created;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_plan_execute(
    void* plan,
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    uint8_t* dst,
    int dst_stride,
    int dst_channels) {
    NativePlan* compiled = static_cast<NativePlan*>(plan);
    if (compiled == nullptr || compiled->context == nullptr || width != compiled->width ||
        height != compiled->height || src_channels != compiled->input_channels ||
        dst_channels != compiled->output_channels ||
        !visionflow_cuda::valid_image(src, width, height, src_stride, src_channels) ||
        !visionflow_cuda::valid_image(
            dst, compiled->output_width, compiled->output_height, dst_stride, dst_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    PersistentContext* context = compiled->context;
    reset_timing(context, true);
    const size_t source_row_bytes = static_cast<size_t>(width) * src_channels;
    cudaError_t error = cudaMemcpy2DAsync(
        context->u8[0], source_row_bytes, src, src_stride, source_row_bytes, height,
        cudaMemcpyHostToDevice, context->stream);
    if (error != cudaSuccess) return cuda_result(error);
    record_timing_event(context, TIMING_AFTER_INPUT);

    return execute_linear_plan_device(
        compiled, context->u8[0], dst, dst_stride, dst_channels);
}

VF_CUDA_API int vf_plan_destroy(void* plan) {
    delete static_cast<NativePlan*>(plan);
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_plan_execute_roi(
    void* plan,
    uint64_t generation,
    int x,
    int y,
    uint8_t* dst,
    int dst_stride,
    int dst_channels) {
    NativePlan* compiled = static_cast<NativePlan*>(plan);
    if (compiled == nullptr || compiled->context == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    PersistentContext* context = compiled->context;
    if (generation == 0 || generation != context->resident_generation ||
        context->resident_channels != compiled->input_channels || x < 0 || y < 0 ||
        x + compiled->width > context->resident_width ||
        y + compiled->height > context->resident_height ||
        dst_channels != compiled->output_channels ||
        !visionflow_cuda::valid_image(
            dst, compiled->output_width, compiled->output_height, dst_stride, dst_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    reset_timing(context, false);
    size_t resident_pitch = static_cast<size_t>(context->resident_width) * context->resident_channels;
    size_t roi_row_bytes = static_cast<size_t>(compiled->width) * compiled->input_channels;
    const uint8_t* source = context->resident_u8 +
        static_cast<size_t>(y) * resident_pitch + static_cast<size_t>(x) * compiled->input_channels;
    cudaError_t error = cudaMemcpy2DAsync(
        context->u8[0], roi_row_bytes, source, resident_pitch, roi_row_bytes, compiled->height,
        cudaMemcpyDeviceToDevice, context->stream);
    if (error != cudaSuccess) return cuda_result(error);
    record_timing_event(context, TIMING_AFTER_INPUT);
    return execute_linear_plan_device(
        compiled, context->u8[0], dst, dst_stride, dst_channels);
}

VF_CUDA_API int vf_plan_find_contours_roi(
    void* plan, uint64_t generation, int x, int y, int mode,
    int* out_contour_count, int* out_point_count) {
    NativePlan* compiled = static_cast<NativePlan*>(plan);
    if (compiled == nullptr || compiled->context == nullptr ||
        out_contour_count == nullptr || out_point_count == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    PersistentContext* context = compiled->context;
    if (generation == 0 || generation != context->resident_generation ||
        context->resident_channels != compiled->input_channels ||
        compiled->output_channels != 1 ||
        compiled->output_width != compiled->width ||
        compiled->output_height != compiled->height ||
        x < 0 || y < 0 || x + compiled->width > context->resident_width ||
        y + compiled->height > context->resident_height) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    reset_timing(context, false);
    const size_t resident_pitch =
        static_cast<size_t>(context->resident_width) * context->resident_channels;
    const size_t roi_row_bytes =
        static_cast<size_t>(compiled->width) * compiled->input_channels;
    const uint8_t* source = context->resident_u8 +
        static_cast<size_t>(y) * resident_pitch + static_cast<size_t>(x) * compiled->input_channels;
    cudaError_t error = cudaMemcpy2DAsync(
        context->u8[0], roi_row_bytes, source, resident_pitch,
        roi_row_bytes, compiled->height, cudaMemcpyDeviceToDevice, context->stream);
    if (error != cudaSuccess) return cuda_result(error);
    record_timing_event(context, TIMING_AFTER_INPUT);
    uint8_t* mask = nullptr;
    int result = execute_linear_plan_device(
        compiled, context->u8[0], nullptr, 0, 1, &mask);
    if (result != VF_CUDA_OK) return result;

    // Reuse the verified OpenCV-equivalent contour tracer without replacing the context-owned
    // colour resident allocation. The temporary metadata swap is safe because the public runtime
    // serializes every context call and the trace completes before the original fields return.
    uint8_t* original_resident = context->resident_u8;
    const int original_width = context->resident_width;
    const int original_height = context->resident_height;
    const int original_channels = context->resident_channels;
    context->resident_u8 = mask;
    context->resident_width = compiled->output_width;
    context->resident_height = compiled->output_height;
    context->resident_channels = 1;
    result = vf_find_contours_u8(
        context, generation, 0, 0,
        compiled->output_width, compiled->output_height, mode,
        out_contour_count, out_point_count);
    context->resident_u8 = original_resident;
    context->resident_width = original_width;
    context->resident_height = original_height;
    context->resident_channels = original_channels;
    return result;
}

VF_CUDA_API int vf_dag_plan_query(
    const VfDagPlanDescV1* desc,
    int width,
    int height,
    char* reason,
    int reason_capacity) {
    return validate_dag_plan_desc(desc, width, height, nullptr, reason, reason_capacity);
}

VF_CUDA_API int vf_dag_plan_create(
    void* context,
    const VfDagPlanDescV1* desc,
    int width,
    int height,
    void** plan) {
    if (context == nullptr || plan == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    *plan = nullptr;
    std::vector<int> node_channels;
    int result = validate_dag_plan_desc(desc, width, height, &node_channels, nullptr, 0);
    if (result != VF_CUDA_OK) return result;
    NativeDagPlan* created = new (std::nothrow) NativeDagPlan();
    if (created == nullptr) return VF_CUDA_ALLOCATION_FAILED;
    created->context = static_cast<PersistentContext*>(context);
    created->width = width;
    created->height = height;
    created->input_channels = desc->input_channels;
    try {
        created->operators.assign(desc->operators, desc->operators + desc->operator_count);
        created->node_channels = std::move(node_channels);
        created->output_nodes.assign(desc->output_nodes, desc->output_nodes + desc->output_count);
    } catch (const std::bad_alloc&) {
        delete created;
        return VF_CUDA_ALLOCATION_FAILED;
    }
    auto allocation_started = std::chrono::steady_clock::now();
    result = reserve_dag_plan_buffers(created->context, *created);
    created->context->pending_allocation_ms += elapsed_host_ms(allocation_started);
    if (result != VF_CUDA_OK) {
        delete created;
        return result;
    }
    *plan = created;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_dag_plan_execute(
    void* plan,
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    const VfDagOutputV1* outputs,
    int output_count) {
    NativeDagPlan* compiled = static_cast<NativeDagPlan*>(plan);
    if (compiled == nullptr || compiled->context == nullptr || width != compiled->width ||
        height != compiled->height || src_channels != compiled->input_channels ||
        output_count != static_cast<int>(compiled->output_nodes.size()) || outputs == nullptr ||
        !visionflow_cuda::valid_image(src, width, height, src_stride, src_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    for (int index = 0; index < output_count; ++index) {
        int node = compiled->output_nodes[index];
        if (outputs[index].struct_size != sizeof(VfDagOutputV1) || outputs[index].node != node ||
            outputs[index].channels != compiled->node_channels[node] ||
            !visionflow_cuda::valid_image(
                outputs[index].data, width, height, outputs[index].stride, outputs[index].channels)) {
            return VF_CUDA_INVALID_ARGUMENT;
        }
    }
    PersistentContext* context = compiled->context;
    reset_timing(context, true);
    const size_t source_row_bytes = static_cast<size_t>(width) * src_channels;
    cudaError_t error = cudaMemcpy2DAsync(
        context->u8[0], source_row_bytes, src, src_stride, source_row_bytes, height,
        cudaMemcpyHostToDevice, context->stream);
    if (error != cudaSuccess) return cuda_result(error);
    record_timing_event(context, TIMING_AFTER_INPUT);

    return execute_dag_plan_device(
        compiled, context->u8[0], outputs, output_count);
}

VF_CUDA_API int vf_dag_plan_destroy(void* plan) {
    delete static_cast<NativeDagPlan*>(plan);
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_dag_plan_execute_roi(
    void* plan,
    uint64_t generation,
    int x,
    int y,
    const VfDagOutputV1* outputs,
    int output_count) {
    NativeDagPlan* compiled = static_cast<NativeDagPlan*>(plan);
    if (compiled == nullptr || compiled->context == nullptr || outputs == nullptr ||
        output_count != static_cast<int>(compiled->output_nodes.size())) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    PersistentContext* context = compiled->context;
    if (generation == 0 || generation != context->resident_generation ||
        context->resident_channels != compiled->input_channels || x < 0 || y < 0 ||
        x + compiled->width > context->resident_width ||
        y + compiled->height > context->resident_height) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    for (int index = 0; index < output_count; ++index) {
        int node = compiled->output_nodes[index];
        if (outputs[index].struct_size != sizeof(VfDagOutputV1) || outputs[index].node != node ||
            outputs[index].channels != compiled->node_channels[node] ||
            !visionflow_cuda::valid_image(
                outputs[index].data, compiled->width, compiled->height,
                outputs[index].stride, outputs[index].channels)) {
            return VF_CUDA_INVALID_ARGUMENT;
        }
    }
    reset_timing(context, false);
    size_t resident_pitch = static_cast<size_t>(context->resident_width) * context->resident_channels;
    size_t roi_row_bytes = static_cast<size_t>(compiled->width) * compiled->input_channels;
    const uint8_t* source = context->resident_u8 +
        static_cast<size_t>(y) * resident_pitch + static_cast<size_t>(x) * compiled->input_channels;
    cudaError_t error = cudaMemcpy2DAsync(
        context->u8[0], roi_row_bytes, source, resident_pitch, roi_row_bytes, compiled->height,
        cudaMemcpyDeviceToDevice, context->stream);
    if (error != cudaSuccess) return cuda_result(error);
    record_timing_event(context, TIMING_AFTER_INPUT);
    return execute_dag_plan_device(
        compiled, context->u8[0], outputs, output_count);
}

VF_CUDA_API int vf_bgr_to_gray_u8(const uint8_t* src, int w, int h, int stride, int sc, uint8_t* dst, int dstride, int dc) {
    if (sc != 3 || dc != 1) return VF_CUDA_INVALID_ARGUMENT;
    uint8_t *ds = nullptr, *dd = nullptr;
    int result = alloc_copy(src, w, h, stride, sc, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    bgr_gray_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y)>>>(ds, dd, w, h);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, 1, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_bgr_to_rgb_u8(const uint8_t* src, int w, int h, int stride, int sc, uint8_t* dst, int dstride, int dc) {
    if (sc != 3 || dc != 3) return VF_CUDA_INVALID_ARGUMENT;
    uint8_t *ds = nullptr, *dd = nullptr;
    int result = alloc_copy(src, w, h, stride, sc, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h * 3);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    bgr_rgb_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y)>>>(ds, dd, w, h);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, 3, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_crop_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int x,int y,int cw,int ch) {
    if (sc != dc || (sc != 1 && sc != 3) || x < 0 || y < 0 || cw <= 0 || ch <= 0 || x + cw > w || y + ch > h) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    uint8_t *ds = nullptr, *dd = nullptr;
    int result = alloc_copy(src, w, h, stride, sc, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(cw) * ch * sc);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    crop_kernel<<<grid2d(cw, ch), dim3(BLOCK_X, BLOCK_Y)>>>(ds, dd, w, x, y, cw, ch, sc);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, cw, ch, sc, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_resize_gray_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int dw,int dh) {
    if (sc != 1 || dc != 1 || dw <= 0 || dh <= 0) return VF_CUDA_INVALID_ARGUMENT;
    uint8_t *ds = nullptr, *dd = nullptr;
    int result = alloc_copy(src, w, h, stride, 1, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(dw) * dh);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    if (dw <= w && dh <= h) {
        AreaResizeTables tables;
        result = prepare_area_resize(w, h, dw, dh, &tables, nullptr);
        if (result == VF_CUDA_OK) {
            launch_area_resize(tables, ds, dd, w, dw, dh);
            result = visionflow_cuda::kernel_result();
        }
    } else {
        resize_gray_kernel<<<grid2d(dw, dh), dim3(BLOCK_X, BLOCK_Y)>>>(ds, dd, w, h, dw, dh);
        result = visionflow_cuda::kernel_result();
    }
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, dw, dh, 1, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_gaussian_blur_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int kernel) {
    if (sc != dc || (sc != 1 && sc != 3) || kernel < 3 || kernel % 2 == 0 || kernel > MAX_GAUSSIAN_KERNEL) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    uint8_t *ds = nullptr, *dd = nullptr;
    uint32_t* intermediate = nullptr;
    int result = alloc_copy(src, w, h, stride, sc, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h * sc);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    cudaError_t error = cudaMalloc(
        &intermediate, static_cast<size_t>(w) * h * sc * sizeof(uint32_t));
    if (error != cudaSuccess) {
        visionflow_cuda::free_device(dd);
        visionflow_cuda::free_device(ds);
        return cuda_result(error);
    }

    int radius = 0;
    result = prepare_gaussian_weights(kernel, &radius);
    if (result != VF_CUDA_OK) {
        visionflow_cuda::free_device(intermediate);
        visionflow_cuda::free_device(dd);
        visionflow_cuda::free_device(ds);
        return result;
    }
    launch_gaussian(ds, intermediate, dd, w, h, sc, radius);
    result = visionflow_cuda::kernel_result();
    visionflow_cuda::free_device(intermediate);
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, sc, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_threshold_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int threshold,int max_value,int invert) {
    if (sc != 1 || dc != 1 || threshold < 0 || threshold > 255 || max_value < 0 || max_value > 255) return VF_CUDA_INVALID_ARGUMENT;
    uint8_t *ds = nullptr, *dd = nullptr;
    int result = alloc_copy(src, w, h, stride, 1, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    int count = w * h;
    threshold_kernel<<<(count + 255) / 256, 256>>>(ds, dd, count, threshold, max_value, invert);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, 1, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_adaptive_mean_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int block,float c,int max_value,int invert) {
    if (w <= 0 || h <= 0 || sc != 1 || dc != 1 || block < 3 || block % 2 == 0 ||
        max_value < 0 || max_value > 255 || !std::isfinite(c)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    size_t scratch_count = 0;
    int result = adaptive_layout(w, h, block, &scratch_count);
    if (result != VF_CUDA_OK) return result;
    uint8_t *ds = nullptr, *dd = nullptr;
    uint32_t* horizontal = nullptr;
    result = alloc_copy(src, w, h, stride, 1, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    cudaError_t error = cudaMalloc(&horizontal, scratch_count * sizeof(uint32_t));
    if (error != cudaSuccess) {
        visionflow_cuda::free_device(horizontal);
        visionflow_cuda::free_device(dd);
        visionflow_cuda::free_device(ds);
        return cuda_result(error);
    }
    launch_adaptive_mean(ds, horizontal, dd, w, h, block, c, max_value, invert);
    result = visionflow_cuda::kernel_result();
    visionflow_cuda::free_device(horizontal);
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, 1, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_preprocess_401_2_u8(
    void* context,
    const uint8_t* src,
    int w,
    int h,
    int stride,
    int sc,
    uint8_t* dst,
    int dstride,
    int gaussian_kernel,
    int adaptive_block,
    float adaptive_c,
    int max_value,
    int invert) {
    if (context == nullptr || w <= 0 || h <= 0 || (sc != 1 && sc != 3) ||
        w > INT_MAX / sc || !visionflow_cuda::valid_image(src, w, h, stride, sc) ||
        !visionflow_cuda::valid_image(dst, w, h, dstride, 1) ||
        max_value < 0 || max_value > 255 || !std::isfinite(adaptive_c)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (static_cast<size_t>(w) > SIZE_MAX / static_cast<size_t>(h)) return VF_CUDA_INVALID_ARGUMENT;
    size_t pixel_count = static_cast<size_t>(w) * static_cast<size_t>(h);
    if (pixel_count > SIZE_MAX / static_cast<size_t>(sc)) return VF_CUDA_INVALID_ARGUMENT;
    size_t source_count = pixel_count * static_cast<size_t>(sc);

    int radius = 0;
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    int result = prepare_gaussian_weights(
        gaussian_kernel, &radius, persistent->stream);
    if (result != VF_CUDA_OK) return result;
    size_t adaptive_scratch_count = 0;
    result = adaptive_layout(
        w,
        h,
        adaptive_block,
        &adaptive_scratch_count);
    if (result != VF_CUDA_OK) return result;

    result = reserve_device(
        &persistent->u8[0], &persistent->u8_capacity[0], source_count, &persistent->allocation_count);
    if (result == VF_CUDA_OK) {
        result = reserve_device(
            &persistent->u8[1], &persistent->u8_capacity[1], pixel_count, &persistent->allocation_count);
    }
    if (result == VF_CUDA_OK) {
        result = reserve_device(
            &persistent->u8[2], &persistent->u8_capacity[2], pixel_count, &persistent->allocation_count);
    }
    if (result == VF_CUDA_OK) {
        result = reserve_device(
            &persistent->gaussian_buffer,
            &persistent->gaussian_capacity,
            std::max(pixel_count, adaptive_scratch_count),
            &persistent->allocation_count);
    }
    if (result != VF_CUDA_OK) return result;

    size_t source_row_bytes = static_cast<size_t>(w) * static_cast<size_t>(sc);
    cudaError_t error = cudaMemcpy2DAsync(
        persistent->u8[0],
        source_row_bytes,
        src,
        stride,
        source_row_bytes,
        h,
        cudaMemcpyHostToDevice,
        persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);

    uint8_t* gray = persistent->u8[0];
    if (sc == 3) {
        gray = persistent->u8[1];
        bgr_gray_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
            persistent->u8[0], gray, w, h);
    }
    gaussian_horizontal_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        gray, persistent->gaussian_buffer, w, h, 1, radius);
    gaussian_vertical_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        persistent->gaussian_buffer, gray, w, h, 1, radius);
    launch_adaptive_mean(
        gray,
        persistent->gaussian_buffer,
        persistent->u8[2],
        w,
        h,
        adaptive_block,
        adaptive_c,
        max_value,
        invert,
        persistent->stream);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    error = cudaMemcpy2DAsync(
        dst,
        dstride,
        persistent->u8[2],
        static_cast<size_t>(w),
        static_cast<size_t>(w),
        h,
        cudaMemcpyDeviceToHost,
        persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    return visionflow_cuda::stream_result(persistent->stream);
}

VF_CUDA_API int vf_morphology_rect_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int operation,int kernel,int iterations) {
    if (sc != dc || (sc != 1 && sc != 3) || kernel < 3 || kernel % 2 == 0 || iterations < 1 || operation < VF_MORPH_OPEN || operation > VF_MORPH_ERODE) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    uint8_t *a = nullptr, *b = nullptr;
    int result = alloc_copy(src, w, h, stride, sc, &a);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&b, static_cast<size_t>(w) * h * sc);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(a); return result; }
    auto pass = [&](int dilate) {
        launch_morph_pass(a, b, w, h, sc, kernel / 2, dilate);
        std::swap(a, b);
    };
    if (operation == VF_MORPH_OPEN) {
        for (int i = 0; i < iterations; ++i) pass(0);
        for (int i = 0; i < iterations; ++i) pass(1);
    } else if (operation == VF_MORPH_CLOSE) {
        for (int i = 0; i < iterations; ++i) pass(1);
        for (int i = 0; i < iterations; ++i) pass(0);
    } else {
        for (int i = 0; i < iterations; ++i) pass(operation == VF_MORPH_DILATE);
    }
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, sc, a);
    else visionflow_cuda::free_device(a);
    visionflow_cuda::free_device(b);
    return result;
}

VF_CUDA_API int vf_match_template_gray_u8(
    void* context,
    uint64_t generation,
    int search_x, int search_y, int search_width, int search_height,
    const uint8_t* templ, int template_width, int template_height,
    int* out_match, float* out_score) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || templ == nullptr || out_match == nullptr || out_score == nullptr ||
        generation == 0 || generation != persistent->resident_generation ||
        persistent->resident_u8 == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (search_x < 0 || search_y < 0 || search_width <= 0 || search_height <= 0 ||
        search_x > persistent->resident_width - search_width ||
        search_y > persistent->resident_height - search_height) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (template_width <= 0 || template_height <= 0 ||
        template_width > search_width || template_height > search_height ||
        static_cast<long long>(template_width) * template_height > INT_MAX) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int output_width = search_width - template_width + 1;
    const int output_height = search_height - template_height + 1;
    if (output_width <= 0 || output_height <= 0) return VF_CUDA_INVALID_ARGUMENT;

    const size_t template_bytes = static_cast<size_t>(template_width) * template_height;
    // The prefix planes are indexed by output_width * roi_height, and the context caches them with
    // grow-only semantics, so reserve the ROI area: it bounds every shape this call can index and
    // keeps a later, wider search from reading past an earlier smaller allocation.
    size_t plane_elements = static_cast<size_t>(search_width) * search_height;
    if (plane_elements < static_cast<size_t>(output_width) * output_height) {
        plane_elements = static_cast<size_t>(output_width) * output_height;
    }
    if (plane_elements > SIZE_MAX / sizeof(long long) / MATCH_PLANE_COUNT) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    cudaError_t error = cudaSuccess;
    int result = reserve_device(
        &persistent->u8[MATCH_TEMPLATE_BUFFER], &persistent->u8_capacity[MATCH_TEMPLATE_BUFFER],
        template_bytes, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    // The gray ROI is fully rewritten by this call, so it must be exactly the requested size:
    // a larger leftover buffer from an earlier call would keep the old row pitch.
    update_context_memory_peak(persistent);
    result = reserve_exact(
        &persistent->u8[MATCH_ROI_BUFFER], &persistent->u8_capacity[MATCH_ROI_BUFFER],
        static_cast<size_t>(search_width) * search_height, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    // planes: sum prefix, square prefix
    for (int plane = 0; plane < MATCH_PLANE_COUNT; ++plane) {
        result = reserve_device(
            &persistent->match_plane[plane], &persistent->match_plane_capacity[plane],
            plane_elements, &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
    }
    result = reserve_device(
        &persistent->match_candidates, &persistent->match_candidate_capacity,
        static_cast<size_t>(match_slot_count(output_width)), &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    persistent->match_candidate_output_width = output_width;
    long long* candidate_storage = persistent->match_candidates;
    unsigned long long* best_keys = reinterpret_cast<unsigned long long*>(candidate_storage);
    double* candidate_scores =
        reinterpret_cast<double*>(candidate_storage + match_score_offset(output_width));
    int* candidate_ys = reinterpret_cast<int*>(candidate_storage + match_row_offset(output_width));
    // Result slots live in the same grow-only block: two int32 for the winner origin, one float
    // score and one packed best key, all int64-aligned.
    const int result_offset = match_result_offset(output_width);
    const int key_offset = match_result_offset(output_width) + 1;
    int* match_xy_device = reinterpret_cast<int*>(candidate_storage + result_offset);
    float* match_score_device = reinterpret_cast<float*>(match_xy_device + 2);
    unsigned long long* best_key = reinterpret_cast<unsigned long long*>(candidate_storage + key_offset);

    // Template statistics on the host: the template is small and constant per Recipe.
    double template_sum = 0.0;
    double template_square_sum = 0.0;
    for (int row = 0; row < template_height; ++row) {
        for (int column = 0; column < template_width; ++column) {
            const int value = templ[static_cast<size_t>(row) * template_width + column];
            template_sum += value;
            template_square_sum += static_cast<double>(value) * value;
        }
    }
    const double template_pixels = static_cast<double>(template_width) * template_height;
    const double template_mean = template_sum / template_pixels;
    const double template_variance = template_square_sum / template_pixels - template_mean * template_mean;
    if (!(template_variance > 1e-12)) {
        // core/tiler.py::_find_grid_anchor switches to TM_SQDIFF_NORMED when the template has no
        // contrast. The extension's difference path is not equivalent to OpenCV yet (it reports
        // scores outside [0, 1] and can pick a neighbouring column), so the caller restarts this
        // step on the CPU reference instead of receiving a wrong anchor. Production anchor
        // templates are structured patterns, so this affects the flat-template edge case only.
        return VF_CUDA_UNSUPPORTED;
    }
    if (output_width > MATCH_COORD_MASK) return VF_CUDA_INVALID_ARGUMENT;
    reset_timing(persistent, true);
    error = cudaMemcpyAsync(
        persistent->u8[MATCH_TEMPLATE_BUFFER], templ, template_bytes,
        cudaMemcpyHostToDevice, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);

    // Gray the search ROI on the device: the resident image is BGR, the reference works on gray,
    // and the weights are the ones vf_bgr_to_gray_u8 already uses. The ROI-aware kernel is needed
    // because a resident sub-rectangle is not tightly packed.
    const size_t resident_pitch =
        static_cast<size_t>(persistent->resident_width) * persistent->resident_channels;
    const uint8_t* roi_source = persistent->resident_u8 +
        static_cast<size_t>(search_y) * resident_pitch +
        static_cast<size_t>(search_x) * persistent->resident_channels;
    if (persistent->resident_channels == 3) {
        bgr_gray_roi_kernel<<<
            grid2d(search_width, search_height), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
            roi_source, persistent->resident_width, 0, 0,
            persistent->u8[MATCH_ROI_BUFFER], search_width, search_height);
    } else {
        error = cudaMemcpy2DAsync(
            persistent->u8[MATCH_ROI_BUFFER], static_cast<size_t>(search_width), roi_source,
            resident_pitch, static_cast<size_t>(search_width), search_height,
            cudaMemcpyDeviceToDevice, persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
    }
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    // Shrink the tile until its shared footprint fits, because a large template widens the halo.
    int tile_cols = MATCH_TILE_COLS;
    int tile_rows = MATCH_TILE_ROWS;
    size_t shared_bytes = 0;
    const size_t halo = static_cast<size_t>(template_height - 1);
    for (;;) {
        shared_bytes = (static_cast<size_t>(tile_rows) + halo) *
                       (static_cast<size_t>(tile_cols) + static_cast<size_t>(template_width - 1));
        if (shared_bytes <= MATCH_SHARED_LIMIT_BYTES) break;
        // Shrink the larger dimension first, and never below a one-row-tall, 8-column-wide tile,
        // which keeps the block occupied while reducing the halo overhead.
        if (tile_cols >= tile_rows && tile_cols > 8) {
            tile_cols = tile_cols > 16 ? tile_cols / 2 : tile_cols - 4;
            continue;
        }
        if (tile_rows > 1) {
            tile_rows = tile_rows > 2 ? tile_rows / 2 : tile_rows - 1;
            continue;
        }
        break;
    }
    // When the ROI tile had to shrink below the tile we would like, the template-in-shared layout
    // is the better trade: the template is the small operand and the ROI streams through L2.
    const bool use_shared_template =
        (tile_cols < MATCH_TILE_COLS || tile_rows < MATCH_TILE_ROWS) &&
        template_bytes <= MATCH_SHARED_LIMIT_BYTES;
    if (shared_bytes > MATCH_SHARED_LIMIT_BYTES && !use_shared_template) {
        // Beyond this the template height alone exceeds the block budget, so the caller restarts
        // localization on the CPU reference instead of receiving a wrong anchor.
        return VF_CUDA_UNSUPPORTED;
    }
    if (cudaFuncSetAttribute(
            match_score_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(MATCH_SHARED_LIMIT_BYTES)) != cudaSuccess) {
        return cuda_result(cudaGetLastError());
    }
    if (use_shared_template) {
        if (cudaFuncSetAttribute(
                match_score_shared_template_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(template_bytes)) != cudaSuccess) {
            return cuda_result(cudaGetLastError());
        }
        const dim3 shared_grid(
            static_cast<unsigned int>((output_width + MATCH_BLOCK_X - 1) / MATCH_BLOCK_X),
            static_cast<unsigned int>((output_height + MATCH_BLOCK_Y - 1) / MATCH_BLOCK_Y));
        const dim3 shared_block(MATCH_BLOCK_X, MATCH_BLOCK_Y);
        error = cudaMemsetAsync(
            best_keys, 0, sizeof(unsigned long long) * match_candidate_slots(output_width),
            persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        match_score_shared_template_kernel<<<
            shared_grid, shared_block, template_bytes, persistent->stream>>>(
            persistent->u8[MATCH_ROI_BUFFER], search_width,
            output_width, output_height, template_width, template_height,
            template_width * template_height, MATCH_BLOCK_X,
            template_mean, template_variance,
            persistent->u8[MATCH_TEMPLATE_BUFFER], best_keys);
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
    } else {
    const int tiles_x = (output_width + tile_cols - 1) / tile_cols;
    const int tiles_y = (output_height + tile_rows - 1) / tile_rows;
    const dim3 score_grid(static_cast<unsigned int>(tiles_x) * static_cast<unsigned int>(tiles_y));
    const dim3 score_block(MATCH_BLOCK_X, MATCH_BLOCK_Y);
        // The per-column key slots are the compare-and-swap targets, so they must start empty.
        error = cudaMemsetAsync(
            best_keys, 0, sizeof(unsigned long long) * match_candidate_slots(output_width),
            persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        match_score_kernel<<<score_grid, score_block, shared_bytes, persistent->stream>>>(
            persistent->u8[MATCH_ROI_BUFFER], search_width,
            output_width, output_height, template_width, template_height,
            template_width * template_height, tile_cols, tile_rows,
            template_mean, template_variance,
            persistent->u8[MATCH_TEMPLATE_BUFFER],
            best_keys);
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
    }

    // Reduce the per-column keys to one global best with the same packed ordering, then unpack it
    // on the device so the host receives coordinates and the quantized score.
    error = cudaMemsetAsync(best_key, 0, sizeof(unsigned long long), persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    match_publish_kernel<<<
        dim3((output_width + MATCH_REDUCE_BLOCK - 1) / MATCH_REDUCE_BLOCK, 1),
        dim3(MATCH_REDUCE_BLOCK, 1), 0, persistent->stream>>>(
        best_keys, output_width, best_key);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    (void)candidate_scores;
    (void)candidate_ys;

    match_unpack_kernel<<<1, 1, 0, persistent->stream>>>(
        best_key, 1.0f, match_xy_device, match_score_device);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    int match_xy[2] = {0, 0};
    error = cudaMemcpyAsync(
        match_xy, match_xy_device, sizeof(match_xy), cudaMemcpyDeviceToHost, persistent->stream);
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            out_score, match_score_device, sizeof(float), cudaMemcpyDeviceToHost, persistent->stream);
    }
    if (error != cudaSuccess) return cuda_result(error);
    result = visionflow_cuda::stream_result(persistent->stream);
    if (result == VF_CUDA_OK) finalize_timing(persistent);
    if (result != VF_CUDA_OK) return result;
    if (match_xy[0] < 0 || match_xy[1] < 0) return VF_CUDA_INTERNAL_ERROR;

    out_match[0] = search_x + match_xy[0];
    out_match[1] = search_y + match_xy[1];
    out_match[2] = template_width;
    out_match[3] = template_height;
    return VF_CUDA_OK;
}

// Smallest power of two >= value. The Stockham kernels in this file are radix-2 and radix-4, so
// every transform length is a power of two and no external FFT library is needed.
static int pattern_fft_length(int value) {
    int length = 1;
    while (length < value) length <<= 1;
    return length;
}

// Runs one batched power-of-two transform over contiguous rows. `data` and `scratch` are
// ping-ponged, so the caller receives whichever buffer ended up holding the result.
static int fft_rows(
    PersistentContext* persistent, float2** data, float2** scratch,
    int n, int batch, float sign) {
    if (n <= 1) return VF_CUDA_OK;
    const int threads = 256;
    int ns = 1;
    int log2n = 0;
    while ((1 << log2n) < n) ++log2n;
    if (log2n % 2 == 1) {
        // An odd power of two needs one radix-2 stage before the radix-4 stages take over.
        const long long butterflies = static_cast<long long>(n >> 1) * batch;
        fft_stage_radix2_kernel<<<
            static_cast<unsigned int>((butterflies + threads - 1) / threads), threads, 0,
            persistent->stream>>>(*data, *scratch, n, ns, batch, sign);
        const int result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
        std::swap(*data, *scratch);
        ns <<= 1;
    }
    const long long butterflies = static_cast<long long>(n >> 2) * batch;
    const unsigned int blocks = static_cast<unsigned int>((butterflies + threads - 1) / threads);
    while (ns < n) {
        fft_stage_radix4_kernel<<<blocks, threads, 0, persistent->stream>>>(
            *data, *scratch, n, ns, batch, sign);
        const int result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
        std::swap(*data, *scratch);
        ns <<= 2;
    }
    return VF_CUDA_OK;
}

// Two-dimensional transform of a rows x columns plane, leaving the result transposed. Both
// operands of the correlation are produced this way, so the pointwise product stays valid, and
// running the same routine on the product with the opposite sign returns it to normal orientation.
static int fft_2d_transposed(
    PersistentContext* persistent, float2** data, float2** scratch,
    int rows, int columns, float sign) {
    int result = fft_rows(persistent, data, scratch, columns, rows, sign);
    if (result != VF_CUDA_OK) return result;
    const dim3 transpose_grid(
        static_cast<unsigned int>((columns + FFT_TRANSPOSE_TILE - 1) / FFT_TRANSPOSE_TILE),
        static_cast<unsigned int>((rows + FFT_TRANSPOSE_TILE - 1) / FFT_TRANSPOSE_TILE));
    fft_transpose_kernel<<<
        transpose_grid, dim3(FFT_TRANSPOSE_TILE, FFT_TRANSPOSE_ROWS), 0, persistent->stream>>>(
        *data, *scratch, columns, rows);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    std::swap(*data, *scratch);
    return fft_rows(persistent, data, scratch, rows, columns, sign);
}

// Uploads the template and grays the whole resident frame into MATCH_ROI_BUFFER. The FFT path
// needs exactly these two inputs; it deliberately does not run the anchor export, whose
// shared-memory scoring kernel rejects large templates and whose int64 prefix planes it never uses.
static int prepare_pattern_frame(
    PersistentContext* persistent, const uint8_t* templ, int template_width, int template_height) {
    const int frame_width = persistent->resident_width;
    const int frame_height = persistent->resident_height;
    const size_t template_bytes = static_cast<size_t>(template_width) * template_height;
    int result = reserve_device(
        &persistent->u8[MATCH_TEMPLATE_BUFFER], &persistent->u8_capacity[MATCH_TEMPLATE_BUFFER],
        template_bytes, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    // The gray frame is fully rewritten here, so it must be exactly this size: a larger leftover
    // buffer from an earlier call would keep the old row pitch.
    result = reserve_exact(
        &persistent->u8[MATCH_ROI_BUFFER], &persistent->u8_capacity[MATCH_ROI_BUFFER],
        static_cast<size_t>(frame_width) * frame_height, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    update_context_memory_peak(persistent);

    cudaError_t error = cudaMemcpyAsync(
        persistent->u8[MATCH_TEMPLATE_BUFFER], templ, template_bytes,
        cudaMemcpyHostToDevice, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    if (persistent->resident_channels == 3) {
        bgr_gray_roi_kernel<<<
            grid2d(frame_width, frame_height), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
            persistent->resident_u8, frame_width, 0, 0,
            persistent->u8[MATCH_ROI_BUFFER], frame_width, frame_height);
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
    } else {
        error = cudaMemcpyAsync(
            persistent->u8[MATCH_ROI_BUFFER], persistent->resident_u8,
            static_cast<size_t>(frame_width) * frame_height,
            cudaMemcpyDeviceToDevice, persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
    }
    return VF_CUDA_OK;
}

// Fills persistent->pattern_scores with the TM_CCOEFF_NORMED response of the whole resident frame:
// an FFT cross correlation supplies the numerator and exact int64 summed-area tables supply the
// window statistics. The transforms are the Stockham kernels in this file, so no external FFT
// library is involved and the frame is padded to powers of two rather than 2/3/5/7-smooth sizes.
static int pattern_match_fft_response(
    PersistentContext* persistent, int template_width, int template_height,
    int output_width, int output_height, double template_mean, double template_variance) {
    const int frame_width = persistent->resident_width;
    const int frame_height = persistent->resident_height;
    const int pad_width = pattern_fft_length(frame_width);
    const int pad_height = pattern_fft_length(frame_height);
    const size_t plane_count = static_cast<size_t>(pad_width) * static_cast<size_t>(pad_height);
    const size_t frame_pixels = static_cast<size_t>(frame_width) * frame_height;
    const size_t plane_bytes = plane_count * sizeof(float2);

    int result = reserve_device(
        &persistent->pattern_fft_a, &persistent->pattern_fft_a_capacity,
        plane_count, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->pattern_fft_b, &persistent->pattern_fft_b_capacity,
        plane_count, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->pattern_fft_c, &persistent->pattern_fft_c_capacity,
        plane_count, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->pattern_sat_sum, &persistent->pattern_sat_sum_capacity,
        frame_pixels, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->pattern_sat_square, &persistent->pattern_sat_square_capacity,
        frame_pixels, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    update_context_memory_peak(persistent);

    // Exact window statistics: row prefixes, then a column sweep, both in int64.
    pattern_sat_rows_kernel<<<
        frame_height, PATTERN_SAT_SCAN_THREADS, 0, persistent->stream>>>(
        persistent->u8[MATCH_ROI_BUFFER], frame_width, frame_height,
        persistent->pattern_sat_sum, persistent->pattern_sat_square);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    pattern_sat_columns_kernel<<<
        (frame_width + PATTERN_SAT_SCAN_THREADS - 1) / PATTERN_SAT_SCAN_THREADS,
        PATTERN_SAT_SCAN_THREADS, 0, persistent->stream>>>(
        persistent->pattern_sat_sum, persistent->pattern_sat_square, frame_width, frame_height);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    // Template spectrum first, so the image transform can keep the two remaining planes.
    float2* template_data = persistent->pattern_fft_c;
    float2* scratch = persistent->pattern_fft_b;
    cudaError_t error = cudaMemsetAsync(template_data, 0, plane_bytes, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    pattern_pad_u8_kernel<<<
        grid2d(template_width, template_height), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        persistent->u8[MATCH_TEMPLATE_BUFFER], template_width, template_height, template_width,
        template_data, pad_width);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    result = fft_2d_transposed(persistent, &template_data, &scratch, pad_height, pad_width, -1.0f);
    if (result != VF_CUDA_OK) return result;

    // The transform ping-pongs, so which plane holds the template spectrum depends on the stage
    // count. Take the two planes it did not land in; anything else would overwrite the template.
    float2* image_data = nullptr;
    float2* image_scratch = nullptr;
    for (float2* candidate : {
             persistent->pattern_fft_a, persistent->pattern_fft_b, persistent->pattern_fft_c}) {
        if (candidate == template_data) continue;
        if (image_data == nullptr) {
            image_data = candidate;
        } else {
            image_scratch = candidate;
        }
    }
    if (image_data == nullptr || image_scratch == nullptr) return VF_CUDA_INTERNAL_ERROR;
    error = cudaMemsetAsync(image_data, 0, plane_bytes, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    pattern_pad_u8_kernel<<<
        grid2d(frame_width, frame_height), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        persistent->u8[MATCH_ROI_BUFFER], frame_width, frame_height, frame_width,
        image_data, pad_width);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    result = fft_2d_transposed(persistent, &image_data, &image_scratch, pad_height, pad_width, -1.0f);
    if (result != VF_CUDA_OK) return result;

    // image = image * conj(template) / N, then the inverse transform returns the correlation plane
    // to normal orientation because both operands were left transposed.
    const int correlate_threads = 256;
    pattern_spectrum_correlate_kernel<<<
        static_cast<unsigned int>((plane_count + correlate_threads - 1) / correlate_threads),
        correlate_threads, 0, persistent->stream>>>(
        image_data, template_data, plane_count, 1.0f / static_cast<float>(plane_count));
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    result = fft_2d_transposed(persistent, &image_data, &image_scratch, pad_width, pad_height, 1.0f);
    if (result != VF_CUDA_OK) return result;

    pattern_fft_score_kernel<<<
        grid2d(output_width, output_height), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        image_data, pad_width,
        persistent->pattern_sat_sum, persistent->pattern_sat_square, frame_width,
        output_width, output_height, template_width, template_height,
        template_width * template_height, template_mean, template_variance,
        persistent->pattern_scores);
    return visionflow_cuda::kernel_launch_result();
}

VF_CUDA_API int vf_pattern_match_gray_u8(
    void* context,
    uint64_t generation,
    const uint8_t* templ, int template_width, int template_height,
    float match_threshold, int max_candidates, float nms_threshold,
    int max_count, int sort_row_tolerance,
    int32_t* out_xy, float* out_scores, int output_capacity, int* out_count) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || templ == nullptr || out_xy == nullptr || out_scores == nullptr ||
        out_count == nullptr || output_capacity <= 0 || !std::isfinite(match_threshold) ||
        !std::isfinite(nms_threshold) || nms_threshold < 0.0f || nms_threshold > 1.0f ||
        generation == 0 || generation != persistent->resident_generation ||
        persistent->resident_u8 == nullptr || template_width <= 0 || template_height <= 0 ||
        template_width > persistent->resident_width ||
        template_height > persistent->resident_height) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int output_width = persistent->resident_width - template_width + 1;
    const int output_height = persistent->resident_height - template_height + 1;
    if (output_width <= 0 || output_height <= 0 || output_width > 0xffff || output_height > 0xffff) {
        return VF_CUDA_UNSUPPORTED;
    }
    const long long element_count_ll =
        static_cast<long long>(output_width) * static_cast<long long>(output_height);
    if (element_count_ll <= 0 || element_count_ll > INT_MAX) return VF_CUDA_UNSUPPORTED;
    const int element_count = static_cast<int>(element_count_ll);

    // Template statistics on the host: the template is constant per Recipe, and both response
    // paths need the mean and variance before any device work is scheduled.
    double template_sum_host = 0.0;
    double template_square_sum_host = 0.0;
    const int template_pixel_count = template_width * template_height;
    for (int index = 0; index < template_pixel_count; ++index) {
        const int value = templ[index];
        template_sum_host += value;
        template_square_sum_host += static_cast<double>(value) * value;
    }
    const double template_mean_host = template_sum_host / template_pixel_count;
    const double template_variance_host =
        template_square_sum_host / template_pixel_count - template_mean_host * template_mean_host;
    if (!(template_variance_host > 1e-12)) return VF_CUDA_UNSUPPORTED;

    // pattern_score_map_kernel costs output_elements x template_pixels while the FFT path is
    // dominated by the frame-sized transforms, so a large template has to take the FFT route.
    // Measured on RTX 3090 with a 16384x13000 frame: see gpu/validate_pattern_match_fft.py.
    const double brute_force_work =
        static_cast<double>(element_count) * static_cast<double>(template_pixel_count);
    const double frame_pixels = static_cast<double>(persistent->resident_width) *
                                static_cast<double>(persistent->resident_height);
    const bool prefer_fft =
        brute_force_work > PATTERN_FFT_CROSSOVER_WORK_PER_PIXEL * frame_pixels;

    bool use_fft = prefer_fft;
    int result = VF_CUDA_OK;
    if (!prefer_fft) {
        // Reuse the verified anchor call to validate the template, upload it, gray the resident
        // image and establish the same scratch/lifetime rules. Its single-best result is ignored.
        int ignored_match[4] = {0, 0, 0, 0};
        float ignored_score = 0.0f;
        result = vf_match_template_gray_u8(
            context, generation, 0, 0,
            persistent->resident_width, persistent->resident_height,
            templ, template_width, template_height, ignored_match, &ignored_score);
        // A template whose shared-memory halo does not fit is not a failure here: the FFT path
        // has no such limit, so fall through to it.
        if (result == VF_CUDA_UNSUPPORTED) {
            use_fft = true;
            result = prepare_pattern_frame(persistent, templ, template_width, template_height);
        } else if (result != VF_CUDA_OK) {
            return result;
        }
    } else {
        result = prepare_pattern_frame(persistent, templ, template_width, template_height);
    }
    if (result != VF_CUDA_OK) return result;

    result = reserve_device(
        &persistent->pattern_scores, &persistent->pattern_score_capacity,
        static_cast<size_t>(element_count), &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->pattern_keys, &persistent->pattern_key_capacity,
        static_cast<size_t>(element_count), &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->pattern_sorted_keys, &persistent->pattern_sorted_key_capacity,
        static_cast<size_t>(element_count), &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->pattern_selected_keys, &persistent->pattern_selected_key_capacity,
        static_cast<size_t>(output_capacity), &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->pattern_out_xy, &persistent->pattern_out_xy_capacity,
        static_cast<size_t>(output_capacity) * 2, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->pattern_out_scores, &persistent->pattern_out_score_capacity,
        static_cast<size_t>(output_capacity), &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->pattern_out_count, &persistent->pattern_out_count_capacity,
        static_cast<size_t>(1), &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;

    size_t sort_scratch_bytes = 0;
    cudaError_t error = cub::DeviceRadixSort::SortKeys(
        nullptr, sort_scratch_bytes,
        persistent->pattern_keys, persistent->pattern_sorted_keys,
        element_count, 0, 64, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    result = reserve_device(
        &persistent->pattern_sort_scratch, &persistent->pattern_sort_scratch_capacity,
        sort_scratch_bytes > 0 ? sort_scratch_bytes : 1, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    update_context_memory_peak(persistent);

    const int template_pixels = template_pixel_count;
    const double template_mean = template_mean_host;
    const double template_variance = template_variance_host;

    reset_timing(persistent, false);
    if (use_fft) {
        result = pattern_match_fft_response(
            persistent, template_width, template_height, output_width, output_height,
            template_mean, template_variance);
    } else {
        pattern_score_map_kernel<<<
            grid2d(output_width, output_height), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
            persistent->u8[MATCH_ROI_BUFFER], persistent->resident_width,
            output_width, output_height, template_width, template_height, template_pixels,
            template_mean, template_variance,
            persistent->u8[MATCH_TEMPLATE_BUFFER], persistent->pattern_scores);
        result = visionflow_cuda::kernel_launch_result();
    }
    if (result != VF_CUDA_OK) return result;
    const int peak_width = min(template_width, output_width);
    const int peak_height = min(template_height, output_height);
    pattern_local_peak_keys_kernel<<<
        grid2d(output_width, output_height), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        persistent->pattern_scores, output_width, output_height,
        peak_width, peak_height, match_threshold, persistent->pattern_keys);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    error = cub::DeviceRadixSort::SortKeys(
        persistent->pattern_sort_scratch, sort_scratch_bytes,
        persistent->pattern_keys, persistent->pattern_sorted_keys,
        element_count, 0, 64, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    pattern_select_nms_kernel<<<1, 1, 0, persistent->stream>>>(
        persistent->pattern_sorted_keys, element_count_ll,
        template_width, template_height, max_candidates, nms_threshold,
        max_count, sort_row_tolerance,
        persistent->pattern_selected_keys, output_capacity,
        persistent->pattern_out_xy, persistent->pattern_out_scores,
        persistent->pattern_out_count);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    int count = 0;
    error = cudaMemcpyAsync(
        &count, persistent->pattern_out_count, sizeof(int),
        cudaMemcpyDeviceToHost, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    result = visionflow_cuda::stream_result(persistent->stream);
    if (result != VF_CUDA_OK) return result;
    if (count < 0 || count > output_capacity) return VF_CUDA_INTERNAL_ERROR;
    if (count > 0) {
        error = cudaMemcpyAsync(
            out_xy, persistent->pattern_out_xy,
            sizeof(int32_t) * static_cast<size_t>(count) * 2,
            cudaMemcpyDeviceToHost, persistent->stream);
        if (error == cudaSuccess) {
            error = cudaMemcpyAsync(
                out_scores, persistent->pattern_out_scores,
                sizeof(float) * static_cast<size_t>(count),
                cudaMemcpyDeviceToHost, persistent->stream);
        }
        if (error != cudaSuccess) return cuda_result(error);
        result = visionflow_cuda::stream_result(persistent->stream);
        if (result != VF_CUDA_OK) return result;
    }
    finalize_timing(persistent);
    *out_count = count;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_pattern_match_fft_available() {
    // The FFT response is built into this library, so it is always available here. The export
    // stays so a caller can still tell a DLL that has the large-template path from one that
    // predates it, which is what old-DLL routing needs.
    return 1;
}

VF_CUDA_API int vf_match_template_debug_key(void* context, unsigned long long* out_key) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || out_key == nullptr || persistent->match_candidates == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int key_offset =
        match_result_offset(static_cast<int>(persistent->match_candidate_output_width)) + 1;
    const unsigned long long* best_key = reinterpret_cast<const unsigned long long*>(
        persistent->match_candidates + key_offset);
    cudaError_t error = cudaMemcpyAsync(
        out_key, best_key, sizeof(unsigned long long), cudaMemcpyDeviceToHost, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    return visionflow_cuda::stream_result(persistent->stream);
}

// Diagnostics: copy one prefix plane back for comparison against the CPU reference.
VF_CUDA_API int vf_match_template_debug_planes(
    void* context, int plane, int64_t* out_values, size_t count) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || out_values == nullptr || count == 0 ||
        plane < 0 || plane >= MATCH_PLANE_COUNT || persistent->match_plane[plane] == nullptr ||
        count > persistent->match_plane_capacity[plane]) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    cudaError_t error = cudaMemcpyAsync(
        out_values, persistent->match_plane[plane], count * sizeof(int64_t),
        cudaMemcpyDeviceToHost, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    return visionflow_cuda::stream_result(persistent->stream);
}

// Diagnostics: copy the gray search ROI back so a caller can compare it with the CPU gray image.
VF_CUDA_API int vf_match_template_debug_roi(void* context, uint8_t* out_values, size_t count) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || out_values == nullptr || count == 0 ||
        persistent->u8[MATCH_ROI_BUFFER] == nullptr ||
        count > persistent->u8_capacity[MATCH_ROI_BUFFER]) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    cudaError_t error = cudaMemcpyAsync(
        out_values, persistent->u8[MATCH_ROI_BUFFER], count, cudaMemcpyDeviceToHost,
        persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    return visionflow_cuda::stream_result(persistent->stream);
}

// Diagnostics: copy the per-column candidate scores and rows of the last localization call, so a
// caller can compare each column's best against the CPU match map. candidate_slots reports how
// many slots the context currently holds.
VF_CUDA_API int vf_match_template_debug_candidates(
    void* context, double* out_scores, int* out_rows, int count, int* candidate_slots) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (candidate_slots != nullptr) {
        *candidate_slots = static_cast<int>(persistent != nullptr
            ? persistent->match_candidate_capacity : 0);
    }
    if (persistent == nullptr || out_scores == nullptr || out_rows == nullptr || count <= 0 ||
        persistent->match_candidates == nullptr || count > MATCH_MAX_OUTPUT_WIDTH ||
        count > persistent->match_candidate_output_width ||
        count * MATCH_CANDIDATE_SLOT_STRIDE > (int)persistent->match_candidate_capacity) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const double* scores = reinterpret_cast<const double*>(persistent->match_candidates);
    const int* rows = reinterpret_cast<const int*>(
        persistent->match_candidates + match_candidate_slots(count));
    cudaError_t error = cudaMemcpyAsync(
        out_scores, scores, sizeof(double) * count, cudaMemcpyDeviceToHost, persistent->stream);
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            out_rows, rows, sizeof(int) * count, cudaMemcpyDeviceToHost, persistent->stream);
    }
    if (error != cudaSuccess) return cuda_result(error);
    return visionflow_cuda::stream_result(persistent->stream);
}

// Contour trace of the requested region of the resident binary mask. Scratch buffers are grow-only;
// when the first guess at the output size is too small the kernel reports the exact requirement and
// the whole trace is re-run with that capacity, so a result is never truncated silently.
VF_CUDA_API int vf_find_contours_u8(
    void* context,
    uint64_t generation,
    int x, int y, int width, int height, int mode,
    int* out_contour_count, int* out_point_count) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || out_contour_count == nullptr || out_point_count == nullptr ||
        generation == 0 || generation != persistent->resident_generation ||
        persistent->resident_u8 == nullptr ||
        (mode != VF_CONTOURS_RETR_EXTERNAL && mode != VF_CONTOURS_RETR_LIST)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    // A colour resident image cannot be reinterpreted as a binary mask without changing the
    // foreground rule, so the caller restarts this step on the CPU reference instead.
    if (persistent->resident_channels != 1) return VF_CUDA_UNSUPPORTED;
    if (x < 0 || y < 0 || width <= 0 || height <= 0 ||
        x > persistent->resident_width - width ||
        y > persistent->resident_height - height) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int label_stride = width + 2;
    const size_t padded = static_cast<size_t>(label_stride) * static_cast<size_t>(height + 2);
    if (padded > static_cast<size_t>(INT_MAX)) return VF_CUDA_INVALID_ARGUMENT;

    const size_t resident_pitch =
        static_cast<size_t>(persistent->resident_width) * persistent->resident_channels;
    // The init kernel applies (x, y) itself: passing an already-offset pointer here would apply
    // the region origin twice and silently trace the wrong window.
    const uint8_t* mask = persistent->resident_u8;

    // First guess at the output size. Sparse production masks are far below these ratios; a denser
    // mask only costs one extra trace with the exact reported capacity.
    int contour_hint = static_cast<int>(std::min<long long>(
        std::max<long long>(static_cast<long long>(padded) / 64, 64), 1LL << 20));
    int point_hint = static_cast<int>(std::min<long long>(
        std::max<long long>(static_cast<long long>(padded) / 8, 256), 1LL << 22));

    int counts[3] = {0, 0, 0};
    int result = reserve_device(
        &persistent->contour_label, &persistent->contour_label_capacity, padded,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    // The scan reports the contour count, the point count and the overflow flag in one block.
    result = reserve_device(
        &persistent->contour_counts, &persistent->contour_count_capacity, static_cast<size_t>(4),
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;

    reset_timing(persistent, false);
    cudaError_t error = cudaSuccess;
    // RETR_LIST walks the zero-ness transition list instead of every row byte. The list depends
    // only on the mask, so it is built once, before the retry loop, and its size is exact (the
    // per-row counts come back to the host and are prefix-summed there).
    const bool list_mode = (mode == VF_CONTOURS_RETR_LIST);
    if (list_mode) {
        std::vector<int> row_counts;
        std::vector<int> row_start;
        try {
            row_counts.assign(static_cast<size_t>(height), 0);
            row_start.assign(static_cast<size_t>(height) + 2, 0);
        } catch (const std::bad_alloc&) {
            return VF_CUDA_ALLOCATION_FAILED;
        }
        result = reserve_device(
            &persistent->contour_row_counts, &persistent->contour_row_count_capacity,
            static_cast<size_t>(height), &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
        contour_row_transition_counts_kernel<<<
            dim3(static_cast<unsigned int>((height + 255) / 256), 1, 1), dim3(256, 1, 1), 0,
            persistent->stream>>>(
            mask, static_cast<int>(resident_pitch), x, y, width, height,
            persistent->contour_row_counts);
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
        error = cudaMemcpyAsync(
            row_counts.data(), persistent->contour_row_counts,
            sizeof(int) * static_cast<size_t>(height), cudaMemcpyDeviceToHost, persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        result = visionflow_cuda::stream_result(persistent->stream);
        if (result != VF_CUDA_OK) return result;

        long long total = 0;
        for (int row = 1; row <= height; ++row) {
            row_start[row] = static_cast<int>(total);
            total += row_counts[static_cast<size_t>(row - 1)];
        }
        if (total > INT_MAX) return VF_CUDA_INVALID_ARGUMENT;
        row_start[height + 1] = static_cast<int>(total);

        result = reserve_device(
            &persistent->contour_transitions, &persistent->contour_transition_capacity,
            static_cast<size_t>(total > 0 ? total : 1), &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
        result = reserve_device(
            &persistent->contour_row_start, &persistent->contour_row_start_capacity,
            static_cast<size_t>(height) + 2, &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
        error = cudaMemcpyAsync(
            persistent->contour_row_start, row_start.data(),
            sizeof(int) * (static_cast<size_t>(height) + 2), cudaMemcpyHostToDevice,
            persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        if (total > 0) {
            contour_fill_transitions_kernel<<<
                dim3(static_cast<unsigned int>((height + 255) / 256), 1, 1), dim3(256, 1, 1), 0,
                persistent->stream>>>(
                mask, static_cast<int>(resident_pitch), x, y, width, height,
                persistent->contour_row_start, label_stride, persistent->contour_transitions);
            result = visionflow_cuda::kernel_launch_result();
            if (result != VF_CUDA_OK) return result;
        }
    }

    bool complete = false;
    for (int attempt = 0; attempt < 4 && !complete; ++attempt) {
        result = reserve_device(
            &persistent->contour_offsets, &persistent->contour_offset_capacity,
            static_cast<size_t>(contour_hint) + 1, &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
        result = reserve_device(
            &persistent->contour_points, &persistent->contour_point_capacity,
            static_cast<size_t>(point_hint) * 2, &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
        result = reserve_device(
            &persistent->contour_out_offsets, &persistent->contour_out_offset_capacity,
            static_cast<size_t>(contour_hint) + 1, &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
        result = reserve_device(
            &persistent->contour_out_points, &persistent->contour_out_point_capacity,
            static_cast<size_t>(point_hint) * 2, &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;

        contour_init_label_kernel<<<
            grid2d(label_stride, height + 2), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
            mask, static_cast<int>(resident_pitch), x, y, width, height,
            persistent->contour_label, label_stride);
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
        record_timing_event(persistent, TIMING_AFTER_INPUT);

        if (list_mode) {
            contour_scan_list_kernel<<<1, 32, 0, persistent->stream>>>(
                persistent->contour_label, label_stride, height,
                persistent->contour_transitions, persistent->contour_row_start,
                persistent->contour_offsets, static_cast<int>(persistent->contour_offset_capacity),
                persistent->contour_points, static_cast<int>(persistent->contour_point_capacity / 2),
                persistent->contour_counts);
        } else {
            contour_scan_kernel<<<1, 1, 0, persistent->stream>>>(
                persistent->contour_label, label_stride, width, height, mode,
                persistent->contour_offsets, static_cast<int>(persistent->contour_offset_capacity),
                persistent->contour_points, static_cast<int>(persistent->contour_point_capacity / 2),
                persistent->contour_counts);
        }
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
        record_timing_event(persistent, TIMING_AFTER_KERNEL);

        error = cudaMemcpyAsync(
            counts, persistent->contour_counts, sizeof(counts), cudaMemcpyDeviceToHost,
            persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        record_timing_event(persistent, TIMING_AFTER_OUTPUT);
        result = visionflow_cuda::stream_result(persistent->stream);
        if (result != VF_CUDA_OK) return result;

        if (counts[2] != 0) {
            // Grow to the exact capacity the trace reported and run it again; the second run has
            // enough room by construction, and identical input always yields identical counts.
            contour_hint = counts[0];
            point_hint = counts[1];
            continue;
        }

        if (counts[0] > 0) {
            contour_reverse_kernel<<<
                dim3(static_cast<unsigned int>((counts[0] + 127) / 128), 1, 1), dim3(128, 1, 1),
                0, persistent->stream>>>(
                persistent->contour_offsets, persistent->contour_points, counts[0], counts[1],
                persistent->contour_out_offsets, persistent->contour_out_points);
            result = visionflow_cuda::kernel_launch_result();
            if (result != VF_CUDA_OK) return result;
        }
        complete = true;
    }
    if (!complete) return VF_CUDA_INTERNAL_ERROR;
    result = visionflow_cuda::stream_result(persistent->stream);
    if (result != VF_CUDA_OK) return result;
    finalize_timing(persistent);

    persistent->contour_count = counts[0];
    persistent->contour_point_count = counts[1];
    persistent->contour_generation = generation;
    persistent->contour_result_valid = true;
    *out_contour_count = counts[0];
    *out_point_count = counts[1];
    return VF_CUDA_OK;
}

// Copies the most recent contour result to the caller. A short buffer is an error, never a partial
// result, and a stale result (the resident image changed after the trace) is rejected.
VF_CUDA_API int vf_find_contours_download(
    void* context,
    int32_t* out_offsets, int offset_capacity,
    int32_t* out_points, int point_capacity) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || out_offsets == nullptr || !persistent->contour_result_valid ||
        persistent->contour_generation != persistent->resident_generation) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int contour_count = persistent->contour_count;
    const int point_count = persistent->contour_point_count;
    if (offset_capacity < contour_count + 1) return VF_CUDA_INVALID_ARGUMENT;
    if (point_count > 0 && (out_points == nullptr || point_capacity < point_count)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (contour_count == 0) {
        out_offsets[0] = 0;
        return VF_CUDA_OK;
    }
    cudaError_t error = cudaMemcpyAsync(
        out_offsets, persistent->contour_out_offsets,
        sizeof(int32_t) * static_cast<size_t>(contour_count + 1), cudaMemcpyDeviceToHost,
        persistent->stream);
    if (error == cudaSuccess && point_count > 0) {
        error = cudaMemcpyAsync(
            out_points, persistent->contour_out_points,
            sizeof(int32_t) * 2 * static_cast<size_t>(point_count), cudaMemcpyDeviceToHost,
            persistent->stream);
    }
    if (error != cudaSuccess) return cuda_result(error);
    return visionflow_cuda::stream_result(persistent->stream);
}

// Monotone float32 -> uint32 order key: positives keep their magnitude and gain the top bit,
// negatives invert every bit. Unsigned integer order then equals float order for every bit pattern,
// including -0.0 < +0.0, subnormals and infinities, so a plain key sort orders floats exactly.
__global__ void median_order_key_kernel(
    const float* values,
    uint32_t* keys,
    int* nan_flag,
    int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint32_t bits = __float_as_uint(values[index]);
    keys[index] = (bits & 0x80000000u) != 0u
        ? (bits ^ 0xFFFFFFFFu)
        : (bits | 0x80000000u);
    // NumPy's median returns NaN whenever the input holds one (_median_nancheck inspects the last
    // partitioned element, and every NaN sorts last), so NaN presence is reported explicitly rather
    // than being folded into an invented key order.
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
        atomicOr(nan_flag, 1);
    }
}

// Inverse of median_order_key_kernel() for the one or two middle keys, evaluated on the host.
float median_key_to_value(uint32_t key) {
    const uint32_t bits = (key & 0x80000000u) != 0u
        ? (key & 0x7FFFFFFFu)
        : (key ^ 0xFFFFFFFFu);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Reserves the exact-median scratch for `items` values and sizes the CUB radix-sort temporary
// storage with the same offset type the sorting calls below use. Shared by vf_median_f32 and
// vf_cnr_mask_f32 so both exports sort through one code path and one set of grow-only buffers.
int reserve_median_scratch(PersistentContext* persistent, int items) {
    const size_t item_count = static_cast<size_t>(items);
    int result = reserve_device(
        &persistent->median_keys, &persistent->median_key_capacity, item_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->median_sorted_keys, &persistent->median_sorted_key_capacity, item_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->median_nan_flag, &persistent->median_nan_flag_capacity,
        static_cast<size_t>(1), &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;

    size_t sort_storage_bytes = 0;
    cudaError_t error = cub::DeviceRadixSort::SortKeys(
        nullptr, sort_storage_bytes, persistent->median_keys, persistent->median_sorted_keys,
        items, 0, static_cast<int>(sizeof(uint32_t) * 8), persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    if (sort_storage_bytes == 0) return VF_CUDA_INTERNAL_ERROR;
    return reserve_device(
        &persistent->median_sort_scratch, &persistent->median_sort_scratch_capacity,
        sort_storage_bytes, &persistent->allocation_count);
}

// Bit-exact np.median of `items` float32 values that are already resident on the device, so the
// operand never crosses PCIe. The full contract is documented in include/visionflow_cuda.h:
// monotone float32 -> uint32 order keys, cub::DeviceRadixSort, middle-key-only readback, and the
// float32 even-count average on the host. NaN presence is reported through the one-word flag and
// decoded into a quiet NaN, mirroring NumPy's _median_nancheck.
//
// reserve_median_scratch() must have run for the same `items` first. `record_timing` preserves the
// event placement vf_median_f32 has always reported (AFTER_KERNEL after the sort, AFTER_OUTPUT
// after the middle-key readback); a caller that owns the whole timing window passes false.
int run_device_median(
    PersistentContext* persistent,
    const float* device_values,
    int items,
    float* out_median,
    bool record_timing) {
    size_t sort_storage_bytes = 0;
    cudaError_t error = cub::DeviceRadixSort::SortKeys(
        nullptr, sort_storage_bytes, persistent->median_keys, persistent->median_sorted_keys,
        items, 0, static_cast<int>(sizeof(uint32_t) * 8), persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    if (sort_storage_bytes == 0 || persistent->median_sort_scratch == nullptr ||
        persistent->median_sort_scratch_capacity < sort_storage_bytes) {
        return VF_CUDA_INTERNAL_ERROR;
    }

    error = cudaMemsetAsync(persistent->median_nan_flag, 0, sizeof(int), persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    constexpr int MEDIAN_THREADS = 256;
    median_order_key_kernel<<<
        (items + MEDIAN_THREADS - 1) / MEDIAN_THREADS, MEDIAN_THREADS, 0, persistent->stream>>>(
        device_values, persistent->median_keys, persistent->median_nan_flag, items);
    int result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    error = cub::DeviceRadixSort::SortKeys(
        persistent->median_sort_scratch, sort_storage_bytes, persistent->median_keys,
        persistent->median_sorted_keys, items, 0, static_cast<int>(sizeof(uint32_t) * 8),
        persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    if (record_timing) {
        record_timing_event(persistent, TIMING_AFTER_KERNEL);
    }

    // Only the middle one or two keys cross PCIe, plus the four-byte NaN-presence word.
    const bool even = (items % 2) == 0;
    const int middle = items / 2;
    const int first_key = even ? (middle - 1) : middle;
    const int key_reads = even ? 2 : 1;
    uint32_t host_keys[2] = {0u, 0u};
    int host_nan = 0;
    error = cudaMemcpyAsync(
        host_keys, persistent->median_sorted_keys + first_key,
        sizeof(uint32_t) * static_cast<size_t>(key_reads), cudaMemcpyDeviceToHost,
        persistent->stream);
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            &host_nan, persistent->median_nan_flag, sizeof(int), cudaMemcpyDeviceToHost,
            persistent->stream);
    }
    if (error != cudaSuccess) return cuda_result(error);
    if (record_timing) {
        record_timing_event(persistent, TIMING_AFTER_OUTPUT);
    }
    result = visionflow_cuda::stream_result(persistent->stream);
    if (result != VF_CUDA_OK) return result;

    if (host_nan != 0) {
        const uint32_t quiet_nan_bits = 0x7FC00000u;
        std::memcpy(out_median, &quiet_nan_bits, sizeof(*out_median));
        return VF_CUDA_OK;
    }
    const float low = median_key_to_value(host_keys[0]);
    if (!even) {
        *out_median = low;
        return VF_CUDA_OK;
    }
    const float high = median_key_to_value(host_keys[1]);
    // np.median averages the two middle values in the input dtype: a float32 add, then a float32
    // divide by two. Reproduce both operations exactly, including their overflow and rounding.
    *out_median = (low + high) / 2.0f;
    return VF_CUDA_OK;
}

// Bit-exact np.median for a host float32 array. The operand is uploaded once into the context's
// grow-only median buffer and the rest of the work is run_device_median(), so this export and
// vf_cnr_mask_f32 share the key/sort machinery instead of duplicating it.
VF_CUDA_API int vf_median_f32(
    void* context,
    const float* values,
    long long count,
    float* out_median) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || values == nullptr || out_median == nullptr || count <= 0 ||
        count > static_cast<long long>(INT_MAX)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int items = static_cast<int>(count);
    const size_t item_count = static_cast<size_t>(items);

    int result = reserve_device(
        &persistent->median_values, &persistent->median_value_capacity, item_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_median_scratch(persistent, items);
    if (result != VF_CUDA_OK) return result;

    reset_timing(persistent, true);
    cudaError_t error = cudaMemcpyAsync(
        persistent->median_values, values, sizeof(float) * item_count, cudaMemcpyHostToDevice,
        persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    record_timing_event(persistent, TIMING_AFTER_INPUT);

    result = run_device_median(persistent, persistent->median_values, items, out_median, true);
    if (result != VF_CUDA_OK) return result;
    finalize_timing(persistent);
    return VF_CUDA_OK;
}

// Shared body of vf_gaussian_blur_f32 and vf_gaussian_blur_f32_roi. The requested rectangle of the
// host float32 source is uploaded with a 2D copy (only the rectangle - the ROI export never
// touches pixels outside it, which is what makes it equal to cv2.GaussianBlur on the same
// sub-array), blurred on the device with the context's grow-only scratch, and copied back.
static int gaussian_blur_f32_device(
    PersistentContext* persistent,
    const float* src, int src_stride, int offset_x, int offset_y,
    int width, int height, float* dst, int dst_stride, int kernel_size, double sigma) {
    const size_t row_values = static_cast<size_t>(width);
    const size_t pixel_count = row_values * static_cast<size_t>(height);
    if (pixel_count == 0 || pixel_count > SIZE_MAX / sizeof(float)) return VF_CUDA_INVALID_ARGUMENT;
    const size_t row_bytes = row_values * sizeof(float);
    int result = reserve_device(
        &persistent->gaussian_f32_input, &persistent->gaussian_f32_input_capacity, pixel_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->gaussian_f32_intermediate, &persistent->gaussian_f32_intermediate_capacity,
        pixel_count, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->gaussian_f32_output, &persistent->gaussian_f32_output_capacity, pixel_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;

    reset_timing(persistent, true);
    cudaError_t error = cudaMemcpy2DAsync(
        persistent->gaussian_f32_input, row_bytes,
        reinterpret_cast<const char*>(src) + static_cast<size_t>(offset_y) * src_stride +
            static_cast<size_t>(offset_x) * sizeof(float),
        static_cast<size_t>(src_stride), row_bytes, static_cast<size_t>(height),
        cudaMemcpyHostToDevice, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    record_timing_event(persistent, TIMING_AFTER_INPUT);

    int radius = 0;
    result = prepare_gaussian_f32_weights(kernel_size, sigma, &radius, persistent->stream);
    if (result != VF_CUDA_OK) return result;
    launch_gaussian_f32(
        persistent->gaussian_f32_input, persistent->gaussian_f32_intermediate,
        persistent->gaussian_f32_output, width, height, radius, persistent->stream);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    record_timing_event(persistent, TIMING_AFTER_KERNEL);

    error = cudaMemcpy2DAsync(
        dst, static_cast<size_t>(dst_stride), persistent->gaussian_f32_output, row_bytes,
        row_bytes, static_cast<size_t>(height), cudaMemcpyDeviceToHost, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    record_timing_event(persistent, TIMING_AFTER_OUTPUT);
    result = visionflow_cuda::stream_result(persistent->stream);
    if (result != VF_CUDA_OK) return result;
    finalize_timing(persistent);
    return VF_CUDA_OK;
}

// Kernel sizes below 3 are a malformed request; odd sizes outside the verified range are reported
// as unsupported rather than computed unvalidated. sigma is validated by the coefficient rule
// (NaN and infinities have no OpenCV equivalent and are refused).
static int gaussian_f32_kernel_request(int kernel_size, double sigma) {
    if (kernel_size < GAUSSIAN_F32_MIN_KERNEL) return VF_CUDA_INVALID_ARGUMENT;
    if (std::isnan(sigma) || std::isinf(sigma)) return VF_CUDA_INVALID_ARGUMENT;
    return gaussian_f32_kernel_supported(kernel_size) ? VF_CUDA_OK : VF_CUDA_UNSUPPORTED;
}

// Separable float32 Gaussian with reflect101 borders, reproducing
// cv2.GaussianBlur(single_channel_float32, (ksize, ksize), sigma) within the tolerance documented
// in include/visionflow_cuda.h. Strides are byte counts, as everywhere else in this ABI.
// `sigma` follows cv2 exactly: a positive value is used as the standard deviation, and zero or a
// negative value selects OpenCV's automatic rule (including its fixed small-kernel table for
// ksize <= 9). It is a double so the caller's sigma is never narrowed on the way in.
VF_CUDA_API int vf_gaussian_blur_f32(
    void* context,
    const float* src, int width, int height, int src_stride,
    float* dst, int dst_stride,
    int kernel_size, double sigma) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || src == nullptr || dst == nullptr || width <= 0 || height <= 0 ||
        width > INT_MAX / static_cast<int>(sizeof(float))) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int minimum_stride = width * static_cast<int>(sizeof(float));
    if (src_stride < minimum_stride || dst_stride < minimum_stride) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int kernel_status = gaussian_f32_kernel_request(kernel_size, sigma);
    if (kernel_status != VF_CUDA_OK) return kernel_status;
    return gaussian_blur_f32_device(
        persistent, src, src_stride, 0, 0, width, height, dst, dst_stride, kernel_size, sigma);
}

// Same operator restricted to one rectangle of a wider host float32 source. The rectangle is
// treated as an isolated image - borders reflect inside it, pixels outside it are never read - so
// the result equals cv2.GaussianBlur(src[y:y+height, x:x+width], (ksize, ksize), sigma) and only
// the rectangle crosses PCIe.
VF_CUDA_API int vf_gaussian_blur_f32_roi(
    void* context,
    const float* src, int src_width, int src_height, int src_stride,
    int x, int y, int width, int height,
    float* dst, int dst_stride,
    int kernel_size, double sigma) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || src == nullptr || dst == nullptr ||
        src_width <= 0 || src_height <= 0 || width <= 0 || height <= 0 ||
        x < 0 || y < 0 || width > src_width || height > src_height ||
        x > src_width - width || y > src_height - height ||
        src_width > INT_MAX / static_cast<int>(sizeof(float)) ||
        width > INT_MAX / static_cast<int>(sizeof(float))) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (src_stride < src_width * static_cast<int>(sizeof(float)) ||
        dst_stride < width * static_cast<int>(sizeof(float))) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int kernel_status = gaussian_f32_kernel_request(kernel_size, sigma);
    if (kernel_status != VF_CUDA_OK) return kernel_status;
    return gaussian_blur_f32_device(
        persistent, src, src_stride, x, y, width, height, dst, dst_stride, kernel_size, sigma);
}

// ---- vf_cnr_mask_f32: the 202-CS-SN-1 automatic CNR mask with the residual kept on the device ----

// residual[i] = image[i] - background[i]. Plain float32 subtraction, matching the NumPy
// `image_float - background` of the reference; gpu/cuda_project.json builds with --fmad=false, so
// nothing is contracted and the result is the IEEE round-to-nearest difference.
__global__ void cnr_residual_kernel(
    const float* image, const float* background, float* residual, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    residual[index] = image[index] - background[index];
}

// absdev[i] = |residual[i] - residual_median|. `np.abs` on a float32 array only clears the sign bit,
// which is exactly what fabsf does, and the subtraction is the same float32 operation the reference
// performs against the Python float holding the decoded median.
__global__ void cnr_absdev_kernel(
    const float* residual, float* absdev, int count, float residual_median) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    absdev[index] = fabsf(residual[index] - residual_median);
}

// mask[i] = (absdev[i] > threshold) ? candidate_value : 0. The comparison is the strict `>` of
// `np.abs(residual - residual_median) > residual_threshold`, performed in float32 because NumPy
// narrows the Python float scalar to the array dtype. Reading the absolute deviation that the
// previous pass already computed is the same value the reference compares.
__global__ void cnr_mask_kernel(
    const float* absdev, unsigned char* mask, int count, float threshold,
    unsigned char candidate_value) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    mask[index] = absdev[index] > threshold ? candidate_value : static_cast<unsigned char>(0);
}

// Convert a resident uint8 ROI directly to the float32 gray operand used by Detector202_1. The BGR
// expression deliberately rounds to uint8 first and only then promotes, matching
// cv2.cvtColor(..., COLOR_BGR2GRAY).astype(np.float32) exactly.
__global__ void resident_gray_f32_kernel(
    const unsigned char* resident, int resident_width, int resident_channels,
    int offset_x, int offset_y, float* gray, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const size_t source_index =
        (static_cast<size_t>(y + offset_y) * resident_width + x + offset_x) * resident_channels;
    unsigned char value = resident[source_index];
    if (resident_channels == 3) {
        constexpr int gray_shift = 15;
        constexpr int blue_to_gray = 3735;
        constexpr int green_to_gray = 19235;
        constexpr int red_to_gray = 9798;
        value = static_cast<unsigned char>(
            (blue_to_gray * resident[source_index] +
             green_to_gray * resident[source_index + 1] +
             red_to_gray * resident[source_index + 2] +
             (1 << (gray_shift - 1))) >> gray_shift);
    }
    gray[static_cast<size_t>(y) * width + x] = static_cast<float>(value);
}

// Python's two-argument max(): the first argument is returned unless the second is strictly greater.
// That keeps a NaN first argument (Python's max propagates it) and ignores a NaN second argument,
// which is what the detector's threshold expressions rely on: `max(mad_scale * mad, noise_floor)`
// is NaN when the operand holds a NaN, while `max(residual_threshold_floor, ...)` then falls back to
// the floor. fmax/fmaxf would resolve both the other way.
double python_max(double first, double second) { return second > first ? second : first; }

// The residual central-moment threshold and candidate mask of 202-CS-SN-1, one additive ABI v1
// export. The full contract is documented in include/visionflow_cuda.h.
VF_CUDA_API int vf_cnr_mask_f32(
    void* context,
    const float* image, int image_stride,
    const float* background, int background_stride,
    int width, int height,
    double sigma_multiplier,
    double threshold_floor,
    double absolute_floor,
    double mad_scale,
    int candidate_value,
    float* out_residual_median,
    float* out_mad,
    double* out_threshold,
    unsigned char* out_mask,
    long long out_mask_capacity) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || image == nullptr || background == nullptr ||
        out_residual_median == nullptr || out_mad == nullptr || out_threshold == nullptr ||
        out_mask == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (width <= 0 || height <= 0 ||
        width > INT_MAX / static_cast<int>(sizeof(float))) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int minimum_stride = width * static_cast<int>(sizeof(float));
    if (image_stride < minimum_stride || background_stride < minimum_stride) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (candidate_value < 0 || candidate_value > 255) return VF_CUDA_INVALID_ARGUMENT;
    const long long requested =
        static_cast<long long>(width) * static_cast<long long>(height);
    // The radix sort indexes with a signed 32-bit offset, so a larger plane cannot be sorted here;
    // report it as unsupported so the caller restarts the step on the CPU reference.
    if (requested > static_cast<long long>(INT_MAX)) return VF_CUDA_UNSUPPORTED;
    if (out_mask_capacity < requested) return VF_CUDA_INVALID_ARGUMENT;

    const int items = static_cast<int>(requested);
    const size_t item_count = static_cast<size_t>(items);
    int result = reserve_device(
        &persistent->cnr_mask_image, &persistent->cnr_mask_image_capacity, item_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->cnr_mask_background, &persistent->cnr_mask_background_capacity, item_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->cnr_mask_residual, &persistent->cnr_mask_residual_capacity, item_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->cnr_mask_absdev, &persistent->cnr_mask_absdev_capacity, item_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->cnr_mask_mask, &persistent->cnr_mask_mask_capacity, item_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_median_scratch(persistent, items);
    if (result != VF_CUDA_OK) return result;

    reset_timing(persistent, true);
    const size_t row_bytes = static_cast<size_t>(width) * sizeof(float);
    cudaError_t error = cudaMemcpy2DAsync(
        persistent->cnr_mask_image, row_bytes, image, static_cast<size_t>(image_stride),
        row_bytes, static_cast<size_t>(height), cudaMemcpyHostToDevice, persistent->stream);
    if (error == cudaSuccess) {
        error = cudaMemcpy2DAsync(
            persistent->cnr_mask_background, row_bytes, background,
            static_cast<size_t>(background_stride), row_bytes, static_cast<size_t>(height),
            cudaMemcpyHostToDevice, persistent->stream);
    }
    if (error != cudaSuccess) return cuda_result(error);
    record_timing_event(persistent, TIMING_AFTER_INPUT);

    constexpr int CNR_THREADS = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        (static_cast<long long>(items) + CNR_THREADS - 1) / CNR_THREADS);
    cnr_residual_kernel<<<blocks, CNR_THREADS, 0, persistent->stream>>>(
        persistent->cnr_mask_image, persistent->cnr_mask_background,
        persistent->cnr_mask_residual, items);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    float residual_median = 0.0f;
    result = run_device_median(
        persistent, persistent->cnr_mask_residual, items, &residual_median, false);
    if (result != VF_CUDA_OK) return result;

    cnr_absdev_kernel<<<blocks, CNR_THREADS, 0, persistent->stream>>>(
        persistent->cnr_mask_residual, persistent->cnr_mask_absdev, items, residual_median);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    float mad = 0.0f;
    result = run_device_median(persistent, persistent->cnr_mask_absdev, items, &mad, false);
    if (result != VF_CUDA_OK) return result;

    // The detector holds both medians as Python floats, so every remaining operation in
    // _automatic_cnr_mask is a double computation: `mad_scale * mad`, the two Python max() calls and
    // `residual_sigma_multiplier * robust_noise_sigma`. Reproducing that nesting in double keeps the
    // threshold bit-identical; the mask then narrows it to float32 for the comparison, which is what
    // NumPy does with a Python float operand.
    const double robust_noise_sigma = python_max(mad_scale * static_cast<double>(mad), absolute_floor);
    const double residual_threshold =
        python_max(threshold_floor, sigma_multiplier * robust_noise_sigma);
    const float threshold_f32 = static_cast<float>(residual_threshold);
    cnr_mask_kernel<<<blocks, CNR_THREADS, 0, persistent->stream>>>(
        persistent->cnr_mask_absdev, persistent->cnr_mask_mask, items, threshold_f32,
        static_cast<unsigned char>(candidate_value));
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    // AFTER_KERNEL covers the whole device pipeline of this export, including the two median
    // readbacks (the sort keys must reach the host to be decoded), so kernel_ms here is the device
    // cost of the complete step rather than a single kernel launch.
    record_timing_event(persistent, TIMING_AFTER_KERNEL);

    const size_t mask_row_bytes = static_cast<size_t>(width);
    error = cudaMemcpy2DAsync(
        out_mask, mask_row_bytes, persistent->cnr_mask_mask, mask_row_bytes, mask_row_bytes,
        static_cast<size_t>(height), cudaMemcpyDeviceToHost, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    record_timing_event(persistent, TIMING_AFTER_OUTPUT);
    result = visionflow_cuda::stream_result(persistent->stream);
    if (result != VF_CUDA_OK) return result;
    finalize_timing(persistent);

    *out_residual_median = residual_median;
    *out_mad = mad;
    *out_threshold = residual_threshold;
    return VF_CUDA_OK;
}

// Resident-ROI 202 CNR chain shared by vf_cnr_mask_u8_roi and vf_cnr_candidates_u8_roi: BGR->gray
// float32 into cnr_mask_image, float32 Gaussian, residual, both exact medians, the double threshold
// and the float32-compared candidate mask left in cnr_mask_mask. Nothing crosses PCIe except the
// median keys run_device_median reads. The caller validated the ROI and parameters, owns the
// AFTER_KERNEL/AFTER_OUTPUT timing events and decides what to download.
int resident_cnr_mask_device(
    PersistentContext* persistent,
    int x, int y, int width, int height,
    int kernel_size, double sigma,
    double sigma_multiplier,
    double threshold_floor,
    double absolute_floor,
    double mad_scale,
    int candidate_value,
    float* out_residual_median,
    float* out_mad,
    double* out_threshold) {
    const int items = width * height;
    const size_t item_count = static_cast<size_t>(items);
    int result = reserve_device(
        &persistent->cnr_mask_image, &persistent->cnr_mask_image_capacity, item_count,
        &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->gaussian_f32_intermediate, &persistent->gaussian_f32_intermediate_capacity,
        item_count, &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->gaussian_f32_output, &persistent->gaussian_f32_output_capacity, item_count,
        &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cnr_mask_residual, &persistent->cnr_mask_residual_capacity, item_count,
        &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cnr_mask_absdev, &persistent->cnr_mask_absdev_capacity, item_count,
        &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cnr_mask_mask, &persistent->cnr_mask_mask_capacity, item_count,
        &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_median_scratch(persistent, items);
    if (result != VF_CUDA_OK) return result;

    reset_timing(persistent, false);
    resident_gray_f32_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        persistent->resident_u8, persistent->resident_width, persistent->resident_channels,
        x, y, persistent->cnr_mask_image, width, height);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    record_timing_event(persistent, TIMING_AFTER_INPUT);

    int radius = 0;
    result = prepare_gaussian_f32_weights(kernel_size, sigma, &radius, persistent->stream);
    if (result != VF_CUDA_OK) return result;
    record_timing_event(persistent, TIMING_GAUSSIAN_START);
    launch_gaussian_f32(
        persistent->cnr_mask_image, persistent->gaussian_f32_intermediate,
        persistent->gaussian_f32_output, width, height, radius, persistent->stream);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    record_timing_event(persistent, TIMING_GAUSSIAN_END);
    persistent->timing_has_gaussian = true;

    constexpr int CNR_THREADS = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        (static_cast<long long>(items) + CNR_THREADS - 1) / CNR_THREADS);
    cnr_residual_kernel<<<blocks, CNR_THREADS, 0, persistent->stream>>>(
        persistent->cnr_mask_image, persistent->gaussian_f32_output,
        persistent->cnr_mask_residual, items);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    float residual_median = 0.0f;
    result = run_device_median(
        persistent, persistent->cnr_mask_residual, items, &residual_median, false);
    if (result != VF_CUDA_OK) return result;
    cnr_absdev_kernel<<<blocks, CNR_THREADS, 0, persistent->stream>>>(
        persistent->cnr_mask_residual, persistent->cnr_mask_absdev, items, residual_median);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    float mad = 0.0f;
    result = run_device_median(persistent, persistent->cnr_mask_absdev, items, &mad, false);
    if (result != VF_CUDA_OK) return result;
    const double robust_noise_sigma = python_max(mad_scale * static_cast<double>(mad), absolute_floor);
    const double residual_threshold = python_max(threshold_floor, sigma_multiplier * robust_noise_sigma);
    cnr_mask_kernel<<<blocks, CNR_THREADS, 0, persistent->stream>>>(
        persistent->cnr_mask_absdev, persistent->cnr_mask_mask, items,
        static_cast<float>(residual_threshold), static_cast<unsigned char>(candidate_value));
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    *out_residual_median = residual_median;
    *out_mad = mad;
    *out_threshold = residual_threshold;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_cnr_mask_u8_roi(
    void* context,
    uint64_t generation,
    int x, int y, int width, int height,
    int kernel_size, double sigma,
    double sigma_multiplier,
    double threshold_floor,
    double absolute_floor,
    double mad_scale,
    int candidate_value,
    float* out_residual_median,
    float* out_mad,
    double* out_threshold,
    unsigned char* out_mask,
    long long out_mask_capacity) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || out_residual_median == nullptr || out_mad == nullptr ||
        out_threshold == nullptr || out_mask == nullptr || generation == 0 ||
        generation != persistent->resident_generation || persistent->resident_u8 == nullptr ||
        (persistent->resident_channels != 1 && persistent->resident_channels != 3) ||
        width <= 0 || height <= 0 || x < 0 || y < 0 ||
        x > persistent->resident_width - width || y > persistent->resident_height - height ||
        candidate_value < 0 || candidate_value > 255) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int kernel_status = gaussian_f32_kernel_request(kernel_size, sigma);
    if (kernel_status != VF_CUDA_OK) return kernel_status;
    const long long requested = static_cast<long long>(width) * static_cast<long long>(height);
    if (requested > static_cast<long long>(INT_MAX)) return VF_CUDA_UNSUPPORTED;
    if (out_mask_capacity < requested) return VF_CUDA_INVALID_ARGUMENT;

    float residual_median = 0.0f;
    float mad = 0.0f;
    double residual_threshold = 0.0;
    int result = resident_cnr_mask_device(
        persistent, x, y, width, height, kernel_size, sigma, sigma_multiplier, threshold_floor,
        absolute_floor, mad_scale, candidate_value, &residual_median, &mad, &residual_threshold);
    if (result != VF_CUDA_OK) return result;
    record_timing_event(persistent, TIMING_AFTER_KERNEL);

    const size_t mask_row_bytes = static_cast<size_t>(width);
    cudaError_t error = cudaMemcpy2DAsync(
        out_mask, mask_row_bytes, persistent->cnr_mask_mask, mask_row_bytes,
        mask_row_bytes, static_cast<size_t>(height), cudaMemcpyDeviceToHost, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    record_timing_event(persistent, TIMING_AFTER_OUTPUT);
    result = visionflow_cuda::stream_result(persistent->stream);
    if (result != VF_CUDA_OK) return result;
    finalize_timing(persistent);
    *out_residual_median = residual_median;
    *out_mad = mad;
    *out_threshold = residual_threshold;
    return VF_CUDA_OK;
}

// ---- vf_cnr_candidates_u8_roi: 202-CS-SN-1 candidates without host image, mask or label map ----
//
// Detector202_1 finishes the CNR mask on the host with a morphology pass, the center/edge
// exclusion AND, cv2.connectedComponentsWithStats and a per-component ring CNR computed with
// np.mean/np.std on boolean-mask gathers. This export keeps that whole tail on the device and
// downloads only one fixed-size record per surviving component. The full contract is documented in
// include/visionflow_cuda.h.

constexpr int CAND_INT_PARAMS = 22;
constexpr int CAND_REAL_PARAMS = 6;
constexpr int CAND_RECORD_INTS = 7;
constexpr int CAND_RECORD_FLOATS = 3;
constexpr int CAND_THREADS = 256;
constexpr int CCL_MAX_ITERATIONS = 4096;
// A single ROI whose ring windows together exceed this many float32 samples is left to the host
// path instead of reserving several GiB of scratch for a pathological mask.
constexpr long long CAND_MAX_GATHER = 1LL << 28;

enum CandidateStatus {
    CAND_STATUS_OK = 0,
    CAND_STATUS_GLOBAL_BACKGROUND = 1,
    CAND_STATUS_CAPACITY = 2,
    CAND_STATUS_GATHER_LIMIT = 3,
};

// The center rectangle and edge insets Detector202._apply_exclusion_masks zeroes, already clamped by
// the caller exactly as the host computes them.
struct CandidateGeometry {
    int width;
    int height;
    int center_enabled;
    int center_x0, center_y0, center_x1, center_y1;
    int inset_top, inset_bottom, inset_left, inset_right;
};

__device__ __forceinline__ bool cand_included(const CandidateGeometry& g, int x, int y) {
    if (g.center_enabled && x >= g.center_x0 && x < g.center_x1 && y >= g.center_y0 && y < g.center_y1) {
        return false;
    }
    if (g.inset_top > 0 && y < g.inset_top) return false;
    if (g.inset_bottom > 0 && y >= g.height - g.inset_bottom) return false;
    if (g.inset_left > 0 && x < g.inset_left) return false;
    if (g.inset_right > 0 && x >= g.width - g.inset_right) return false;
    return true;
}

// cv2.bitwise_and(candidate_mask, inclusion_mask) with a 0/255 inclusion mask.
__global__ void cand_exclusion_kernel(unsigned char* mask, CandidateGeometry geometry) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geometry.width || y >= geometry.height) return;
    if (!cand_included(geometry, x, y)) mask[static_cast<size_t>(y) * geometry.width + x] = 0;
}

__global__ void ccl_init_kernel(const unsigned char* mask, int32_t* parent, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    parent[index] = mask[index] != 0 ? index : -1;
}

__global__ void cand_ramp_kernel(int32_t* ramp, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    ramp[index] = index;
}

// Every parent points at a smaller or equal index, so following it always reaches a root even while
// other threads lower entries concurrently.
__device__ __forceinline__ int32_t ccl_find(const int32_t* parent, int32_t index) {
    while (parent[index] != index) index = parent[index];
    return index;
}

// Hook step of a data-parallel union-find. For each foreground pixel and each backward neighbour of
// the requested connectivity, the larger root is lowered towards the smaller one. Concurrent writes
// only ever decrease a parent below its own index, so no cycle can form; a lost write keeps two roots
// apart for one more iteration, and `changed` forces that iteration. When a full pass observes no
// differing roots, no thread wrote anything and every adjacent pair already shares its root.
__global__ void ccl_hook_kernel(int32_t* parent, int width, int height, int connectivity, int32_t* changed) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int32_t index = y * width + x;
    if (parent[index] < 0) return;
    const int neighbour_dx[4] = {-1, 0, -1, 1};
    const int neighbour_dy[4] = {0, -1, -1, -1};
    const int neighbours = connectivity == 8 ? 4 : 2;
    for (int n = 0; n < neighbours; ++n) {
        const int nx = x + neighbour_dx[n];
        const int ny = y + neighbour_dy[n];
        if (nx < 0 || nx >= width || ny < 0) continue;
        const int32_t other = ny * width + nx;
        if (parent[other] < 0) continue;
        const int32_t a = ccl_find(parent, index);
        const int32_t b = ccl_find(parent, other);
        if (a == b) continue;
        const int32_t high = a > b ? a : b;
        const int32_t low = a > b ? b : a;
        if (parent[high] > low) parent[high] = low;
        changed[0] = 1;
    }
}

__global__ void ccl_compress_kernel(int32_t* parent, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count || parent[index] < 0) return;
    parent[index] = ccl_find(parent, index);
}

__global__ void cand_key_kernel(const int32_t* foreground, const int32_t* parent, int32_t* keys, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    keys[index] = parent[foreground[index]];
}

// Per component: the bounding box and area cv2.connectedComponentsWithStats reports, then the area
// and border-margin filters of _collect_candidates_with_labels. Pixels are grouped in raster order,
// so the first pixel carries the top row.
__global__ void cand_component_kernel(
    const int32_t* sorted_pixels, const int32_t* offsets, const int32_t* areas, int count,
    int width, int height, int min_area, int max_area, int max_area_enabled, int border_margin,
    int32_t* boxes, unsigned char* keep) {
    const int component = blockIdx.x * blockDim.x + threadIdx.x;
    if (component >= count) return;
    const int32_t start = offsets[component];
    const int32_t area = areas[component];
    int min_x = width;
    int max_x = -1;
    int max_y = -1;
    const int min_y = sorted_pixels[start] / width;
    for (int32_t k = 0; k < area; ++k) {
        const int32_t pixel = sorted_pixels[start + k];
        const int px = pixel % width;
        const int py = pixel / width;
        if (px < min_x) min_x = px;
        if (px > max_x) max_x = px;
        if (py > max_y) max_y = py;
    }
    const int box_width = max_x - min_x + 1;
    const int box_height = max_y - min_y + 1;
    int32_t* box = boxes + static_cast<size_t>(component) * 5;
    box[0] = min_x;
    box[1] = min_y;
    box[2] = box_width;
    box[3] = box_height;
    box[4] = area;
    bool kept = !(area < min_area || (max_area_enabled && area > max_area));
    if (kept && (static_cast<long long>(min_x) <= border_margin ||
                 static_cast<long long>(min_y) <= border_margin ||
                 static_cast<long long>(min_x) + box_width >= static_cast<long long>(width) - border_margin ||
                 static_cast<long long>(min_y) + box_height >= static_cast<long long>(height) - border_margin)) {
        kept = false;
    }
    keep[component] = kept ? 1 : 0;
}

// Python: pad = int(max(padding_min, min(padding_max, max(w, h) * padding_scale))). Python's
// two-argument min/max return the first argument unless the second is strictly smaller/greater, and
// int() truncates towards zero. The window is the clamped slice [start, stop) of that padding.
__global__ void cand_window_kernel(
    const int32_t* kept, const int32_t* boxes, int count, int width, int height,
    int padding_min, int padding_max, double padding_scale, int32_t* windows, long long* sizes) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= count) return;
    const int32_t* box = boxes + static_cast<size_t>(kept[candidate]) * 5;
    const int longest = box[2] > box[3] ? box[2] : box[3];
    const double product = static_cast<double>(longest) * padding_scale;
    const double limited = product < static_cast<double>(padding_max) ? product : static_cast<double>(padding_max);
    const double padded = limited > static_cast<double>(padding_min) ? limited : static_cast<double>(padding_min);
    const long long pad = static_cast<long long>(padded);
    long long x_start = static_cast<long long>(box[0]) - pad;
    long long y_start = static_cast<long long>(box[1]) - pad;
    long long x_stop = static_cast<long long>(box[0]) + box[2] + pad;
    long long y_stop = static_cast<long long>(box[1]) + box[3] + pad;
    if (x_start < 0) x_start = 0;
    if (y_start < 0) y_start = 0;
    if (x_stop > width) x_stop = width;
    if (y_stop > height) y_stop = height;
    int32_t* window = windows + static_cast<size_t>(candidate) * 4;
    window[0] = static_cast<int32_t>(x_start);
    window[1] = static_cast<int32_t>(y_start);
    window[2] = static_cast<int32_t>(x_stop > x_start ? x_stop : x_start);
    window[3] = static_cast<int32_t>(y_stop > y_start ? y_stop : y_start);
    sizes[candidate] = static_cast<long long>(window[2] - window[0]) * (window[3] - window[1]);
}

// NumPy's float32 add.reduce (pairwise_sum in numpy/_core/src/umath/loops_utils.h.src) over
// value(start) ... value(start + n - 1): fewer than 8 values accumulate sequentially from -0.0, up to
// 128 values use eight lanes combined as ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)) plus the remainder,
// and longer runs split at n/2 rounded down to a multiple of 8. The recursion is evaluated with an
// explicit stack because the combination order, not just the leaves, fixes the float32 result.
// `indices` selects value(i) = values[indices[i]] (component pixels) instead of values[i]; with
// `squared`, value(i) = (v - mean) * (v - mean) as np.subtract then np.square produce it.
__device__ __forceinline__ float cand_value(
    const float* values, const int32_t* indices, long long i, bool squared, float mean) {
    float value = indices != nullptr ? values[indices[i]] : values[i];
    if (squared) {
        const float deviation = value - mean;
        value = deviation * deviation;
    }
    return value;
}

__device__ float cand_pairwise_leaf(
    const float* values, const int32_t* indices, long long start, long long n, bool squared, float mean) {
    if (n < 8) {
        float result = -0.0f;
        for (long long i = 0; i < n; ++i) result += cand_value(values, indices, start + i, squared, mean);
        return result;
    }
    float r[8];
    for (int lane = 0; lane < 8; ++lane) r[lane] = cand_value(values, indices, start + lane, squared, mean);
    long long i = 8;
    for (; i < n - (n % 8); i += 8) {
        for (int lane = 0; lane < 8; ++lane) r[lane] += cand_value(values, indices, start + i + lane, squared, mean);
    }
    float result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) result += cand_value(values, indices, start + i, squared, mean);
    return result;
}

// The pairwise recursion splits a run of n > 128 values at n/2 rounded down to a multiple of 8 and
// evaluates everything else as leaves. The leaf layout depends only on n, so the statistics are built
// in three data-parallel phases instead of one long loop per candidate: every leaf of every sequence is
// summed in its own device thread, then each sequence walks the same recursion over its leaf sums in
// left-to-right order. Only that walk fixes the float32 combination order, and it touches about n/100
// values, so no thread runs a window-sized loop.
constexpr long long CAND_LEAF_SIZE = 128;
constexpr int CAND_STACK = 64;

__device__ __forceinline__ long long cand_split(long long count) {
    long long half = count / 2;
    return half - half % 8;
}

__device__ long long cand_leaf_count(long long n) {
    if (n <= 0) return 0;
    long long pending[CAND_STACK];
    int top = 0;
    long long leaves = 0;
    pending[top++] = n;
    while (top > 0) {
        const long long count = pending[--top];
        if (count <= CAND_LEAF_SIZE) {
            ++leaves;
            continue;
        }
        const long long half = cand_split(count);
        pending[top++] = count - half;
        pending[top++] = half;
    }
    return leaves;
}

// Emits the leaves of one sequence in left-to-right order, which is the order the walk consumes them.
__device__ void cand_emit_leaves(
    long long start, long long n, int sequence, long long first_leaf,
    long long* leaf_start, int32_t* leaf_length, int32_t* leaf_sequence) {
    if (n <= 0) return;
    long long pending_start[CAND_STACK];
    long long pending_n[CAND_STACK];
    int top = 0;
    long long leaf = first_leaf;
    pending_start[top] = start;
    pending_n[top] = n;
    ++top;
    while (top > 0) {
        --top;
        const long long base = pending_start[top];
        const long long count = pending_n[top];
        if (count <= CAND_LEAF_SIZE) {
            leaf_start[leaf] = base;
            leaf_length[leaf] = static_cast<int32_t>(count);
            leaf_sequence[leaf] = sequence;
            ++leaf;
            continue;
        }
        const long long half = cand_split(count);
        pending_start[top] = base + half;
        pending_n[top] = count - half;
        ++top;
        pending_start[top] = base;
        pending_n[top] = half;
        ++top;
    }
}

// The same recursion as NumPy's pairwise_sum, reading the precomputed leaf sums in order.
__device__ float cand_combine_leaves(const float* leaf_values, long long first_leaf, long long n) {
    long long frame_n[CAND_STACK];
    int frame_stage[CAND_STACK];
    float partial[CAND_STACK];
    int frames = 1;
    int partials = 0;
    long long leaf = first_leaf;
    frame_n[0] = n;
    frame_stage[0] = 0;
    while (frames > 0) {
        const int top = frames - 1;
        const long long count = frame_n[top];
        if (count <= CAND_LEAF_SIZE) {
            partial[partials++] = leaf_values[leaf++];
            --frames;
            continue;
        }
        const long long half = cand_split(count);
        if (frame_stage[top] == 0) {
            frame_stage[top] = 1;
            frame_n[frames] = half;
            frame_stage[frames] = 0;
            ++frames;
        } else if (frame_stage[top] == 1) {
            frame_stage[top] = 2;
            frame_n[frames] = count - half;
            frame_stage[frames] = 0;
            ++frames;
        } else {
            const float right = partial[--partials];
            const float left = partial[--partials];
            partial[partials++] = left + right;
            --frames;
        }
    }
    return partial[0];
}

// The kept candidate whose half-open slot range [offsets[i], offsets[i + 1]) contains `slot`. Empty
// ranges share their start with the next range, so the largest start not above the slot is the
// non-empty range that owns it.
__device__ __forceinline__ int cand_owner(const long long* offsets, int count, long long slot) {
    int low = 0;
    int high = count - 1;
    while (low < high) {
        const int middle = low + (high - low + 1) / 2;
        if (offsets[middle] <= slot) low = middle;
        else high = middle - 1;
    }
    return low;
}

__device__ __forceinline__ long long cand_slot() {
    return static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
}

// Component pixel values in raster order, one thread per pixel of every kept component.
__global__ void cand_component_values_kernel(
    const long long* value_offsets, int count, const int32_t* kept, const int32_t* component_offsets,
    const int32_t* sorted_pixels, const float* gray, float* values, long long total) {
    const long long slot = cand_slot();
    if (slot >= total) return;
    const int candidate = cand_owner(value_offsets, count, slot);
    const long long local = slot - value_offsets[candidate];
    values[slot] = gray[sorted_pixels[component_offsets[kept[candidate]] + local]];
}

// Ring background candidates, one thread per pixel of every clamped window: the value and whether the
// host gather `local_image[(local_labels != label) & local_inclusion]` keeps it.
__global__ void cand_background_slots_kernel(
    const long long* window_offsets, int count, const int32_t* kept, const int32_t* windows,
    const int32_t* roots, const int32_t* parent, const float* gray, CandidateGeometry geometry,
    unsigned char* flags, float* values, long long total) {
    const long long slot = cand_slot();
    if (slot >= total) return;
    const int candidate = cand_owner(window_offsets, count, slot);
    const int32_t* window = windows + static_cast<size_t>(candidate) * 4;
    const long long window_width = window[2] - window[0];
    const long long local = slot - window_offsets[candidate];
    const int xx = window[0] + static_cast<int>(local % window_width);
    const int yy = window[1] + static_cast<int>(local / window_width);
    const int32_t pixel = yy * geometry.width + xx;
    flags[slot] = (parent[pixel] != roots[kept[candidate]] && cand_included(geometry, xx, yy)) ? 1 : 0;
    values[slot] = gray[pixel];
}

__global__ void cand_segment_ends_kernel(
    const long long* offsets, const long long* sizes, long long* ends, int count) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= count) return;
    ends[candidate] = offsets[candidate] + sizes[candidate];
}

// Sequence 2c holds candidate c's component values, sequence 2c+1 its compacted background values.
__global__ void cand_sequences_kernel(
    const long long* value_offsets, const int32_t* kept, const int32_t* boxes,
    const long long* background_offsets, const long long* background_counts, long long component_total,
    long long* sequence_start, long long* sequence_length, long long* leaf_counts, int count) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= count) return;
    const size_t component = static_cast<size_t>(candidate) * 2;
    sequence_start[component] = value_offsets[candidate];
    sequence_length[component] = boxes[static_cast<size_t>(kept[candidate]) * 5 + 4];
    sequence_start[component + 1] = component_total + background_offsets[candidate];
    sequence_length[component + 1] = background_counts[candidate];
    leaf_counts[component] = cand_leaf_count(sequence_length[component]);
    leaf_counts[component + 1] = cand_leaf_count(sequence_length[component + 1]);
}

__global__ void cand_emit_leaves_kernel(
    const long long* sequence_start, const long long* sequence_length, const long long* leaf_offsets,
    long long* leaf_start, int32_t* leaf_length, int32_t* leaf_sequence, int count) {
    const int sequence = blockIdx.x * blockDim.x + threadIdx.x;
    if (sequence >= count) return;
    cand_emit_leaves(
        sequence_start[sequence], sequence_length[sequence], sequence, leaf_offsets[sequence],
        leaf_start, leaf_length, leaf_sequence);
}

// Leaf sums; the squared pass applies np.subtract/np.square against the sequence mean and only runs
// for background sequences, which are the only ones np.std is taken of.
__global__ void cand_leaf_sum_kernel(
    const float* values, const long long* leaf_start, const int32_t* leaf_length,
    const int32_t* leaf_sequence, const float* sequence_mean, int squared, float* leaf_values,
    long long total) {
    const long long leaf = cand_slot();
    if (leaf >= total) return;
    const int sequence = leaf_sequence[leaf];
    if (squared && (sequence & 1) == 0) return;
    leaf_values[leaf] = cand_pairwise_leaf(
        values, nullptr, leaf_start[leaf], leaf_length[leaf], squared != 0,
        squared ? sequence_mean[sequence] : 0.0f);
}

// np.mean: float32(float64(sum) / float64(count)). np.std (_var): the same float64 division of the
// squared-deviation sum, then a float32 sqrt.
__global__ void cand_combine_kernel(
    const float* leaf_values, const long long* leaf_offsets, const long long* sequence_length,
    int squared, float* sequence_mean, float* sequence_std, int count) {
    const int sequence = blockIdx.x * blockDim.x + threadIdx.x;
    if (sequence >= count) return;
    const long long n = sequence_length[sequence];
    if (squared) {
        if ((sequence & 1) == 0) return;
        sequence_std[sequence] = 0.0f;
        if (n <= 0) return;
        const float squares = cand_combine_leaves(leaf_values, leaf_offsets[sequence], n);
        sequence_std[sequence] = sqrtf(static_cast<float>(static_cast<double>(squares) / static_cast<double>(n)));
        return;
    }
    sequence_mean[sequence] = 0.0f;
    if (n <= 0) return;
    const float total = cand_combine_leaves(leaf_values, leaf_offsets[sequence], n);
    sequence_mean[sequence] = static_cast<float>(static_cast<double>(total) / static_cast<double>(n));
}

__global__ void cand_records_kernel(
    const int32_t* kept, const int32_t* boxes, const long long* background_counts,
    const float* sequence_mean, const float* sequence_std, int min_background_pixels,
    int32_t* out_ints, float* out_floats, int count) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= count) return;
    const int32_t* box = boxes + static_cast<size_t>(kept[candidate]) * 5;
    int32_t* record = out_ints + static_cast<size_t>(candidate) * CAND_RECORD_INTS;
    float* stats = out_floats + static_cast<size_t>(candidate) * CAND_RECORD_FLOATS;
    const long long background = background_counts[candidate];
    const size_t sequence = static_cast<size_t>(candidate) * 2;
    for (int field = 0; field < 5; ++field) record[field] = box[field];
    record[5] = static_cast<int32_t>(background);
    record[6] = background < min_background_pixels ? CAND_STATUS_GLOBAL_BACKGROUND : CAND_STATUS_OK;
    stats[0] = sequence_mean[sequence];
    stats[1] = background > 0 ? sequence_mean[sequence + 1] : 0.0f;
    stats[2] = background > 0 ? sequence_std[sequence + 1] : 0.0f;
}

int cand_reserve_cub(PersistentContext* persistent, size_t bytes) {
    return reserve_device(
        &persistent->cand_cub_scratch, &persistent->cand_cub_scratch_capacity,
        bytes > 0 ? bytes : static_cast<size_t>(1), &persistent->allocation_count);
}

unsigned int cand_blocks(long long items) {
    return static_cast<unsigned int>((items + CAND_THREADS - 1) / CAND_THREADS);
}

int cand_read_word(PersistentContext* persistent, int32_t* host, const int32_t* device) {
    cudaError_t error = cudaMemcpyAsync(
        host, device, sizeof(int32_t), cudaMemcpyDeviceToHost, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    return visionflow_cuda::stream_result(persistent->stream);
}

// Groups foreground pixels by component and fills cand_roots/cand_areas/cand_offsets/cand_boxes and
// the kept-component list. Returns the component and kept counts through the output pointers.
int cand_group_components(
    PersistentContext* persistent, const unsigned char* mask, int items, int width, int height,
    int min_area, int max_area, int max_area_enabled, int border_margin,
    int* out_component_count, int* out_kept_count) {
    *out_component_count = 0;
    *out_kept_count = 0;
    size_t cub_bytes = 0;
    cudaError_t error = cub::DeviceSelect::Flagged(
        nullptr, cub_bytes, persistent->cand_ramp, mask, persistent->cand_foreground,
        persistent->cand_words, items, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    int result = cand_reserve_cub(persistent, cub_bytes);
    if (result != VF_CUDA_OK) return result;
    error = cub::DeviceSelect::Flagged(
        persistent->cand_cub_scratch, cub_bytes, persistent->cand_ramp, mask, persistent->cand_foreground,
        persistent->cand_words, items, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    int32_t word = 0;
    result = cand_read_word(persistent, &word, persistent->cand_words);
    if (result != VF_CUDA_OK) return result;
    const int foreground = word;
    if (foreground == 0) return VF_CUDA_OK;

    const size_t foreground_count = static_cast<size_t>(foreground);
    result = reserve_device(&persistent->cand_keys, &persistent->cand_keys_capacity, foreground_count,
                            &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_sorted_keys, &persistent->cand_sorted_keys_capacity, foreground_count,
        &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_sorted_pixels, &persistent->cand_sorted_pixels_capacity, foreground_count,
        &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_roots, &persistent->cand_roots_capacity, foreground_count,
        &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_areas, &persistent->cand_areas_capacity, foreground_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    cand_key_kernel<<<cand_blocks(foreground), CAND_THREADS, 0, persistent->stream>>>(
        persistent->cand_foreground, persistent->ccl_parent, persistent->cand_keys, foreground);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    int key_bits = 1;
    while (key_bits < 31 && (1LL << key_bits) < static_cast<long long>(items)) ++key_bits;
    cub_bytes = 0;
    error = cub::DeviceRadixSort::SortPairs(
        nullptr, cub_bytes, persistent->cand_keys, persistent->cand_sorted_keys,
        persistent->cand_foreground, persistent->cand_sorted_pixels, foreground, 0, key_bits,
        persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    result = cand_reserve_cub(persistent, cub_bytes);
    if (result != VF_CUDA_OK) return result;
    error = cub::DeviceRadixSort::SortPairs(
        persistent->cand_cub_scratch, cub_bytes, persistent->cand_keys, persistent->cand_sorted_keys,
        persistent->cand_foreground, persistent->cand_sorted_pixels, foreground, 0, key_bits,
        persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    cub_bytes = 0;
    error = cub::DeviceRunLengthEncode::Encode(
        nullptr, cub_bytes, persistent->cand_sorted_keys, persistent->cand_roots,
        persistent->cand_areas, persistent->cand_words, foreground, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    result = cand_reserve_cub(persistent, cub_bytes);
    if (result != VF_CUDA_OK) return result;
    error = cub::DeviceRunLengthEncode::Encode(
        persistent->cand_cub_scratch, cub_bytes, persistent->cand_sorted_keys, persistent->cand_roots,
        persistent->cand_areas, persistent->cand_words, foreground, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    result = cand_read_word(persistent, &word, persistent->cand_words);
    if (result != VF_CUDA_OK) return result;
    const int component_count = word;
    *out_component_count = component_count;
    const size_t components = static_cast<size_t>(component_count);

    result = reserve_device(&persistent->cand_offsets, &persistent->cand_offsets_capacity, components,
                            &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_boxes, &persistent->cand_boxes_capacity, components * 5,
        &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_keep, &persistent->cand_keep_capacity, components, &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_kept, &persistent->cand_kept_capacity, components, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    cub_bytes = 0;
    error = cub::DeviceScan::ExclusiveSum(
        nullptr, cub_bytes, persistent->cand_areas, persistent->cand_offsets, component_count,
        persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    result = cand_reserve_cub(persistent, cub_bytes);
    if (result != VF_CUDA_OK) return result;
    error = cub::DeviceScan::ExclusiveSum(
        persistent->cand_cub_scratch, cub_bytes, persistent->cand_areas, persistent->cand_offsets,
        component_count, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    cand_component_kernel<<<cand_blocks(component_count), CAND_THREADS, 0, persistent->stream>>>(
        persistent->cand_sorted_pixels, persistent->cand_offsets, persistent->cand_areas, component_count,
        width, height, min_area, max_area, max_area_enabled, border_margin,
        persistent->cand_boxes, persistent->cand_keep);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    cub_bytes = 0;
    error = cub::DeviceSelect::Flagged(
        nullptr, cub_bytes, persistent->cand_ramp, persistent->cand_keep, persistent->cand_kept,
        persistent->cand_words, component_count, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    result = cand_reserve_cub(persistent, cub_bytes);
    if (result != VF_CUDA_OK) return result;
    error = cub::DeviceSelect::Flagged(
        persistent->cand_cub_scratch, cub_bytes, persistent->cand_ramp, persistent->cand_keep,
        persistent->cand_kept, persistent->cand_words, component_count, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    result = cand_read_word(persistent, &word, persistent->cand_words);
    if (result != VF_CUDA_OK) return result;
    *out_kept_count = word;
    return VF_CUDA_OK;
}

__global__ void cand_kept_areas_kernel(const int32_t* kept, const int32_t* boxes, long long* areas, int count) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= count) return;
    areas[candidate] = boxes[static_cast<size_t>(kept[candidate]) * 5 + 4];
}

int cand_exclusive_sum(PersistentContext* persistent, const long long* input, long long* output, int count) {
    size_t cub_bytes = 0;
    cudaError_t error = cub::DeviceScan::ExclusiveSum(nullptr, cub_bytes, input, output, count, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    int result = cand_reserve_cub(persistent, cub_bytes);
    if (result != VF_CUDA_OK) return result;
    error = cub::DeviceScan::ExclusiveSum(
        persistent->cand_cub_scratch, cub_bytes, input, output, count, persistent->stream);
    return error == cudaSuccess ? VF_CUDA_OK : cuda_result(error);
}

// Reads offsets[count - 1] + sizes[count - 1], the total a prefix sum covers.
int cand_read_total(
    PersistentContext* persistent, const long long* offsets, const long long* sizes, int count, long long* total) {
    long long last_offset = 0;
    long long last_size = 0;
    cudaError_t error = cudaMemcpyAsync(
        &last_offset, offsets + (count - 1), sizeof(long long), cudaMemcpyDeviceToHost, persistent->stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        &last_size, sizes + (count - 1), sizeof(long long), cudaMemcpyDeviceToHost, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    int result = visionflow_cuda::stream_result(persistent->stream);
    if (result != VF_CUDA_OK) return result;
    *total = last_offset + last_size;
    return VF_CUDA_OK;
}

// Ring statistics for `kept_count` candidates whose windows and window offsets are already on the
// device, written into cand_out_ints/cand_out_floats. See the leaf-layout comment above.
int cand_ring_statistics(
    PersistentContext* persistent, int kept_count, long long window_total, CandidateGeometry geometry,
    int min_background_pixels) {
    const size_t kept = static_cast<size_t>(kept_count);
    const int sequences = kept_count * 2;
    int result = reserve_device(
        &persistent->cand_value_offsets, &persistent->cand_value_offsets_capacity, kept, &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_segment_ends, &persistent->cand_segment_ends_capacity, kept, &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_background_counts, &persistent->cand_background_counts_capacity, kept,
        &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_background_offsets, &persistent->cand_background_offsets_capacity, kept,
        &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_seq_start, &persistent->cand_seq_start_capacity, kept * 2, &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_seq_length, &persistent->cand_seq_length_capacity, kept * 2, &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_leaf_counts, &persistent->cand_leaf_counts_capacity, kept * 2, &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_leaf_offsets, &persistent->cand_leaf_offsets_capacity, kept * 2,
        &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_seq_mean, &persistent->cand_seq_mean_capacity, kept * 2, &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_seq_std, &persistent->cand_seq_std_capacity, kept * 2, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    const unsigned int candidate_blocks = cand_blocks(kept_count);

    // Component values: one prefix sum over the kept areas, then one thread per component pixel.
    cand_kept_areas_kernel<<<candidate_blocks, CAND_THREADS, 0, persistent->stream>>>(
        persistent->cand_kept, persistent->cand_boxes, persistent->cand_segment_ends, kept_count);
    result = visionflow_cuda::kernel_launch_result();
    if (result == VF_CUDA_OK) result = cand_exclusive_sum(
        persistent, persistent->cand_segment_ends, persistent->cand_value_offsets, kept_count);
    long long component_total = 0;
    if (result == VF_CUDA_OK) result = cand_read_total(
        persistent, persistent->cand_value_offsets, persistent->cand_segment_ends, kept_count, &component_total);
    if (result != VF_CUDA_OK) return result;
    const long long value_total = component_total + window_total;
    if (value_total > static_cast<long long>(INT_MAX)) return VF_CUDA_UNSUPPORTED;
    result = reserve_device(
        &persistent->cand_values, &persistent->cand_values_capacity, static_cast<size_t>(value_total),
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    cand_component_values_kernel<<<cand_blocks(component_total), CAND_THREADS, 0, persistent->stream>>>(
        persistent->cand_value_offsets, kept_count, persistent->cand_kept, persistent->cand_offsets,
        persistent->cand_sorted_pixels, persistent->cnr_mask_image, persistent->cand_values, component_total);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    // Background: classify every window pixel in parallel, compact in order, count per candidate.
    if (window_total > 0) {
        const size_t window_count = static_cast<size_t>(window_total);
        result = reserve_device(
            &persistent->cand_flags, &persistent->cand_flags_capacity, window_count, &persistent->allocation_count);
        if (result == VF_CUDA_OK) result = reserve_device(
            &persistent->cand_gather, &persistent->cand_gather_capacity, window_count, &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
        cand_background_slots_kernel<<<cand_blocks(window_total), CAND_THREADS, 0, persistent->stream>>>(
            persistent->cand_gather_offsets, kept_count, persistent->cand_kept, persistent->cand_windows,
            persistent->cand_roots, persistent->ccl_parent, persistent->cnr_mask_image, geometry,
            persistent->cand_flags, persistent->cand_gather, window_total);
        cand_segment_ends_kernel<<<candidate_blocks, CAND_THREADS, 0, persistent->stream>>>(
            persistent->cand_gather_offsets, persistent->cand_window_sizes, persistent->cand_segment_ends, kept_count);
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
        const int window_items = static_cast<int>(window_total);
        size_t cub_bytes = 0;
        cudaError_t error = cub::DeviceSelect::Flagged(
            nullptr, cub_bytes, persistent->cand_gather, persistent->cand_flags,
            persistent->cand_values + component_total, persistent->cand_words, window_items, persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        result = cand_reserve_cub(persistent, cub_bytes);
        if (result != VF_CUDA_OK) return result;
        error = cub::DeviceSelect::Flagged(
            persistent->cand_cub_scratch, cub_bytes, persistent->cand_gather, persistent->cand_flags,
            persistent->cand_values + component_total, persistent->cand_words, window_items, persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        cub_bytes = 0;
        error = cub::DeviceSegmentedReduce::Sum(
            nullptr, cub_bytes, persistent->cand_flags, persistent->cand_background_counts, kept_count,
            persistent->cand_gather_offsets, persistent->cand_segment_ends, persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        result = cand_reserve_cub(persistent, cub_bytes);
        if (result != VF_CUDA_OK) return result;
        error = cub::DeviceSegmentedReduce::Sum(
            persistent->cand_cub_scratch, cub_bytes, persistent->cand_flags, persistent->cand_background_counts,
            kept_count, persistent->cand_gather_offsets, persistent->cand_segment_ends, persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
    } else {
        cudaError_t error = cudaMemsetAsync(
            persistent->cand_background_counts, 0, sizeof(long long) * kept, persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
    }
    result = cand_exclusive_sum(
        persistent, persistent->cand_background_counts, persistent->cand_background_offsets, kept_count);
    if (result != VF_CUDA_OK) return result;

    // Sequences and their pairwise leaf layout.
    cand_sequences_kernel<<<candidate_blocks, CAND_THREADS, 0, persistent->stream>>>(
        persistent->cand_value_offsets, persistent->cand_kept, persistent->cand_boxes,
        persistent->cand_background_offsets, persistent->cand_background_counts, component_total,
        persistent->cand_seq_start, persistent->cand_seq_length, persistent->cand_leaf_counts, kept_count);
    result = visionflow_cuda::kernel_launch_result();
    if (result == VF_CUDA_OK) result = cand_exclusive_sum(
        persistent, persistent->cand_leaf_counts, persistent->cand_leaf_offsets, sequences);
    long long leaf_total = 0;
    if (result == VF_CUDA_OK) result = cand_read_total(
        persistent, persistent->cand_leaf_offsets, persistent->cand_leaf_counts, sequences, &leaf_total);
    if (result != VF_CUDA_OK) return result;
    const size_t leaves = static_cast<size_t>(leaf_total > 0 ? leaf_total : 1);
    result = reserve_device(
        &persistent->cand_leaf_start, &persistent->cand_leaf_start_capacity, leaves, &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_leaf_length, &persistent->cand_leaf_length_capacity, leaves, &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_leaf_sequence, &persistent->cand_leaf_sequence_capacity, leaves,
        &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_leaf_values, &persistent->cand_leaf_values_capacity, leaves, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    const unsigned int sequence_blocks = cand_blocks(sequences);
    const unsigned int leaf_blocks = cand_blocks(leaf_total > 0 ? leaf_total : 1);
    cand_emit_leaves_kernel<<<sequence_blocks, CAND_THREADS, 0, persistent->stream>>>(
        persistent->cand_seq_start, persistent->cand_seq_length, persistent->cand_leaf_offsets,
        persistent->cand_leaf_start, persistent->cand_leaf_length, persistent->cand_leaf_sequence, sequences);

    // Means for every sequence, then the squared-deviation sums for the background sequences.
    for (int squared = 0; squared <= 1; ++squared) {
        cand_leaf_sum_kernel<<<leaf_blocks, CAND_THREADS, 0, persistent->stream>>>(
            persistent->cand_values, persistent->cand_leaf_start, persistent->cand_leaf_length,
            persistent->cand_leaf_sequence, persistent->cand_seq_mean, squared, persistent->cand_leaf_values,
            leaf_total);
        cand_combine_kernel<<<sequence_blocks, CAND_THREADS, 0, persistent->stream>>>(
            persistent->cand_leaf_values, persistent->cand_leaf_offsets, persistent->cand_seq_length, squared,
            persistent->cand_seq_mean, persistent->cand_seq_std, sequences);
    }
    cand_records_kernel<<<candidate_blocks, CAND_THREADS, 0, persistent->stream>>>(
        persistent->cand_kept, persistent->cand_boxes, persistent->cand_background_counts,
        persistent->cand_seq_mean, persistent->cand_seq_std, min_background_pixels,
        persistent->cand_out_ints, persistent->cand_out_floats, kept_count);
    return visionflow_cuda::kernel_launch_result();
}

VF_CUDA_API int vf_cnr_candidates_u8_roi(
    void* context,
    uint64_t generation,
    int x, int y, int width, int height,
    const int32_t* int_params, int int_param_count,
    const double* real_params, int real_param_count,
    float* out_residual_median,
    float* out_mad,
    double* out_threshold,
    int32_t* out_candidate_ints,
    float* out_candidate_floats,
    int candidate_capacity,
    int* out_candidate_count,
    int* out_component_count,
    int* out_status) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || int_params == nullptr || real_params == nullptr ||
        out_residual_median == nullptr || out_mad == nullptr || out_threshold == nullptr ||
        out_candidate_count == nullptr || out_component_count == nullptr || out_status == nullptr ||
        candidate_capacity < 0 ||
        (candidate_capacity > 0 && (out_candidate_ints == nullptr || out_candidate_floats == nullptr)) ||
        int_param_count != CAND_INT_PARAMS || real_param_count != CAND_REAL_PARAMS ||
        generation == 0 || generation != persistent->resident_generation ||
        persistent->resident_u8 == nullptr ||
        (persistent->resident_channels != 1 && persistent->resident_channels != 3) ||
        width <= 0 || height <= 0 || x < 0 || y < 0 ||
        x > persistent->resident_width - width || y > persistent->resident_height - height) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    *out_candidate_count = 0;
    *out_component_count = 0;
    *out_status = CAND_STATUS_OK;
    const int kernel_size = int_params[0];
    const int candidate_value = int_params[1];
    const int morph_operation = int_params[2];
    const int morph_kernel = int_params[3];
    const int morph_iterations = int_params[4];
    CandidateGeometry geometry{};
    geometry.width = width;
    geometry.height = height;
    geometry.center_enabled = int_params[5] != 0 ? 1 : 0;
    geometry.center_x0 = int_params[6];
    geometry.center_y0 = int_params[7];
    geometry.center_x1 = int_params[8];
    geometry.center_y1 = int_params[9];
    geometry.inset_top = int_params[10];
    geometry.inset_bottom = int_params[11];
    geometry.inset_left = int_params[12];
    geometry.inset_right = int_params[13];
    const int connectivity = int_params[14];
    const int min_area = int_params[15];
    const int max_area = int_params[16];
    const int max_area_enabled = int_params[17] != 0 ? 1 : 0;
    const int border_margin = int_params[18];
    const int padding_min = int_params[19];
    const int padding_max = int_params[20];
    const int min_background_pixels = int_params[21];
    const double sigma = real_params[0];
    const double padding_scale = real_params[5];
    if (candidate_value < 0 || candidate_value > 255 ||
        morph_operation < -1 || morph_operation > VF_MORPH_ERODE) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    // Only the semantics verified against OpenCV run here; everything else stays on the host path.
    if ((morph_operation >= 0 && (morph_kernel < 3 || morph_kernel % 2 == 0 || morph_iterations < 1)) ||
        (connectivity != 4 && connectivity != 8) || std::isnan(padding_scale)) {
        return VF_CUDA_UNSUPPORTED;
    }
    const int kernel_status = gaussian_f32_kernel_request(kernel_size, sigma);
    if (kernel_status != VF_CUDA_OK) return kernel_status;
    const long long requested = static_cast<long long>(width) * static_cast<long long>(height);
    if (requested > static_cast<long long>(INT_MAX)) return VF_CUDA_UNSUPPORTED;
    const int items = static_cast<int>(requested);
    const size_t item_count = static_cast<size_t>(items);

    float residual_median = 0.0f;
    float mad = 0.0f;
    double residual_threshold = 0.0;
    int result = resident_cnr_mask_device(
        persistent, x, y, width, height, kernel_size, sigma, real_params[1], real_params[2],
        real_params[3], real_params[4], candidate_value, &residual_median, &mad, &residual_threshold);
    if (result != VF_CUDA_OK) return result;
    *out_residual_median = residual_median;
    *out_mad = mad;
    *out_threshold = residual_threshold;

    // Morphology on the candidate mask, then the exclusion AND, in the detector's order.
    unsigned char* mask = persistent->cnr_mask_mask;
    if (morph_operation >= 0) {
        result = reserve_device(
            &persistent->cand_mask_scratch, &persistent->cand_mask_scratch_capacity, item_count,
            &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
        unsigned char* source = persistent->cnr_mask_mask;
        unsigned char* target = persistent->cand_mask_scratch;
        const int radius = morph_kernel / 2;
        auto pass = [&](int dilate) {
            launch_morph_pass(source, target, width, height, 1, radius, dilate, persistent->stream);
            std::swap(source, target);
        };
        if (morph_operation == VF_MORPH_OPEN) {
            for (int i = 0; i < morph_iterations; ++i) pass(0);
            for (int i = 0; i < morph_iterations; ++i) pass(1);
        } else if (morph_operation == VF_MORPH_CLOSE) {
            for (int i = 0; i < morph_iterations; ++i) pass(1);
            for (int i = 0; i < morph_iterations; ++i) pass(0);
        } else {
            for (int i = 0; i < morph_iterations; ++i) pass(morph_operation == VF_MORPH_DILATE ? 1 : 0);
        }
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
        mask = source;
    }
    cand_exclusion_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        mask, geometry);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    // Connected components by hook-and-compress union-find until a pass changes nothing.
    result = reserve_device(
        &persistent->ccl_parent, &persistent->ccl_parent_capacity, item_count, &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_words, &persistent->cand_words_capacity, static_cast<size_t>(4),
        &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_ramp, &persistent->cand_ramp_capacity, item_count, &persistent->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &persistent->cand_foreground, &persistent->cand_foreground_capacity, item_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    const unsigned int pixel_blocks = cand_blocks(items);
    ccl_init_kernel<<<pixel_blocks, CAND_THREADS, 0, persistent->stream>>>(mask, persistent->ccl_parent, items);
    cand_ramp_kernel<<<pixel_blocks, CAND_THREADS, 0, persistent->stream>>>(persistent->cand_ramp, items);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    int32_t changed = 0;
    for (int iteration = 0;; ++iteration) {
        if (iteration >= CCL_MAX_ITERATIONS) return VF_CUDA_INTERNAL_ERROR;
        cudaError_t error = cudaMemsetAsync(persistent->cand_words, 0, sizeof(int32_t), persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        ccl_hook_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
            persistent->ccl_parent, width, height, connectivity, persistent->cand_words);
        ccl_compress_kernel<<<pixel_blocks, CAND_THREADS, 0, persistent->stream>>>(
            persistent->ccl_parent, items);
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
        result = cand_read_word(persistent, &changed, persistent->cand_words);
        if (result != VF_CUDA_OK) return result;
        if (changed == 0) break;
    }

    int component_count = 0;
    int kept_count = 0;
    result = cand_group_components(
        persistent, mask, items, width, height, min_area, max_area, max_area_enabled, border_margin,
        &component_count, &kept_count);
    if (result != VF_CUDA_OK) return result;
    *out_component_count = component_count;
    *out_candidate_count = kept_count;
    if (kept_count > candidate_capacity) {
        *out_status = CAND_STATUS_CAPACITY;
        return VF_CUDA_UNSUPPORTED;
    }

    if (kept_count > 0) {
        const size_t kept = static_cast<size_t>(kept_count);
        result = reserve_device(&persistent->cand_windows, &persistent->cand_windows_capacity, kept * 4,
                                &persistent->allocation_count);
        if (result == VF_CUDA_OK) result = reserve_device(
            &persistent->cand_window_sizes, &persistent->cand_window_sizes_capacity, kept,
            &persistent->allocation_count);
        if (result == VF_CUDA_OK) result = reserve_device(
            &persistent->cand_gather_offsets, &persistent->cand_gather_offsets_capacity, kept,
            &persistent->allocation_count);
        if (result == VF_CUDA_OK) result = reserve_device(
            &persistent->cand_out_ints, &persistent->cand_out_ints_capacity, kept * CAND_RECORD_INTS,
            &persistent->allocation_count);
        if (result == VF_CUDA_OK) result = reserve_device(
            &persistent->cand_out_floats, &persistent->cand_out_floats_capacity, kept * CAND_RECORD_FLOATS,
            &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
        cand_window_kernel<<<cand_blocks(kept_count), CAND_THREADS, 0, persistent->stream>>>(
            persistent->cand_kept, persistent->cand_boxes, kept_count, width, height, padding_min, padding_max,
            padding_scale, persistent->cand_windows, persistent->cand_window_sizes);
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
        size_t cub_bytes = 0;
        cudaError_t error = cub::DeviceScan::ExclusiveSum(
            nullptr, cub_bytes, persistent->cand_window_sizes, persistent->cand_gather_offsets, kept_count,
            persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        result = cand_reserve_cub(persistent, cub_bytes);
        if (result != VF_CUDA_OK) return result;
        error = cub::DeviceScan::ExclusiveSum(
            persistent->cand_cub_scratch, cub_bytes, persistent->cand_window_sizes,
            persistent->cand_gather_offsets, kept_count, persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        long long last_offset = 0;
        long long last_size = 0;
        error = cudaMemcpyAsync(
            &last_offset, persistent->cand_gather_offsets + (kept - 1), sizeof(long long),
            cudaMemcpyDeviceToHost, persistent->stream);
        if (error == cudaSuccess) error = cudaMemcpyAsync(
            &last_size, persistent->cand_window_sizes + (kept - 1), sizeof(long long),
            cudaMemcpyDeviceToHost, persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        result = visionflow_cuda::stream_result(persistent->stream);
        if (result != VF_CUDA_OK) return result;
        const long long gather_total = last_offset + last_size;
        if (gather_total > CAND_MAX_GATHER) {
            *out_status = CAND_STATUS_GATHER_LIMIT;
            return VF_CUDA_UNSUPPORTED;
        }
        result = cand_ring_statistics(persistent, kept_count, gather_total, geometry, min_background_pixels);
        if (result != VF_CUDA_OK) return result;
        record_timing_event(persistent, TIMING_AFTER_KERNEL);
        error = cudaMemcpyAsync(
            out_candidate_ints, persistent->cand_out_ints, sizeof(int32_t) * kept * CAND_RECORD_INTS,
            cudaMemcpyDeviceToHost, persistent->stream);
        if (error == cudaSuccess) error = cudaMemcpyAsync(
            out_candidate_floats, persistent->cand_out_floats, sizeof(float) * kept * CAND_RECORD_FLOATS,
            cudaMemcpyDeviceToHost, persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
    } else {
        record_timing_event(persistent, TIMING_AFTER_KERNEL);
    }
    record_timing_event(persistent, TIMING_AFTER_OUTPUT);
    result = visionflow_cuda::stream_result(persistent->stream);
    if (result != VF_CUDA_OK) return result;
    finalize_timing(persistent);
    for (int candidate = 0; candidate < kept_count; ++candidate) {
        if (out_candidate_ints[static_cast<size_t>(candidate) * CAND_RECORD_INTS + 6] != CAND_STATUS_OK) {
            *out_status = CAND_STATUS_GLOBAL_BACKGROUND;
            return VF_CUDA_UNSUPPORTED;
        }
    }
    return VF_CUDA_OK;
}
