#ifndef VISIONFLOW_CUDA_H
#define VISIONFLOW_CUDA_H

#include <stdint.h>
#include "visionflow_cuda_errors.h"

#define VF_CUDA_ABI_VERSION 1
#define VF_CUDA_PLAN_VERSION 1
#define VF_PLAN_INPUT_NODE (-1)

/*
 * ABI rules:
 * - All image pointers are host pointers to uint8 interleaved data.
 * - Strides are byte counts, not pixel counts.
 * - The caller owns every input/output buffer and must allocate the output.
 * - Calls are synchronous: output is ready when the function returns.
 * - A return value of VF_CUDA_OK means success; other values are declared in
 *   visionflow_cuda_errors.h and can be described by vf_gpu_error_message().
 * - The Python bridge serializes calls sharing one GpuRuntime. Native callers
 *   should also serialize calls unless they provide their own higher-level
 *   synchronization.
 * - Context APIs are additive ABI v1 extensions. Callers may probe their
 *   exports and keep using the stateless primitive APIs with an older DLL.
 * - A context owns reusable device buffers and must be destroyed by the same
 *   module with vf_context_destroy(). It is not safe for concurrent calls.
 */

#if defined(_WIN32)
#  if defined(VISIONFLOW_CUDA_EXPORTS)
#    define VF_CUDA_API __declspec(dllexport)
#  else
#    define VF_CUDA_API __declspec(dllimport)
#  endif
#else
#  define VF_CUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum VisionFlowMorphologyOperation {
    VF_MORPH_OPEN = 0,
    VF_MORPH_CLOSE = 1,
    VF_MORPH_DILATE = 2,
    VF_MORPH_ERODE = 3
};

enum VisionFlowPlanOperatorKind {
    VF_PLAN_GRAY = 1,
    VF_PLAN_GAUSSIAN = 2,
    VF_PLAN_THRESHOLD = 3,
    VF_PLAN_ADAPTIVE_MEAN = 4,
    VF_PLAN_MORPHOLOGY = 5,
    VF_PLAN_RESIZE_AREA = 6
};

typedef struct VfPlanOperatorV1 {
    uint32_t struct_size;
    int32_t kind;
    int32_t input_node;
    int32_t output_node;
    int32_t int_params[4];
    float float_params[2];
} VfPlanOperatorV1;

typedef struct VfPlanDescV1 {
    uint32_t struct_size;
    uint32_t version;
    int32_t input_channels;
    int32_t operator_count;
    const VfPlanOperatorV1* operators;
    int32_t output_node;
} VfPlanDescV1;

typedef struct VfDagPlanDescV1 {
    uint32_t struct_size;
    uint32_t version;
    int32_t input_channels;
    int32_t operator_count;
    const VfPlanOperatorV1* operators;
    int32_t output_count;
    const int32_t* output_nodes;
} VfDagPlanDescV1;

typedef struct VfDagOutputV1 {
    uint32_t struct_size;
    int32_t node;
    uint8_t* data;
    int32_t stride;
    int32_t channels;
} VfDagOutputV1;

typedef struct VfRoiV1 {
    uint32_t struct_size;
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
} VfRoiV1;

typedef struct VfCudaTimingsV1 {
    uint32_t struct_size;
    uint32_t version;
    float context_create_ms;
    float allocation_ms;
    float h2d_ms;
    float device_copy_ms;
    float kernel_ms;
    float d2h_ms;
    float synchronize_ms;
    float free_ms;
    float gaussian_ms;
    float adaptive_integral_ms;
    float threshold_ms;
    float morphology_ms;
    float total_device_ms;
} VfCudaTimingsV1;

/* Optional detailed accounting for context-owned cudaMalloc allocations.
 * Driver/context overhead and allocations owned by other libraries/processes are intentionally
 * excluded; compare reserved_bytes with process-level VRAM diagnostics when investigating gaps.
 */
typedef struct VfCudaContextMemoryStatsV1 {
    uint32_t struct_size;
    uint32_t version;
    uint64_t reserved_bytes;
    uint64_t peak_reserved_bytes;
    uint64_t allocation_count;
    uint64_t plan_bytes;
    uint64_t resident_bytes;
    uint64_t template_match_bytes;
    uint64_t contour_bytes;
    uint64_t median_bytes;
    uint64_t gaussian_f32_bytes;
    uint64_t cnr_mask_bytes;
    uint64_t cnr_candidate_bytes;
} VfCudaContextMemoryStatsV1;

VF_CUDA_API int vf_gpu_abi_version(void);
VF_CUDA_API int vf_gpu_device_count(void);
VF_CUDA_API int vf_gpu_compute_capability(void);
VF_CUDA_API int vf_gpu_device_name(char* output, int capacity);
VF_CUDA_API int vf_gpu_error_message(int error_code, char* output, int capacity);
VF_CUDA_API int vf_gpu_memory_info(uint64_t* free_bytes, uint64_t* total_bytes);

VF_CUDA_API int vf_context_create(void** context);
VF_CUDA_API int vf_context_destroy(void* context);
VF_CUDA_API int vf_context_stats(
    void* context, uint64_t* reserved_bytes, uint64_t* allocation_count);
VF_CUDA_API int vf_context_memory_stats_v1(
    void* context, VfCudaContextMemoryStatsV1* stats);
/* Optional diagnostics control. Disabled mode skips CUDA event recording on the hot path.
 * Older DLLs without this export retain their historical always-on timing behavior. */
VF_CUDA_API int vf_context_set_timing_enabled(void* context, int enabled);
VF_CUDA_API int vf_context_last_timings(void* context, VfCudaTimingsV1* timings);
VF_CUDA_API int vf_context_upload_u8(
    void* context,
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint64_t* generation);
/* A negative src_stride denotes bottom-up rows; src points at the logical top row. */
VF_CUDA_API int vf_context_upload_u8_file_order(
    void* context,
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint64_t* generation);
/* Page-lock caller-owned host bytes so later uploads from them skip the driver staging copy.
 * The caller keeps the memory alive while registered and unregisters it before freeing it. */
VF_CUDA_API int vf_host_register_u8(void* context, uint8_t* host, uint64_t bytes);
VF_CUDA_API int vf_host_unregister_u8(void* context, uint8_t* host);
VF_CUDA_API int vf_roi_batch_create(
    void* context, uint64_t generation,
    const VfRoiV1* rois, int roi_count, void** batch);
VF_CUDA_API int vf_roi_batch_info(
    void* batch, int* roi_count, int* width, int* height, int* channels);
VF_CUDA_API int vf_roi_batch_download_u8(
    void* batch, int roi_index,
    uint8_t* dst, int dst_stride, int dst_channels);
VF_CUDA_API int vf_roi_batch_destroy(void* batch);

/*
 * Optional generic plan ABI. The descriptor is backend-neutral and contains
 * no detector ID/name. vf_plan_create copies and validates the descriptor;
 * vf_plan_execute only transfers image data and launches the compiled plan.
 * A plan borrows its context and must be destroyed before that context.
 */
VF_CUDA_API int vf_plan_query(
    const VfPlanDescV1* desc, int width, int height,
    char* reason, int reason_capacity);
VF_CUDA_API int vf_plan_create(
    void* context, const VfPlanDescV1* desc, int width, int height, void** plan);
VF_CUDA_API int vf_plan_execute(
    void* plan,
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels);
VF_CUDA_API int vf_plan_destroy(void* plan);
VF_CUDA_API int vf_plan_execute_roi(
    void* plan, uint64_t generation, int x, int y,
    uint8_t* dst, int dst_stride, int dst_channels);

/*
 * Optional detector-neutral DAG extension. Nodes are topologically ordered,
 * may reference the root or an earlier node, and may expose multiple named-by-
 * index outputs. Execution uploads the root once and synchronizes after all
 * requested host outputs have been copied.
 */
VF_CUDA_API int vf_dag_plan_query(
    const VfDagPlanDescV1* desc, int width, int height,
    char* reason, int reason_capacity);
VF_CUDA_API int vf_dag_plan_create(
    void* context, const VfDagPlanDescV1* desc, int width, int height, void** plan);
VF_CUDA_API int vf_dag_plan_execute(
    void* plan,
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    const VfDagOutputV1* outputs, int output_count);
VF_CUDA_API int vf_dag_plan_destroy(void* plan);
VF_CUDA_API int vf_dag_plan_execute_roi(
    void* plan, uint64_t generation, int x, int y,
    const VfDagOutputV1* outputs, int output_count);

VF_CUDA_API int vf_bgr_to_gray_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels);

VF_CUDA_API int vf_bgr_to_rgb_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels);

VF_CUDA_API int vf_crop_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int crop_x, int crop_y, int crop_width, int crop_height);

VF_CUDA_API int vf_resize_gray_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int dst_width, int dst_height);

VF_CUDA_API int vf_gaussian_blur_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int kernel_size);

VF_CUDA_API int vf_threshold_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int threshold, int max_value, int invert);

VF_CUDA_API int vf_adaptive_mean_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int block_size, float c, int max_value, int invert);

VF_CUDA_API int vf_morphology_rect_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int operation, int kernel_size, int iterations);

/*
 * Optional Template Anchor Grid localization extension. Reads a search ROI out of the
 * context's resident image, converts it to gray with the same weights as
 * vf_bgr_to_gray_u8, and returns the best TM_CCOEFF_NORMED match of a host gray template as
 * a half-open rectangle plus its score. Only the result rectangle and score cross PCIe; the
 * template is uploaded once per call (it is small and constant per Recipe).
 *
 * Semantics match the CPU reference in core/tiler.py::Tiler._find_grid_anchor:
 *   score = sum((window - mean(window)) * (templ - mean(templ)))
 *           / sqrt(sum((window - mean(window))^2) * sum((templ - mean(templ))^2))
 * Ties (equal score) resolve to the topmost, then leftmost match. The template must not be
 * flat (standard deviation <= 1e-6); the CPU reference switches to TM_SQDIFF_NORMED there,
 * which is reported as VF_CUDA_UNSUPPORTED so the caller can restart the step on the CPU.
 * out_match must hold 4 int32 values [x, y, width, height] and out_score 1 float.
 */
VF_CUDA_API int vf_match_template_gray_u8(
    void* context,
    uint64_t generation,
    int search_x, int search_y, int search_width, int search_height,
    const uint8_t* templ, int template_width, int template_height,
    int* out_match, float* out_score);

/*
 * Debug helper for the localization extension: after vf_match_template_gray_u8 has run, copies
 * the packed winning key back so a caller can inspect the raw candidate comparison. Not used by
 * production code paths.
 */
VF_CUDA_API int vf_match_template_debug_key(void* context, unsigned long long* out_key);

/* Diagnostics: copy one prefix plane of the localization scratch back to the host. */
VF_CUDA_API int vf_match_template_debug_planes(
    void* context, int plane, int64_t* out_values, size_t count);

/* Diagnostics: copy the gray search ROI that the localization step computed back to the host. */
VF_CUDA_API int vf_match_template_debug_roi(void* context, uint8_t* out_values, size_t count);

/* Diagnostics: copy the per-column candidate scores and rows of the last localization call. */
VF_CUDA_API int vf_match_template_debug_candidates(
    void* context, double* out_scores, int* out_rows, int count, int* candidate_slots);

/*
 * Optional contour extension: cv2.findContours(binary, mode, CHAIN_APPROX_SIMPLE) equivalence.
 *
 * Input path. The operator reads `width` x `height` pixels at (x, y) of the context's *resident*
 * image and treats every non-zero pixel as foreground, exactly like OpenCV binarizes the padded
 * image with THRESH_BINARY. The caller uploads the binary mask itself with vf_context_upload_u8
 * as a single-channel image (1 byte per pixel); the colour image is never uploaded for this step.
 * A resident colour image is reported as VF_CUDA_UNSUPPORTED because reinterpreting it as a mask
 * would silently change the foreground rule.
 *
 * The uploaded region is treated as an isolated image: a one-pixel zero frame is added on the
 * device, so pixels outside the region never influence the result. That is what makes a
 * sub-region call identical to cv2.findContours on the same sub-array.
 *
 * Semantics. Port of tools/contour_reference.py, which is verified point-for-point against
 * cv2.findContours (Suzuki-Abe border following, icvFetchContour with CHAIN_APPROX_SIMPLE):
 *   - VF_CONTOURS_RETR_LIST     == cv2.RETR_LIST
 *   - VF_CONTOURS_RETR_EXTERNAL == cv2.RETR_EXTERNAL
 *   - contour order: OpenCV reports the flat list in *reverse discovery order*; this operator
 *     already returns that order.
 *   - point coordinates: 0-based within the requested region (add x/y for image coordinates).
 *   - deterministic: one serialized thread performs the raster scan and the border traces, no
 *     atomics are used, and the auxiliary kernels write every output element from exactly one
 *     thread. The same resident mask and region therefore always produce the same bytes.
 *
 * Output layout (two calls, mirroring vf_roi_batch_create / vf_roi_batch_download_u8):
 *   vf_find_contours_u8() runs the trace into context-owned device scratch and reports how many
 *     contours and how many points the result holds.
 *   vf_find_contours_download() copies that result to host buffers:
 *     out_offsets: int32[contour_count + 1]; contour j owns point indices
 *                  [out_offsets[j], out_offsets[j + 1]). Use those as *point* indices into
 *                  out_points, which holds point pairs.
 *     out_points:  int32[2 * point_count] as (x, y) pairs; point_capacity counts pairs.
 *   offset_capacity must be at least contour_count + 1 and point_capacity at least point_count;
 *   a short buffer is rejected with VF_CUDA_INVALID_ARGUMENT instead of being filled partially.
 *   The scratch keeps the most recent call only, and a download is rejected when the resident
 *   image changed (generation) after the trace ran.
 */
enum VisionFlowContourMode {
    VF_CONTOURS_RETR_EXTERNAL = 0,
    VF_CONTOURS_RETR_LIST = 1
};

VF_CUDA_API int vf_find_contours_u8(
    void* context,
    uint64_t generation,
    int x, int y, int width, int height, int mode,
    int* out_contour_count, int* out_point_count);

VF_CUDA_API int vf_find_contours_download(
    void* context,
    int32_t* out_offsets, int offset_capacity,
    int32_t* out_points, int point_capacity);

VF_CUDA_API int vf_preprocess_401_2_u8(
    void* context,
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride,
    int gaussian_kernel_size,
    int adaptive_block_size, float adaptive_c,
    int max_value, int invert);

/*
 * Optional exact-median extension: the bit-exact NumPy result of ``np.median`` for float32 input.
 *
 * The detector that motivates this export spends most of its time in ``np.median(residual)`` and in
 * the median absolute deviation around it, so the median is the highest-value device step there.
 *
 * Semantics. Every float32 is mapped to a monotone-orderable uint32 key (positives flip the sign
 * bit, negatives invert every bit), so unsigned integer order is exactly float order, including
 * negatives, zeros, subnormals and infinities. The keys are radix-sorted on the device with
 * ``cub::DeviceRadixSort``; only the one or two middle keys cross PCIe, and the float32 average for
 * an even count (add, then divide by two, both in float32 -- NumPy averages in the input dtype)
 * is computed on the host. No device-side floating-point arithmetic is involved, so the result
 * cannot drift from the reference for any finite or infinite input.
 *
 * ``values`` is a host float32 array of ``count`` elements and is only read, never written.
 * ``count`` must be positive and must fit in a signed 32-bit integer (the radix-sort offset type).
 * A NaN anywhere in ``values`` returns NaN, mirroring NumPy's ``_median_nancheck``; note that
 * ``NaN == NaN`` is false, so callers comparing against ``np.median`` must use a NaN-aware test
 * for that case only.
 *
 * The operation is deterministic: identical input bytes always produce identical output bytes.
 * Deviceless callers that want to restart the step on the CPU reference should probe this export
 * and fall back when it is missing.
 */
VF_CUDA_API int vf_median_f32(
    void* context,
    const float* values,
    long long count,
    float* out_median);

/*
 * Optional float32 Gaussian extension: the separable blur the 202-CS-SN-1 detector builds its
 * background with, moved off the host so the residual can stay on the device.
 *
 * Semantics. Single-channel float32, separable horizontal-then-vertical convolution with
 * reflect101 borders and float32 accumulation, using the same coefficients OpenCV's
 * cv2.getGaussianKernel(ksize, sigma, CV_32F) produces: `sigma > 0` is used directly for both
 * axes (cv2.GaussianBlur defaults sigma2 to sigma1), while `sigma <= 0` selects OpenCV's automatic
 * rule 0.3*((ksize-1)*0.5-1)+0.8, including its fixed small-kernel table for odd ksize <= 9
 * (SMALL_GAUSSIAN_SIZE is 9 in OpenCV 5.x). `sigma` is a double so the caller's value is never
 * narrowed on the way in; a NaN or infinite sigma has no OpenCV rule to reproduce and is refused
 * with VF_CUDA_INVALID_ARGUMENT instead of being silently treated as the automatic sigma.
 * Coefficients were checked bit-for-bit against cv2.getGaussianKernel for every supported size and
 * for sigma 0.0/0.5/1.0/1.25/2.5/5.0, so the difference against
 * cv2.GaussianBlur(src, (ksize, ksize), sigma) comes only from the summation order inside OpenCV's
 * SIMD filter:
 *
 *   Verified tolerance (tools/gaussian_f32_equivalence.py,
 *   outputs_validation/cnr_profile/gaussian_f32_equivalence.txt):
 *     - input values in [0, 255] (the detector's gray/residual domain), any supported ksize,
 *       1008-case sigma sweep over {0.0, 0.5, 1.0, 1.25, 2.5, 5.0}:
 *       max |device - cv2| <= 4.0e-4 (measured 9.2e-5 with sigma <= 0, 7.6e-5 with sigma > 0),
 *       mean <= 5.0e-5 (measured 1.1e-5 with sigma <= 0, 3.1e-5 with sigma > 0).
 *     - the error scales with the magnitude of the source, not with the image size: it stayed
 *       below 5.0e-7 of the source range (measured 3.5e-7 on a +-1000 float32 source).
 *   The 202-CS-SN-1 result was then compared against the cv2.GaussianBlur reference on 47 scenes /
 *   78 defects (tools/gaussian_202_matrix.py,
 *   outputs_validation/cnr_profile/gaussian_202_final_output_matrix.txt), including eight scenes
 *   that set the detector's gaussian_sigma recipe parameter to 0.5..5.0: PASS/NG, defect count,
 *   every defect's bbox, area and confidence, the candidate mask and every other metadata field
 *   were identical in all of them. The only fields that moved were the residual-derived diagnostic
 *   floats reported inside the defect metadata - residual_median <= 1.5e-5, mad <= 7.6e-6,
 *   robust_noise_sigma <= 1.1e-5, residual_threshold <= 3.4e-5 absolute - which is the tolerance
 *   above appearing in the reported numbers, not in any detection decision.
 *
 * Supported kernel sizes. Exactly the odd sizes in [3, 127] - all 63 of them are covered by the
 * tolerance evidence above. Anything else (even, below 3, or above 127) is rejected: sizes below 3
 * with VF_CUDA_INVALID_ARGUMENT, other unverified sizes with VF_CUDA_UNSUPPORTED, so a caller can
 * restart that step on the CPU reference instead of receiving an unvalidated result.
 *
 * Strides are byte counts. `src` and `dst` are host pointers to single-channel float32 images and
 * both must be at least `width * 4` bytes per row. The result is deterministic: identical input
 * bytes and kernel size always produce identical output bytes.
 *
 * The scratch buffers (uploaded source rectangle, horizontal intermediate, packed result) are owned
 * by the context, grow-only, and separate from the plan scratch, so a call never disturbs a
 * compiled plan. Calls are serialized like every other context export.
 */
VF_CUDA_API int vf_gaussian_blur_f32(
    void* context,
    const float* src, int width, int height, int src_stride,
    float* dst, int dst_stride,
    int kernel_size, double sigma);

/*
 * Optional rectangle variant of the float32 Gaussian: the same operator restricted to
 * `width` x `height` pixels at (x, y) of a wider `src_width` x `src_height` host image.
 *
 * The rectangle is treated as an isolated image: borders reflect inside it and pixels outside it
 * are never read, so the result equals cv2.GaussianBlur(src[y:y+height, x:x+width], (ksize, ksize),
 * sigma) and the caller can blur a sub-window without uploading the whole plane. `src_stride` is
 * the byte stride of the full source, `dst_stride` the byte stride of the `width` x `height`
 * single-channel float32 result. Validation, sigma semantics, tolerance and ksize rules are
 * identical to vf_gaussian_blur_f32.
 */
VF_CUDA_API int vf_gaussian_blur_f32_roi(
    void* context,
    const float* src, int src_width, int src_height, int src_stride,
    int x, int y, int width, int height,
    float* dst, int dst_stride,
    int kernel_size, double sigma);

/*
 * Optional residual-threshold extension: the "automatic CNR mask" of 202-CS-SN-1
 * (detectors/detector_202_1.py::_automatic_cnr_mask) reduced to the two device medians, the
 * threshold and the candidate mask in one call, so the residual and its absolute deviation never
 * cross PCIe.
 *
 * Motivation. The median export above is called twice per ROI with a *derived* operand each time:
 * once with the residual and once with |residual - median|. For a 2000 x 12000 ROI that is two
 * 96 MiB host-to-device uploads of arrays that only exist to feed the sort. This export uploads the
 * operands the caller already has (the float32 image and the float32 background) and builds every
 * derived array on the device:
 *
 *   residual[i] = image[i] - background[i]                      (float32 subtraction)
 *   residual_median = np.median(residual)                       (bit-exact, see vf_median_f32)
 *   absdev[i]   = |residual[i] - residual_median|               (float32 fabsf = np.abs sign clear)
 *   mad         = np.median(absdev)                             (bit-exact)
 *   robust      = max(mad_scale * mad, absolute_floor)          (Python max, in double)
 *   threshold   = max(threshold_floor, sigma_multiplier * robust)  (Python max, in double)
 *   out_mask[i] = (absdev[i] > threshold) ? candidate_value : 0    (strict >, in float32)
 *
 * Bit-exactness. The medians reuse the key/sort machinery of vf_median_f32 unchanged (monotone
 * float32 -> uint32 order keys, cub::DeviceRadixSort, middle-key-only readback, host float32
 * even-count average, quiet NaN when any input is a NaN), and the mask subtracts that decoded
 * float32 value, never an approximation through the order keys. The mask comparison happens in
 * float32 because NumPy compares a float32 array against a Python float with the scalar narrowed to
 * the array dtype (NEP 50 weak scalar), which is also why `threshold` is computed in double and only
 * then narrowed for the comparison: the detector holds mad as a Python float, so
 * `mad_scale * mad`, `max(...)` and `sigma_multiplier * robust` are double operations there.
 * Python's two-argument max is reproduced exactly (the first argument wins unless the second is
 * strictly greater, so a NaN first argument propagates and a NaN second argument is ignored).
 *
 * Parameters. `image` and `background` are host pointers to single-channel row-major float32 planes
 * of `width` x `height` pixels; `image_stride`/`background_stride` are byte strides and must be at
 * least `width * 4`. Both planes are uploaded once each with a 2D copy and are only read. Passing
 * the same array for both produces an exactly zero residual. `width * height` must not exceed
 * INT_MAX (the radix-sort offset type); a larger request is refused with VF_CUDA_UNSUPPORTED so the
 * caller can restart the step on the CPU reference.
 *
 * `sigma_multiplier`, `threshold_floor`, `absolute_floor` and `mad_scale` are doubles so the
 * caller's values are never narrowed on the way in - the threshold arithmetic is a double
 * computation in the reference. `candidate_value` is an int validated to 0..255 and is refused with
 * VF_CUDA_INVALID_ARGUMENT otherwise (a narrower type would silently wrap).
 *
 * Outputs. `out_residual_median` and `out_mad` receive the two decoded float32 medians (quiet NaN if
 * that operand held a NaN; compare NaN-aware because NaN != NaN). `out_threshold` receives the
 * double threshold before it is narrowed for the comparison, so it is bit-identical to the
 * detector's `residual_threshold`. `out_mask` receives `width` x `height` bytes, tightly packed, and
 * `out_mask_capacity` must be at least `width * height`; a short buffer is refused with
 * VF_CUDA_INVALID_ARGUMENT instead of being filled partially.
 *
 * Traffic for a `width` x `height` plane (N = width * height): host-to-device 2 * N * 4 bytes
 * (image + background) and device-to-host N bytes (the mask plane; the two medians and the threshold
 * are decoded on the host and written straight into the caller's scalars, without a further copy).
 * The two vf_median_f32 calls this replaces also move 2 * N * 4 bytes per plane, because they upload
 * the derived residual and its absolute deviation instead, so the isolated median stage's upload
 * volume is unchanged: what this export removes is the two N-element host temporaries, the host
 * arithmetic that fills them, and the surrounding stage's extra traffic. The measured numbers that
 * quantify all three stages - this export, the isolated two-median baseline and the current detector
 * stage - are recorded in outputs_validation/cnr_profile/cnr_mask_f32_equivalence.txt.
 *
 * Determinism and failure. Identical input bytes always produce identical output bytes; the two
 * sorts and the three elementwise passes are deterministic. Null pointers, a non-positive width or
 * height, a stride below `width * 4`, a `candidate_value` outside 0..255 and a short mask buffer are
 * refused with VF_CUDA_INVALID_ARGUMENT before any work is launched. The scratch buffers (uploaded
 * image and background, residual, absolute deviation, device mask) are owned by the context,
 * grow-only, and separate from the plan, median and Gaussian scratch.
 */
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
    long long out_mask_capacity);

/*
 * Resident-image variant of the 202 automatic CNR operation. It reads one isolated ROI from the
 * image previously uploaded by vf_context_upload_u8, converts BGR to the exact OpenCV uint8 gray
 * value (or reads a one-channel resident image), promotes that value to float32, runs the verified
 * float32 Gaussian, residual median/MAD and threshold pipeline, and downloads only the mask and
 * three scalar diagnostics. No ROI image or Gaussian background crosses PCIe.
 *
 * `generation` must identify the current resident image. The ROI must be in bounds, the resident
 * image must have one or three channels, and all remaining parameter/output contracts are the same
 * as vf_gaussian_blur_f32 plus vf_cnr_mask_f32. This is an additive ABI-v1 export; callers must
 * probe for it and fall back to the host-operand exports when loading an older DLL.
 */
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
    long long out_mask_capacity);

/*
 * Resident-image 202-CS-SN-1 candidate extraction. Runs the vf_cnr_mask_u8_roi chain on one ROI of
 * the current resident image and then keeps the rest of Detector202_1's candidate stage on the
 * device: the rectangular morphology pass, the center/edge exclusion AND, connected components with
 * stats, the area and border-margin filters, and the ring CNR statistics. Neither the gray image,
 * the mask nor a label map crosses PCIe; only the three residual scalars and one record per surviving
 * component are copied back.
 *
 * int_params (exactly 22): [0] Gaussian kernel size, [1] candidate value 0..255, [2] morphology
 * operation (-1 none, otherwise VfMorphologyOperation), [3] odd kernel size >= 3, [4] iterations >= 1,
 * [5] center exclusion enabled, [6..9] center x0, y0, x1, y1 (half-open, already clamped),
 * [10..13] edge insets top, bottom, left, right, [14] connectivity 4 or 8, [15] minimum area,
 * [16] maximum area, [17] maximum area enabled, [18] border margin, [19] padding minimum,
 * [20] padding maximum, [21] minimum background pixels.
 * real_params (exactly 6): [0] Gaussian sigma, [1] sigma multiplier, [2] threshold floor,
 * [3] absolute floor, [4] MAD scale, [5] padding scale.
 *
 * Each surviving component writes 7 int32 values to out_candidate_ints (x, y, width, height, area,
 * background pixel count, status) and 3 float32 values to out_candidate_floats (defect mean,
 * background mean, background std), ordered by the component's first raster pixel. The means and
 * standard deviation reproduce np.mean/np.std on the host's boolean-mask gathers bit-exactly: values
 * are visited in raster order, summed with NumPy's float32 pairwise order, divided in float64 and
 * narrowed to float32. The caller computes contrast, CNR and the final ordering.
 *
 * out_status and VF_CUDA_UNSUPPORTED report cases the host must handle on its existing path:
 * 1 = a component's ring has fewer than minimum background pixels (the host then uses the whole
 * included image), 2 = more surviving components than candidate_capacity (out_candidate_count holds
 * the required capacity), 3 = the ring windows exceed the device gather limit. An even kernel,
 * non-positive iterations, a connectivity other than 4 or 8, or a NaN padding scale is also
 * VF_CUDA_UNSUPPORTED. This is an additive ABI-v1 export; callers must probe for it.
 */
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
    int* out_status);

#ifdef __cplusplus
}
#endif

#endif
