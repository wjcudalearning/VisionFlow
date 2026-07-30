#define VISIONFLOW_CUDA_EXPORTS
#include "visionflow_cuda.h"
#include "visionflow_cuda_internal.cuh"
#include <algorithm>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstring>
#include <new>
#include <utility>
#include <vector>

namespace {
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;
constexpr int SCAN_THREADS = 256;
constexpr int TRANSPOSE_TILE = 32;
constexpr int TRANSPOSE_ROWS = 8;
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

struct PersistentContext {
    uint8_t* u8[5]{};
    size_t u8_capacity[5]{};
    uint32_t* gaussian_buffer = nullptr;
    size_t gaussian_capacity = 0;
    unsigned long long* u64[2]{};
    size_t u64_capacity[2]{};
    std::vector<uint8_t*> dag_u8;
    std::vector<size_t> dag_u8_capacity;
    uint8_t* resident_u8 = nullptr;
    size_t resident_capacity = 0;
    int resident_width = 0;
    int resident_height = 0;
    int resident_channels = 0;
    uint64_t resident_generation = 0;
    unsigned long long allocation_count = 0;
    cudaStream_t stream = nullptr;
    cudaError_t initialization_error = cudaSuccess;
    cudaEvent_t timing_events[TIMING_EVENT_COUNT]{};
    VfCudaTimingsV1 last_timings{};
    float pending_allocation_ms = 0.0f;
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
        for (void* pointer : u64) visionflow_cuda::free_device(pointer);
        for (void* pointer : dag_u8) visionflow_cuda::free_device(pointer);
        visionflow_cuda::free_device(resident_u8);
        for (cudaEvent_t event : timing_events) {
            if (event != nullptr) cudaEventDestroy(event);
        }
        if (stream != nullptr) cudaStreamDestroy(stream);
    }
};

float elapsed_host_ms(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - started).count();
}

void reset_timing(PersistentContext* context, bool input_is_host) {
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
    cudaEventRecord(context->timing_events[TIMING_START], context->stream);
}

void finalize_timing(PersistentContext* context) {
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

struct NativePlan {
    PersistentContext* context = nullptr;
    int width = 0;
    int height = 0;
    int output_width = 0;
    int output_height = 0;
    int input_channels = 0;
    int output_channels = 0;
    std::vector<VfPlanOperatorV1> operators;
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

int prepare_gaussian_weights(int kernel, int* radius_out) {
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
    cudaError_t error = cudaMemcpyToSymbol(
        gaussian_weights, weights.data(), static_cast<size_t>(kernel) * sizeof(uint16_t));
    if (error != cudaSuccess) return visionflow_cuda::runtime_error(error);
    *radius_out = radius;
    return VF_CUDA_OK;
}

int adaptive_layout(
    int width,
    int height,
    int block,
    int* radius_out,
    int* padded_width_out,
    int* padded_height_out,
    size_t* padded_count_out) {
    if (width <= 0 || height <= 0 || block < 3 || block % 2 == 0 || radius_out == nullptr ||
        padded_width_out == nullptr || padded_height_out == nullptr || padded_count_out == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    int radius = block / 2;
    if (radius > (INT_MAX - width) / 2 || radius > (INT_MAX - height) / 2) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    int padded_width = width + radius * 2;
    int padded_height = height + radius * 2;
    if (static_cast<size_t>(padded_width) > SIZE_MAX / static_cast<size_t>(padded_height)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    size_t padded_count = static_cast<size_t>(padded_width) * static_cast<size_t>(padded_height);
    if (padded_count > SIZE_MAX / sizeof(unsigned long long)) return VF_CUDA_INVALID_ARGUMENT;
    *radius_out = radius;
    *padded_width_out = padded_width;
    *padded_height_out = padded_height;
    *padded_count_out = padded_count;
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
                int radius = 0, padded_width = 0, padded_height = 0;
                size_t padded_count = 0;
                if (channels != 1 || op.int_params[1] < 0 || op.int_params[1] > 255 ||
                    (op.int_params[2] != 0 && op.int_params[2] != 1) ||
                    !std::isfinite(op.float_params[0]) ||
                    adaptive_layout(current_width, current_height, op.int_params[0], &radius, &padded_width,
                                    &padded_height, &padded_count) != VF_CUDA_OK) {
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
                int radius = 0, padded_width = 0, padded_height = 0;
                size_t padded_count = 0;
                if (input_channels != 1 || op.int_params[1] < 0 || op.int_params[1] > 255 ||
                    (op.int_params[2] != 0 && op.int_params[2] != 1) ||
                    !std::isfinite(op.float_params[0]) ||
                    adaptive_layout(width, height, op.int_params[0], &radius, &padded_width,
                                    &padded_height, &padded_count) != VF_CUDA_OK) {
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
    bool needs_gaussian = false;
    bool needs_morph_scratch = false;
    size_t maximum_padded_count = 0;
    for (size_t index = 0; index < plan.operators.size(); ++index) {
        int result = reserve_device(
            &context->dag_u8[index], &context->dag_u8_capacity[index],
            pixels * static_cast<size_t>(plan.node_channels[index]), &context->allocation_count);
        if (result != VF_CUDA_OK) return result;
        const VfPlanOperatorV1& op = plan.operators[index];
        needs_gaussian = needs_gaussian || op.kind == VF_PLAN_GAUSSIAN;
        needs_morph_scratch = needs_morph_scratch || op.kind == VF_PLAN_MORPHOLOGY;
        if (op.kind == VF_PLAN_ADAPTIVE_MEAN) {
            int radius = 0, padded_width = 0, padded_height = 0;
            size_t padded_count = 0;
            result = adaptive_layout(plan.width, plan.height, op.int_params[0], &radius,
                                     &padded_width, &padded_height, &padded_count);
            if (result != VF_CUDA_OK) return result;
            maximum_padded_count = std::max(maximum_padded_count, padded_count);
        }
    }
    int result = reserve_device(&context->u8[0], &context->u8_capacity[0],
                                pixels * static_cast<size_t>(plan.input_channels),
                                &context->allocation_count);
    if (result == VF_CUDA_OK && needs_morph_scratch) result = reserve_device(
        &context->u8[4], &context->u8_capacity[4], pixels * 3, &context->allocation_count);
    if (result == VF_CUDA_OK && needs_gaussian) result = reserve_device(
        &context->gaussian_buffer, &context->gaussian_capacity,
        pixels * 3, &context->allocation_count);
    if (result == VF_CUDA_OK && maximum_padded_count > 0) result = reserve_device(
        &context->u8[3], &context->u8_capacity[3], maximum_padded_count, &context->allocation_count);
    if (result == VF_CUDA_OK && maximum_padded_count > 0) result = reserve_device(
        &context->u64[0], &context->u64_capacity[0], maximum_padded_count, &context->allocation_count);
    if (result == VF_CUDA_OK && maximum_padded_count > 0) result = reserve_device(
        &context->u64[1], &context->u64_capacity[1], maximum_padded_count, &context->allocation_count);
    return result;
}

int reserve_plan_buffers(PersistentContext* context, const NativePlan& plan) {
    if (context == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    const size_t input_pixels = static_cast<size_t>(plan.width) * plan.height;
    size_t maximum_pixels = input_pixels;
    int maximum_channels = plan.input_channels;
    bool needs_gaussian = false;
    bool needs_morph_scratch = false;
    size_t maximum_padded_count = 0;
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
        needs_gaussian = needs_gaussian || op.kind == VF_PLAN_GAUSSIAN;
        needs_morph_scratch = needs_morph_scratch || op.kind == VF_PLAN_MORPHOLOGY;
        if (op.kind == VF_PLAN_ADAPTIVE_MEAN) {
            int radius = 0, padded_width = 0, padded_height = 0;
            size_t padded_count = 0;
            int result = adaptive_layout(current_width, current_height, op.int_params[0], &radius,
                                         &padded_width, &padded_height, &padded_count);
            if (result != VF_CUDA_OK) return result;
            maximum_padded_count = std::max(maximum_padded_count, padded_count);
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
    if (result == VF_CUDA_OK && needs_gaussian) result = reserve_device(
        &context->gaussian_buffer, &context->gaussian_capacity,
        maximum_pixels * static_cast<size_t>(maximum_channels), &context->allocation_count);
    if (result == VF_CUDA_OK && maximum_padded_count > 0) result = reserve_device(
        &context->u8[3], &context->u8_capacity[3], maximum_padded_count,
        &context->allocation_count);
    if (result == VF_CUDA_OK && maximum_padded_count > 0) result = reserve_device(
        &context->u64[0], &context->u64_capacity[0], maximum_padded_count,
        &context->allocation_count);
    if (result == VF_CUDA_OK && maximum_padded_count > 0) result = reserve_device(
        &context->u64[1], &context->u64_capacity[1], maximum_padded_count,
        &context->allocation_count);
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

__global__ void resize_gray_kernel(const uint8_t* src, uint8_t* dst, int sw, int sh, int dw, int dh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    if (dw <= sw && dh <= sh) {
        float scale_x = static_cast<float>(sw) / dw;
        float scale_y = static_cast<float>(sh) / dh;
        float source_x0 = x * scale_x;
        float source_x1 = (x + 1) * scale_x;
        float source_y0 = y * scale_y;
        float source_y1 = (y + 1) * scale_y;
        int start_x = static_cast<int>(floorf(source_x0));
        int end_x = static_cast<int>(ceilf(source_x1));
        int start_y = static_cast<int>(floorf(source_y0));
        int end_y = static_cast<int>(ceilf(source_y1));
        float sum = 0.0f;
        for (int source_y = start_y; source_y < end_y; ++source_y) {
            float weight_y = fmaxf(0.0f, fminf(source_y1, source_y + 1.0f) - fmaxf(source_y0, static_cast<float>(source_y)));
            int clamped_y = max(0, min(sh - 1, source_y));
            for (int source_x = start_x; source_x < end_x; ++source_x) {
                float weight_x = fmaxf(0.0f, fminf(source_x1, source_x + 1.0f) - fmaxf(source_x0, static_cast<float>(source_x)));
                int clamped_x = max(0, min(sw - 1, source_x));
                sum += src[clamped_y * sw + clamped_x] * weight_x * weight_y;
            }
        }
        dst[y * dw + x] = static_cast<uint8_t>(sum / (scale_x * scale_y) + 0.5f);
        return;
    }
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
    for (int c = 0; c < channels; ++c) {
        uint32_t sum = 0;
        for (int kx = -radius; kx <= radius; ++kx) {
            int sx = reflect101(x + kx, width);
            sum += static_cast<uint32_t>(
                src[(y * width + sx) * channels + c]) *
                gaussian_weights[kx + radius];
        }
        intermediate[(y * width + x) * channels + c] = sum;
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
    for (int c = 0; c < channels; ++c) {
        unsigned long long sum = 0;
        for (int ky = -radius; ky <= radius; ++ky) {
            int sy = reflect101(y + ky, height);
            sum += static_cast<unsigned long long>(
                intermediate[(sy * width + x) * channels + c]) *
                gaussian_weights[ky + radius];
        }
        dst[(y * width + x) * channels + c] = static_cast<uint8_t>(
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

__global__ void replicate_border_kernel(
    const uint8_t* src,
    uint8_t* padded,
    int width,
    int height,
    int padded_width,
    int padded_height,
    int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= padded_width || y >= padded_height) return;
    int source_x = max(0, min(width - 1, x - radius));
    int source_y = max(0, min(height - 1, y - radius));
    padded[y * padded_width + x] = src[source_y * width + source_x];
}

__global__ void row_prefix_u8_kernel(
    const uint8_t* src,
    unsigned long long* prefix,
    int width,
    int height) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    if (row >= height) return;
    __shared__ unsigned long long scan[SCAN_THREADS];
    __shared__ unsigned long long carry;
    __shared__ unsigned long long chunk_carry;
    if (lane == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < width; base += SCAN_THREADS) {
        int column = base + lane;
        scan[lane] = column < width ? static_cast<unsigned long long>(src[row * width + column]) : 0ULL;
        __syncthreads();
        for (int offset = 1; offset < SCAN_THREADS; offset <<= 1) {
            unsigned long long add = lane >= offset ? scan[lane - offset] : 0ULL;
            __syncthreads();
            scan[lane] += add;
            __syncthreads();
        }
        if (lane == 0) chunk_carry = carry;
        __syncthreads();
        if (column < width) prefix[row * width + column] = scan[lane] + chunk_carry;
        __syncthreads();
        int valid = min(SCAN_THREADS, width - base);
        if (lane == 0) carry = chunk_carry + scan[valid - 1];
        __syncthreads();
    }
}

__global__ void transpose_u64_kernel(
    const unsigned long long* src,
    unsigned long long* dst,
    int width,
    int height) {
    __shared__ unsigned long long tile[TRANSPOSE_TILE][TRANSPOSE_TILE + 1];
    int x = blockIdx.x * TRANSPOSE_TILE + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE + threadIdx.y;
    for (int offset = 0; offset < TRANSPOSE_TILE; offset += TRANSPOSE_ROWS) {
        if (x < width && y + offset < height) {
            tile[threadIdx.y + offset][threadIdx.x] = src[(y + offset) * width + x];
        }
    }
    __syncthreads();
    x = blockIdx.y * TRANSPOSE_TILE + threadIdx.x;
    y = blockIdx.x * TRANSPOSE_TILE + threadIdx.y;
    for (int offset = 0; offset < TRANSPOSE_TILE; offset += TRANSPOSE_ROWS) {
        if (x < height && y + offset < width) {
            dst[(y + offset) * height + x] = tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

__global__ void row_prefix_u64_inplace_kernel(
    unsigned long long* values,
    int width,
    int height) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    if (row >= height) return;
    __shared__ unsigned long long scan[SCAN_THREADS];
    __shared__ unsigned long long carry;
    __shared__ unsigned long long chunk_carry;
    if (lane == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < width; base += SCAN_THREADS) {
        int column = base + lane;
        scan[lane] = column < width ? values[row * width + column] : 0ULL;
        __syncthreads();
        for (int offset = 1; offset < SCAN_THREADS; offset <<= 1) {
            unsigned long long add = lane >= offset ? scan[lane - offset] : 0ULL;
            __syncthreads();
            scan[lane] += add;
            __syncthreads();
        }
        if (lane == 0) chunk_carry = carry;
        __syncthreads();
        if (column < width) values[row * width + column] = scan[lane] + chunk_carry;
        __syncthreads();
        int valid = min(SCAN_THREADS, width - base);
        if (lane == 0) carry = chunk_carry + scan[valid - 1];
        __syncthreads();
    }
}

__device__ unsigned long long integral_value_transposed(
    const unsigned long long* integral_transposed,
    int padded_height,
    int x,
    int y) {
    if (x < 0 || y < 0) return 0ULL;
    return integral_transposed[x * padded_height + y];
}

__global__ void adaptive_integral_kernel(
    const uint8_t* src,
    const unsigned long long* integral_transposed,
    uint8_t* dst,
    int width,
    int height,
    int padded_height,
    int block_size,
    float c,
    int max_value,
    int invert) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int x0 = x;
    int y0 = y;
    int x1 = x + block_size - 1;
    int y1 = y + block_size - 1;
    unsigned long long bottom_right =
        integral_value_transposed(integral_transposed, padded_height, x1, y1);
    unsigned long long above =
        integral_value_transposed(integral_transposed, padded_height, x1, y0 - 1);
    unsigned long long left =
        integral_value_transposed(integral_transposed, padded_height, x0 - 1, y1);
    unsigned long long above_left =
        integral_value_transposed(integral_transposed, padded_height, x0 - 1, y0 - 1);
    unsigned long long sum = (bottom_right + above_left) - (above + left);
    unsigned long long area = static_cast<unsigned long long>(block_size) * block_size;
    int mean = static_cast<int>((sum + area / 2ULL) / area);
    bool selected = invert
        ? static_cast<int>(src[y * width + x]) <= mean - static_cast<int>(floorf(c))
        : static_cast<int>(src[y * width + x]) > mean - static_cast<int>(ceilf(c));
    dst[y * width + x] = static_cast<uint8_t>(selected ? max_value : 0);
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
}

static int execute_linear_plan_device(
    NativePlan* compiled,
    uint8_t* current,
    uint8_t* dst,
    int dst_stride,
    int dst_channels) {
    PersistentContext* context = compiled->context;
    int width = compiled->width;
    int height = compiled->height;
    int channels = compiled->input_channels;
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
                resize_gray_kernel<<<grid2d(target_width, target_height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                    current, next, width, height, target_width, target_height);
                current = next;
                width = target_width;
                height = target_height;
                break;
            }
            case VF_PLAN_GAUSSIAN: {
                context->timing_has_gaussian = true;
                cudaEventRecord(context->timing_events[TIMING_GAUSSIAN_START], context->stream);
                int radius = 0;
                int result = prepare_gaussian_weights(op.int_params[0], &radius);
                if (result != VF_CUDA_OK) return result;
                gaussian_horizontal_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                    current, context->gaussian_buffer, width, height, channels, radius);
                gaussian_vertical_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                    context->gaussian_buffer, next, width, height, channels, radius);
                cudaEventRecord(context->timing_events[TIMING_GAUSSIAN_END], context->stream);
                current = next;
                break;
            }
            case VF_PLAN_THRESHOLD:
                context->timing_has_threshold = true;
                cudaEventRecord(context->timing_events[TIMING_THRESHOLD_START], context->stream);
                threshold_kernel<<<(width * height + 255) / 256, 256, 0, context->stream>>>(
                    current, next, width * height, op.int_params[0],
                    op.int_params[1], op.int_params[2]);
                cudaEventRecord(context->timing_events[TIMING_THRESHOLD_END], context->stream);
                current = next;
                break;
            case VF_PLAN_ADAPTIVE_MEAN: {
                context->timing_has_adaptive = true;
                cudaEventRecord(context->timing_events[TIMING_ADAPTIVE_START], context->stream);
                int radius = 0, padded_width = 0, padded_height = 0;
                size_t padded_count = 0;
                int result = adaptive_layout(width, height, op.int_params[0], &radius, &padded_width,
                                             &padded_height, &padded_count);
                if (result != VF_CUDA_OK) return result;
                replicate_border_kernel<<<grid2d(padded_width, padded_height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                    current, context->u8[3], width, height, padded_width, padded_height, radius);
                row_prefix_u8_kernel<<<padded_height, SCAN_THREADS, 0, context->stream>>>(
                    context->u8[3], context->u64[0], padded_width, padded_height);
                dim3 transpose_block(TRANSPOSE_TILE, TRANSPOSE_ROWS);
                dim3 transpose_grid(
                    (padded_width + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
                    (padded_height + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);
                transpose_u64_kernel<<<transpose_grid, transpose_block, 0, context->stream>>>(
                    context->u64[0], context->u64[1], padded_width, padded_height);
                row_prefix_u64_inplace_kernel<<<padded_width, SCAN_THREADS, 0, context->stream>>>(
                    context->u64[1], padded_height, padded_width);
                adaptive_integral_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                    current, context->u64[1], next, width, height, padded_height,
                    op.int_params[0], op.float_params[0], op.int_params[1], op.int_params[2]);
                cudaEventRecord(context->timing_events[TIMING_ADAPTIVE_END], context->stream);
                current = next;
                break;
            }
            case VF_PLAN_MORPHOLOGY: {
                context->timing_has_morphology = true;
                cudaEventRecord(context->timing_events[TIMING_MORPHOLOGY_START], context->stream);
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
                    morph_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                        source_buffer, destination, width, height, channels, kernel / 2, dilate);
                    source_buffer = destination;
                    destination = destination == next ? context->u8[4] : next;
                }
                cudaEventRecord(context->timing_events[TIMING_MORPHOLOGY_END], context->stream);
                current = source_buffer;
                break;
            }
            default:
                return VF_CUDA_UNSUPPORTED;
        }
    }
    int result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    cudaEventRecord(context->timing_events[TIMING_AFTER_KERNEL], context->stream);
    const size_t output_row_bytes = static_cast<size_t>(width) * dst_channels;
    cudaError_t error = cudaMemcpy2DAsync(
        dst, dst_stride, current, output_row_bytes, output_row_bytes, height,
        cudaMemcpyDeviceToHost, context->stream);
    if (error != cudaSuccess) return cuda_result(error);
    cudaEventRecord(context->timing_events[TIMING_AFTER_OUTPUT], context->stream);
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
                cudaEventRecord(context->timing_events[TIMING_GAUSSIAN_START], context->stream);
                int radius = 0;
                int result = prepare_gaussian_weights(op.int_params[0], &radius);
                if (result != VF_CUDA_OK) return result;
                gaussian_horizontal_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                    input, context->gaussian_buffer, width, height, channels, radius);
                gaussian_vertical_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                    context->gaussian_buffer, output, width, height, channels, radius);
                cudaEventRecord(context->timing_events[TIMING_GAUSSIAN_END], context->stream);
                values[index] = output;
                break;
            }
            case VF_PLAN_THRESHOLD:
                context->timing_has_threshold = true;
                cudaEventRecord(context->timing_events[TIMING_THRESHOLD_START], context->stream);
                threshold_kernel<<<(static_cast<int>(pixels) + 255) / 256, 256, 0, context->stream>>>(
                    input, output, static_cast<int>(pixels), op.int_params[0],
                    op.int_params[1], op.int_params[2]);
                cudaEventRecord(context->timing_events[TIMING_THRESHOLD_END], context->stream);
                values[index] = output;
                break;
            case VF_PLAN_ADAPTIVE_MEAN: {
                context->timing_has_adaptive = true;
                cudaEventRecord(context->timing_events[TIMING_ADAPTIVE_START], context->stream);
                int radius = 0, padded_width = 0, padded_height = 0;
                size_t padded_count = 0;
                int result = adaptive_layout(width, height, op.int_params[0], &radius, &padded_width,
                                             &padded_height, &padded_count);
                if (result != VF_CUDA_OK) return result;
                replicate_border_kernel<<<grid2d(padded_width, padded_height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                    input, context->u8[3], width, height, padded_width, padded_height, radius);
                row_prefix_u8_kernel<<<padded_height, SCAN_THREADS, 0, context->stream>>>(
                    context->u8[3], context->u64[0], padded_width, padded_height);
                dim3 transpose_block(TRANSPOSE_TILE, TRANSPOSE_ROWS);
                dim3 transpose_grid(
                    (padded_width + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
                    (padded_height + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);
                transpose_u64_kernel<<<transpose_grid, transpose_block, 0, context->stream>>>(
                    context->u64[0], context->u64[1], padded_width, padded_height);
                row_prefix_u64_inplace_kernel<<<padded_width, SCAN_THREADS, 0, context->stream>>>(
                    context->u64[1], padded_height, padded_width);
                adaptive_integral_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                    input, context->u64[1], output, width, height, padded_height,
                    op.int_params[0], op.float_params[0], op.int_params[1], op.int_params[2]);
                cudaEventRecord(context->timing_events[TIMING_ADAPTIVE_END], context->stream);
                values[index] = output;
                break;
            }
            case VF_PLAN_MORPHOLOGY: {
                context->timing_has_morphology = true;
                cudaEventRecord(context->timing_events[TIMING_MORPHOLOGY_START], context->stream);
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
                    morph_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                        source_buffer, destination, width, height, channels,
                        op.int_params[1] / 2, dilate);
                    source_buffer = destination;
                    destination = destination == output ? context->u8[4] : output;
                }
                cudaEventRecord(context->timing_events[TIMING_MORPHOLOGY_END], context->stream);
                values[index] = source_buffer;
                break;
            }
            default:
                return VF_CUDA_UNSUPPORTED;
        }
    }
    int result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    cudaEventRecord(context->timing_events[TIMING_AFTER_KERNEL], context->stream);
    for (int index = 0; index < output_count; ++index) {
        int node = compiled->output_nodes[index];
        size_t row_bytes = static_cast<size_t>(width) * compiled->node_channels[node];
        cudaError_t error = cudaMemcpy2DAsync(
            outputs[index].data, outputs[index].stride, values[node], row_bytes,
            row_bytes, height, cudaMemcpyDeviceToHost, context->stream);
        if (error != cudaSuccess) return cuda_result(error);
    }
    cudaEventRecord(context->timing_events[TIMING_AFTER_OUTPUT], context->stream);
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
    uint64_t bytes = 0;
    for (size_t capacity : persistent->u8_capacity) bytes += static_cast<uint64_t>(capacity);
    bytes += static_cast<uint64_t>(persistent->gaussian_capacity) * sizeof(uint32_t);
    for (size_t capacity : persistent->u64_capacity) {
        bytes += static_cast<uint64_t>(capacity) * sizeof(unsigned long long);
    }
    for (size_t capacity : persistent->dag_u8_capacity) bytes += static_cast<uint64_t>(capacity);
    bytes += static_cast<uint64_t>(persistent->resident_capacity);
    *reserved_bytes = bytes;
    *allocation_count = persistent->allocation_count;
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
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || generation == nullptr ||
        (src_channels != 1 && src_channels != 3) ||
        !visionflow_cuda::valid_image(src, width, height, src_stride, src_channels)) {
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
    cudaError_t error = cudaMemcpy2DAsync(
        persistent->resident_u8, row_bytes, src, src_stride, row_bytes, height,
        cudaMemcpyHostToDevice, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    cudaEventRecord(persistent->timing_events[TIMING_AFTER_INPUT], persistent->stream);
    cudaEventRecord(persistent->timing_events[TIMING_AFTER_KERNEL], persistent->stream);
    cudaEventRecord(persistent->timing_events[TIMING_AFTER_OUTPUT], persistent->stream);
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
    cudaEventRecord(context->timing_events[TIMING_AFTER_INPUT], context->stream);

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
    cudaEventRecord(context->timing_events[TIMING_AFTER_INPUT], context->stream);
    return execute_linear_plan_device(
        compiled, context->u8[0], dst, dst_stride, dst_channels);
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
    cudaEventRecord(context->timing_events[TIMING_AFTER_INPUT], context->stream);

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
    cudaEventRecord(context->timing_events[TIMING_AFTER_INPUT], context->stream);
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
    resize_gray_kernel<<<grid2d(dw, dh), dim3(BLOCK_X, BLOCK_Y)>>>(ds, dd, w, h, dw, dh);
    result = visionflow_cuda::kernel_result();
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
    gaussian_horizontal_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y)>>>(ds, intermediate, w, h, sc, radius);
    gaussian_vertical_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y)>>>(intermediate, dd, w, h, sc, radius);
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
    int radius = 0, padded_width = 0, padded_height = 0;
    size_t padded_count = 0;
    int result = adaptive_layout(
        w, h, block, &radius, &padded_width, &padded_height, &padded_count);
    if (result != VF_CUDA_OK) return result;
    uint8_t *ds = nullptr, *dd = nullptr;
    uint8_t* padded = nullptr;
    unsigned long long* row_prefix = nullptr;
    unsigned long long* integral_transposed = nullptr;
    result = alloc_copy(src, w, h, stride, 1, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    result = visionflow_cuda::allocate_bytes(&padded, padded_count);
    if (result != VF_CUDA_OK) {
        visionflow_cuda::free_device(dd);
        visionflow_cuda::free_device(ds);
        return result;
    }
    cudaError_t error = cudaMalloc(&row_prefix, padded_count * sizeof(unsigned long long));
    if (error == cudaSuccess) error = cudaMalloc(&integral_transposed, padded_count * sizeof(unsigned long long));
    if (error != cudaSuccess) {
        visionflow_cuda::free_device(integral_transposed);
        visionflow_cuda::free_device(row_prefix);
        visionflow_cuda::free_device(padded);
        visionflow_cuda::free_device(dd);
        visionflow_cuda::free_device(ds);
        return cuda_result(error);
    }
    replicate_border_kernel<<<grid2d(padded_width, padded_height), dim3(BLOCK_X, BLOCK_Y)>>>(
        ds, padded, w, h, padded_width, padded_height, radius);
    row_prefix_u8_kernel<<<padded_height, SCAN_THREADS>>>(padded, row_prefix, padded_width, padded_height);
    dim3 transpose_block(TRANSPOSE_TILE, TRANSPOSE_ROWS);
    dim3 transpose_grid(
        (padded_width + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
        (padded_height + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);
    transpose_u64_kernel<<<transpose_grid, transpose_block>>>(
        row_prefix, integral_transposed, padded_width, padded_height);
    row_prefix_u64_inplace_kernel<<<padded_width, SCAN_THREADS>>>(
        integral_transposed, padded_height, padded_width);
    adaptive_integral_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y)>>>(
        ds, integral_transposed, dd, w, h, padded_height, block, c, max_value, invert);
    result = visionflow_cuda::kernel_result();
    visionflow_cuda::free_device(integral_transposed);
    visionflow_cuda::free_device(row_prefix);
    visionflow_cuda::free_device(padded);
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
    int result = prepare_gaussian_weights(gaussian_kernel, &radius);
    if (result != VF_CUDA_OK) return result;
    int adaptive_radius = 0, padded_width = 0, padded_height = 0;
    size_t padded_count = 0;
    result = adaptive_layout(
        w,
        h,
        adaptive_block,
        &adaptive_radius,
        &padded_width,
        &padded_height,
        &padded_count);
    if (result != VF_CUDA_OK) return result;

    PersistentContext* persistent = static_cast<PersistentContext*>(context);
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
            &persistent->u8[3], &persistent->u8_capacity[3], padded_count, &persistent->allocation_count);
    }
    if (result == VF_CUDA_OK) {
        result = reserve_device(
            &persistent->gaussian_buffer,
            &persistent->gaussian_capacity,
            pixel_count,
            &persistent->allocation_count);
    }
    if (result == VF_CUDA_OK) {
        result = reserve_device(
            &persistent->u64[0],
            &persistent->u64_capacity[0],
            padded_count,
            &persistent->allocation_count);
    }
    if (result == VF_CUDA_OK) {
        result = reserve_device(
            &persistent->u64[1],
            &persistent->u64_capacity[1],
            padded_count,
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
    replicate_border_kernel<<<grid2d(padded_width, padded_height), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        gray,
        persistent->u8[3],
        w,
        h,
        padded_width,
        padded_height,
        adaptive_radius);
    row_prefix_u8_kernel<<<padded_height, SCAN_THREADS, 0, persistent->stream>>>(
        persistent->u8[3], persistent->u64[0], padded_width, padded_height);
    dim3 transpose_block(TRANSPOSE_TILE, TRANSPOSE_ROWS);
    dim3 transpose_grid(
        (padded_width + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
        (padded_height + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);
    transpose_u64_kernel<<<transpose_grid, transpose_block, 0, persistent->stream>>>(
        persistent->u64[0], persistent->u64[1], padded_width, padded_height);
    row_prefix_u64_inplace_kernel<<<padded_width, SCAN_THREADS, 0, persistent->stream>>>(
        persistent->u64[1], padded_height, padded_width);
    adaptive_integral_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        gray,
        persistent->u64[1],
        persistent->u8[2],
        w,
        h,
        padded_height,
        adaptive_block,
        adaptive_c,
        max_value,
        invert);
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
        morph_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y)>>>(a, b, w, h, sc, kernel / 2, dilate);
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
