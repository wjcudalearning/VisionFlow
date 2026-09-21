from __future__ import annotations

import ctypes


class VfPlanOperatorV1(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("kind", ctypes.c_int32),
                ("input_node", ctypes.c_int32), ("output_node", ctypes.c_int32),
                ("int_params", ctypes.c_int32 * 4), ("float_params", ctypes.c_float * 2)]


class VfPlanDescV1(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("version", ctypes.c_uint32),
                ("input_channels", ctypes.c_int32), ("operator_count", ctypes.c_int32),
                ("operators", ctypes.POINTER(VfPlanOperatorV1)), ("output_node", ctypes.c_int32)]


class VfDagPlanDescV1(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("version", ctypes.c_uint32),
                ("input_channels", ctypes.c_int32), ("operator_count", ctypes.c_int32),
                ("operators", ctypes.POINTER(VfPlanOperatorV1)), ("output_count", ctypes.c_int32),
                ("output_nodes", ctypes.POINTER(ctypes.c_int32))]


class VfDagOutputV1(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("node", ctypes.c_int32),
                ("data", ctypes.POINTER(ctypes.c_uint8)), ("stride", ctypes.c_int32),
                ("channels", ctypes.c_int32)]


class VfRoiV1(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("x", ctypes.c_int32),
                ("y", ctypes.c_int32), ("width", ctypes.c_int32), ("height", ctypes.c_int32)]


class VfCudaTimingsV1(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("version", ctypes.c_uint32),
                ("context_create_ms", ctypes.c_float), ("allocation_ms", ctypes.c_float),
                ("h2d_ms", ctypes.c_float), ("device_copy_ms", ctypes.c_float),
                ("kernel_ms", ctypes.c_float), ("d2h_ms", ctypes.c_float),
                ("synchronize_ms", ctypes.c_float), ("free_ms", ctypes.c_float),
                ("gaussian_ms", ctypes.c_float), ("adaptive_integral_ms", ctypes.c_float),
                ("threshold_ms", ctypes.c_float), ("morphology_ms", ctypes.c_float),
                ("total_device_ms", ctypes.c_float)]


class VfCudaContextMemoryStatsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("reserved_bytes", ctypes.c_uint64),
        ("peak_reserved_bytes", ctypes.c_uint64),
        ("allocation_count", ctypes.c_uint64),
        ("plan_bytes", ctypes.c_uint64),
        ("resident_bytes", ctypes.c_uint64),
        ("template_match_bytes", ctypes.c_uint64),
        ("contour_bytes", ctypes.c_uint64),
        ("median_bytes", ctypes.c_uint64),
        ("gaussian_f32_bytes", ctypes.c_uint64),
        ("cnr_mask_bytes", ctypes.c_uint64),
        ("cnr_candidate_bytes", ctypes.c_uint64),
    ]


class VfCudaContextMemoryStatsV2(ctypes.Structure):
    _fields_ = VfCudaContextMemoryStatsV1._fields_ + [
        ("peak_plan_bytes", ctypes.c_uint64),
        ("peak_resident_bytes", ctypes.c_uint64),
        ("peak_template_match_bytes", ctypes.c_uint64),
        ("peak_contour_bytes", ctypes.c_uint64),
        ("peak_median_bytes", ctypes.c_uint64),
        ("peak_gaussian_f32_bytes", ctypes.c_uint64),
        ("peak_cnr_mask_bytes", ctypes.c_uint64),
        ("peak_cnr_candidate_bytes", ctypes.c_uint64),
    ]
