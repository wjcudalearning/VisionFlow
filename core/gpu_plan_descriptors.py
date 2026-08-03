from __future__ import annotations

import ctypes

import numpy as np

from core.gpu_abi import VfDagPlanDescV1, VfPlanDescV1, VfPlanOperatorV1


class GpuPlanDescriptorBuilder:
    """Compile backend-neutral plan objects into the detector-neutral ABI v1 descriptors."""

    def __init__(self, error_type, plan_version: int = 1):
        self.error_type = error_type
        self.plan_version = int(plan_version)

    def linear(self, plan, image: np.ndarray):
        kinds = {"Gray": 1, "Gaussian": 2, "Threshold": 3, "AdaptiveMean": 4, "Morphology": 5, "Resize": 6}
        morphology_operations = {"open": 0, "close": 1, "dilate": 2, "erode": 3}
        encoded = []
        previous_node = -1
        for operator in plan.operations:
            name = type(operator).__name__
            if name not in kinds:
                raise self.error_type(f"Generic native plan does not support {name}")
            if name == "Morphology" and (
                str(operator.operation).lower() in {"", "none"}
                or int(operator.iterations) <= 0
                or int(operator.kernel_size) <= 1
            ):
                continue
            int_params = [0, 0, 0, 0]
            float_params = [0.0, 0.0]
            if name == "Gaussian":
                int_params[0] = int(operator.kernel_size)
            elif name == "Resize":
                if str(operator.interpolation).lower() != "area":
                    raise self.error_type(f"Generic native plan does not support Resize({operator.interpolation})")
                int_params[:2] = [int(operator.width), int(operator.height)]
            elif name == "Threshold":
                int_params[:3] = [int(operator.threshold), int(operator.max_value), int(operator.invert)]
            elif name == "AdaptiveMean":
                int_params[:3] = [int(operator.block_size), int(operator.max_value), int(operator.invert)]
                float_params[0] = float(operator.c)
            elif name == "Morphology":
                operation = str(operator.operation).lower()
                if operation not in morphology_operations:
                    raise self.error_type(f"Generic native plan does not support morphology {operation}")
                int_params[:3] = [morphology_operations[operation], int(operator.kernel_size), int(operator.iterations)]
            output_node = len(encoded)
            encoded.append(VfPlanOperatorV1(
                ctypes.sizeof(VfPlanOperatorV1),
                kinds[name],
                previous_node,
                output_node,
                (ctypes.c_int32 * 4)(*int_params),
                (ctypes.c_float * 2)(*float_params),
            ))
            previous_node = output_node
        if not encoded:
            raise self.error_type("Generic native plan contains only no-op operators")
        operators = (VfPlanOperatorV1 * len(encoded))(*encoded)
        input_channels = 1 if image.ndim == 2 else int(image.shape[2])
        descriptor = VfPlanDescV1(
            ctypes.sizeof(VfPlanDescV1),
            self.plan_version,
            input_channels,
            len(encoded),
            operators,
            previous_node,
        )
        return descriptor, operators

    def encode_operator(self, operator, input_node: int, output_node: int):
        kinds = {"Gray": 1, "Gaussian": 2, "Threshold": 3, "AdaptiveMean": 4, "Morphology": 5}
        morphology_operations = {"open": 0, "close": 1, "dilate": 2, "erode": 3}
        name = type(operator).__name__
        if name not in kinds:
            raise self.error_type(f"Generic native plan does not support {name}")
        int_params = [0, 0, 0, 0]
        float_params = [0.0, 0.0]
        if name == "Gaussian":
            int_params[0] = int(operator.kernel_size)
        elif name == "Threshold":
            int_params[:3] = [int(operator.threshold), int(operator.max_value), int(operator.invert)]
        elif name == "AdaptiveMean":
            int_params[:3] = [int(operator.block_size), int(operator.max_value), int(operator.invert)]
            float_params[0] = float(operator.c)
        elif name == "Morphology":
            operation = str(operator.operation).lower()
            if operation not in morphology_operations:
                raise self.error_type(f"Generic native plan does not support morphology {operation}")
            int_params[:3] = [morphology_operations[operation], int(operator.kernel_size), int(operator.iterations)]
        return VfPlanOperatorV1(
            ctypes.sizeof(VfPlanOperatorV1),
            kinds[name],
            input_node,
            output_node,
            (ctypes.c_int32 * 4)(*int_params),
            (ctypes.c_float * 2)(*float_params),
        )

    def dag(self, plan, image: np.ndarray):
        node_index = {node.name: index for index, node in enumerate(plan.nodes)}
        encoded = [
            self.encode_operator(
                node.operator,
                -1 if node.input_name == "root" else node_index[node.input_name],
                index,
            )
            for index, node in enumerate(plan.nodes)
        ]
        operators = (VfPlanOperatorV1 * len(encoded))(*encoded)
        output_nodes = (ctypes.c_int32 * len(plan.outputs))(*(node_index[name] for name in plan.outputs))
        input_channels = 1 if image.ndim == 2 else int(image.shape[2])
        descriptor = VfDagPlanDescV1(
            ctypes.sizeof(VfDagPlanDescV1),
            self.plan_version,
            input_channels,
            len(encoded),
            operators,
            len(plan.outputs),
            output_nodes,
        )
        return descriptor, operators, output_nodes

    def dag_node_channels(self, plan, image: np.ndarray) -> dict[str, int]:
        channels = {"root": 1 if image.ndim == 2 else int(image.shape[2])}
        for node in plan.nodes:
            input_channels = channels[node.input_name]
            name = type(node.operator).__name__
            if name in {"Threshold", "AdaptiveMean"} and input_channels != 1:
                raise self.error_type(f"{name} requires one-channel DAG input")
            channels[node.name] = 1 if name == "Gray" else input_channels
        return channels

    @staticmethod
    def kernel_launch_count(plan, input_channels: int) -> int:
        launches = 0
        channels = int(input_channels)
        for operator in plan.operations:
            name = type(operator).__name__
            if name == "Gray" and channels == 3:
                launches += 1
                channels = 1
            elif name == "Gaussian":
                launches += 2
            elif name in {"Threshold", "Resize"}:
                launches += 1
            elif name == "AdaptiveMean":
                launches += 5
            elif name == "Morphology":
                iterations = max(0, int(operator.iterations))
                launches += iterations * (2 if str(operator.operation).lower() in {"open", "close"} else 1)
        return launches
