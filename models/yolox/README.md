# YOLOX model registry

`registry.yaml` is the only supported entry point for YOLOX models. Recipes store a
stable `model_id`; the registry resolves the model file and verifies its SHA-256
before ONNX Runtime creates a session.

The Recipe Designer lets the operator select an `.onnx` file directly, then
requires that exact file to be declared by the `registry.yaml` beside it. PyTorch
`.pt` and `.pth` checkpoints are not supported by the current ONNX Runtime
inference backend.

`yolox_tiny_fixture.onnx` is a 758-byte deterministic test fixture. It returns a
fixed raw YOLOX tensor for CPU reference, NMS, coordinate, recipe, and CLI tests.
It is marked `test_only` and must not be used as a production inspection model.
The fixture permits both `onnxruntime_cpu` and `onnxruntime_cuda` so the M3
validator can compare the same graph on both execution providers. CUDA remains
opt-in and requires `CUDAExecutionProvider`; this manifest entry is not evidence
of RTX or production accuracy acceptance.

Production model entries must define their own class names, input preprocessing,
letterbox behavior, output decoder/strides, supported backends, precision, version,
and checksum. Do not overwrite an existing model file while keeping the same
version or checksum.
