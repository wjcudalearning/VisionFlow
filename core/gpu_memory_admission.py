from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MIB = 1024 * 1024

# Mirrors PATTERN_FFT_CROSSOVER_WORK_PER_PIXEL in gpu/visionflow_cuda.cu. The native Pattern Match
# picks the FFT response once the brute-force work (response elements x template pixels) exceeds
# this many multiply-adds per frame pixel, so admission has to reserve the FFT working set for
# exactly the same templates. Keep both in step.
PATTERN_FFT_CROSSOVER_WORK_PER_PIXEL = 4000.0

# cuFFT allocates its own work area outside the context, so it never appears in the context memory
# breakdown. Measured on RTX 3090 for the padded production transform; expressed as a multiple of
# the padded real plane so it scales with the frame.
_CUFFT_WORK_AREA_MULTIPLIER = 3.0


def pattern_fft_length(value: int) -> int:
    """Smallest 2/3/5/7-smooth transform length >= value, as the native path pads to."""
    candidate = max(1, int(value))
    while True:
        remaining = candidate
        for factor in (2, 3, 5, 7):
            while remaining % factor == 0:
                remaining //= factor
        if remaining == 1:
            return candidate
        candidate += 1


@dataclass(frozen=True, slots=True)
class ResidentWorkingSetEstimate:
    resident_frame_bytes: int
    tile_input_bytes: int
    plan_scratch_bytes: int
    dag_output_bytes: int
    detector_scratch_bytes: int
    anchor_scratch_bytes: int
    estimated_context_bytes: int
    current_context_bytes: int
    current_resident_bytes: int
    estimated_additional_bytes: int
    safety_headroom_bytes: int
    required_free_bytes: int
    estimated_working_set_bytes: int
    in_flight_slots: int
    policy: str = "resident-working-set-v1"

    def to_dict(self) -> dict[str, int | str]:
        return {
            "policy": self.policy,
            "resident_frame_bytes": self.resident_frame_bytes,
            "tile_input_bytes": self.tile_input_bytes,
            "plan_scratch_bytes": self.plan_scratch_bytes,
            "dag_output_bytes": self.dag_output_bytes,
            "detector_scratch_bytes": self.detector_scratch_bytes,
            "anchor_scratch_bytes": self.anchor_scratch_bytes,
            "estimated_context_bytes": self.estimated_context_bytes,
            "current_context_bytes": self.current_context_bytes,
            "current_resident_bytes": self.current_resident_bytes,
            "estimated_additional_bytes": self.estimated_additional_bytes,
            "safety_headroom_bytes": self.safety_headroom_bytes,
            "required_free_bytes": self.required_free_bytes,
            "estimated_working_set_bytes": self.estimated_working_set_bytes,
            "in_flight_slots": self.in_flight_slots,
        }


def _positive(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _pattern_template_extent(
    image_shape: tuple[int, ...], template_size: tuple[int, int] | None
) -> tuple[int, int] | None:
    """Return a usable (width, height) template size, or None when admission must stay conservative."""
    if template_size is None:
        return None
    try:
        width, height = (int(template_size[0]), int(template_size[1]))
    except (TypeError, ValueError, IndexError):
        return None
    image_height, image_width = (int(image_shape[0]), int(image_shape[1]))
    if width <= 0 or height <= 0 or width > image_width or height > image_height:
        return None
    return width, height


def _tile_extent(
    image_shape: tuple[int, ...],
    tile_config: dict,
    pattern_template: tuple[int, int] | None = None,
) -> tuple[int, int]:
    image_height, image_width = (int(image_shape[0]), int(image_shape[1]))
    if str(tile_config.get("mode", "grid")).lower() == "pattern_match" and pattern_template is not None:
        # PatternMatchTiler crops each match as template plus crop_padding on every side; the
        # grid width/height keys do not apply to this mode and usually are absent, which used to
        # charge Detector scratch for the whole frame.
        padding = max(0, int((tile_config.get("pattern_match") or {}).get("crop_padding", 0) or 0))
        return (
            min(pattern_template[0] + 2 * padding, image_width),
            min(pattern_template[1] + 2 * padding, image_height),
        )
    anchored = bool(str(tile_config.get("template_path", "")).strip())
    width_key, height_key = (("roi_w", "roi_h") if anchored else ("width", "height"))
    width = _positive(tile_config.get(width_key, tile_config.get("width")), image_width)
    height = _positive(tile_config.get(height_key, tile_config.get("height")), image_height)
    return min(width, image_width), min(height, image_height)


def _enabled_gpu_configs(detector_configs: dict) -> list[tuple[str, dict]]:
    return [
        (str(detector_id), config or {})
        for detector_id, config in (detector_configs or {}).items()
        if bool((config or {}).get("enabled", True)) and bool((config or {}).get("use_gpu", False))
    ]


def estimate_resident_working_set(
    image_shape: tuple[int, ...],
    image_nbytes: int,
    tile_config: dict,
    detector_configs: dict,
    *,
    total_device_bytes: int,
    context_stats: dict | None = None,
    pattern_template_size: tuple[int, int] | None = None,
) -> ResidentWorkingSetEstimate:
    """Conservatively estimate allocations needed before accepting a resident upload.

    The native context is serialized, so the production path has one live execution slot even
    when a throughput session allows several callers to wait at the Python queue. Grow-only native
    buffers are shared across tiles and detectors; capacity is therefore the maximum working set,
    not the sum of every tile in an image.

    ``pattern_template_size`` is the decoded (width, height) of the ``pattern_match`` template.
    When it is known, tiles and the response/sort planes are sized from it; otherwise the
    whole-frame bound is kept.
    """
    if len(image_shape) not in {2, 3} or int(image_nbytes) <= 0:
        raise ValueError("Resident image shape and byte size must be positive")
    channels = 1 if len(image_shape) == 2 else int(image_shape[2])
    if channels not in {1, 3}:
        raise ValueError("Resident image must have one or three channels")

    pattern_template = _pattern_template_extent(image_shape, pattern_template_size)
    tile_width, tile_height = _tile_extent(image_shape, tile_config or {}, pattern_template)
    tile_pixels = tile_width * tile_height
    tile_input_bytes = tile_pixels * channels
    # Linear native plan worst case: input plus u8 work/morphology planes and one shared uint32
    # plane. Gaussian stores its fixed-point horizontal pass there; Adaptive Mean reuses the same
    # allocation for exact uint32 row prefixes, so no padded image or uint64 integral planes
    # are admitted anymore. Nine bytes per input byte is conservative for both 1/3-channel plans.
    plan_scratch_bytes = 9 * tile_input_bytes

    enabled = _enabled_gpu_configs(detector_configs)
    # Native DAG nodes are grow-only u8 planes. Four planes per enabled detector safely covers
    # current production DAGs while keeping this estimate independent of detector implementation.
    dag_output_bytes = 4 * tile_pixels * max(1, len(enabled))
    # Automatic CNR candidate extraction owns union-find, sort/compact, box, ring and reduction
    # arrays in addition to plan buffers. Its measured production footprint is bounded here with
    # a conservative per-pixel allowance; other current detectors keep candidates on the CPU.
    detector_scratch_bytes = (
        64 * tile_pixels if any(detector_id == "202-CS-SN-1" for detector_id, _ in enabled) else 0
    )

    anchor_scratch_bytes = 0
    tile_mode = str((tile_config or {}).get("mode", "grid")).lower()
    if tile_mode == "pattern_match":
        image_height, image_width = int(image_shape[0]), int(image_shape[1])
        pattern = (tile_config or {}).get("pattern_match") or {}
        max_candidates = _positive(pattern.get("max_candidates"), 20000)
        if pattern_template is None:
            # Template unreadable here: the tiler will report it, so keep the whole-frame bound
            # (measured 48 B/px on RTX 3090 with the response as large as the frame).
            anchor_scratch_bytes = image_width * image_height * 48 + max_candidates * 32
        else:
            template_width, template_height = pattern_template
            response_elements = (image_width - template_width + 1) * (image_height - template_height + 1)
            frame_pixels = image_width * image_height
            brute_force_work = response_elements * template_width * template_height
            if brute_force_work > PATTERN_FFT_CROSSOVER_WORK_PER_PIXEL * frame_pixels:
                # FFT response path: the gray frame, two int64 summed-area tables, the padded real
                # plane, two R2C spectra, the same response/sort arrays as above, the template and
                # an allowance for cuFFT's own work area, which it allocates outside the context.
                pad_width = pattern_fft_length(image_width)
                pad_height = pattern_fft_length(image_height)
                spectrum_elements = (pad_width // 2 + 1) * pad_height
                fft_bytes = 4 * pad_width * pad_height + 16 * spectrum_elements
                anchor_scratch_bytes = (
                    frame_pixels
                    + 16 * frame_pixels
                    + fft_bytes
                    + int(_CUFFT_WORK_AREA_MULTIPLIER * 4 * pad_width * pad_height)
                    + response_elements * 29
                    + template_width * template_height
                    + 1 * MIB
                    + max_candidates * 32
                )
            else:
                # vf_pattern_match_gray_u8 over the whole frame: gray ROI (1 B) and two int64 prefix
                # planes (16 B) per frame pixel, then float scores, packed keys, sorted keys and
                # CUB's alternate key buffer (28 B) per response element. CUB's onesweep storage
                # adds about 0.25 B per element on RTX 3090 (16384x13000), so 29 B keeps a measured
                # margin. The template, histograms and bounded records are added on top.
                anchor_scratch_bytes = (
                    image_width * image_height * 17
                    + response_elements * 29
                    + template_width * template_height
                    + 1 * MIB
                    + max_candidates * 32
                )
    elif str((tile_config or {}).get("template_path", "")).strip():
        image_height, image_width = int(image_shape[0]), int(image_shape[1])
        search_width = min(_positive(tile_config.get("search_w"), image_width), image_width)
        search_height = min(_positive(tile_config.get("search_h"), image_height), image_height)
        # Three int64 score planes plus the smaller per-column candidate/result area.
        anchor_scratch_bytes = search_width * search_height * 24 + search_width * 32

    estimated_context_bytes = (
        plan_scratch_bytes + dag_output_bytes + detector_scratch_bytes + anchor_scratch_bytes
    )
    stats = context_stats if isinstance(context_stats, dict) else {}
    current_context_bytes = int(stats.get("reserved_bytes") or 0)
    breakdown = stats.get("breakdown") if isinstance(stats.get("breakdown"), dict) else {}
    current_resident_bytes = int(breakdown.get("resident_bytes") or 0)
    current_nonresident_bytes = max(0, current_context_bytes - current_resident_bytes)
    required_context_growth = max(0, estimated_context_bytes - current_nonresident_bytes)
    required_resident_growth = max(0, int(image_nbytes) - current_resident_bytes)
    estimated_additional_bytes = required_context_growth + required_resident_growth

    # Preserve room for driver/JIT allocations and other processes. Five percent scales on large
    # cards; 256 MiB remains a useful minimum on smaller cards without rejecting ordinary tiles.
    safety_headroom_bytes = max(256 * MIB, int(max(0, total_device_bytes) * 0.05))
    required_free_bytes = estimated_additional_bytes + safety_headroom_bytes
    estimated_working_set_bytes = (
        int(image_nbytes) + max(estimated_context_bytes, current_nonresident_bytes) + safety_headroom_bytes
    )
    return ResidentWorkingSetEstimate(
        resident_frame_bytes=int(image_nbytes),
        tile_input_bytes=tile_input_bytes,
        plan_scratch_bytes=plan_scratch_bytes,
        dag_output_bytes=dag_output_bytes,
        detector_scratch_bytes=detector_scratch_bytes,
        anchor_scratch_bytes=anchor_scratch_bytes,
        estimated_context_bytes=estimated_context_bytes,
        current_context_bytes=current_context_bytes,
        current_resident_bytes=current_resident_bytes,
        estimated_additional_bytes=estimated_additional_bytes,
        safety_headroom_bytes=safety_headroom_bytes,
        required_free_bytes=required_free_bytes,
        estimated_working_set_bytes=estimated_working_set_bytes,
        in_flight_slots=1,
    )
