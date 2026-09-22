from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MIB = 1024 * 1024


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


def _tile_extent(image_shape: tuple[int, ...], tile_config: dict) -> tuple[int, int]:
    image_height, image_width = (int(image_shape[0]), int(image_shape[1]))
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
) -> ResidentWorkingSetEstimate:
    """Conservatively estimate allocations needed before accepting a resident upload.

    The native context is serialized, so the production path has one live execution slot even
    when a throughput session allows several callers to wait at the Python queue. Grow-only native
    buffers are shared across tiles and detectors; capacity is therefore the maximum working set,
    not the sum of every tile in an image.
    """
    if len(image_shape) not in {2, 3} or int(image_nbytes) <= 0:
        raise ValueError("Resident image shape and byte size must be positive")
    channels = 1 if len(image_shape) == 2 else int(image_shape[2])
    if channels not in {1, 3}:
        raise ValueError("Resident image must have one or three channels")

    tile_width, tile_height = _tile_extent(image_shape, tile_config or {})
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
        # Gray frame + response plane + two uint64 radix-sort planes, CUB temporary storage and
        # bounded selected/output records. This is intentionally conservative because admission
        # occurs before the template is decoded and its exact response dimensions are known.
        anchor_scratch_bytes = image_width * image_height * 48 + max_candidates * 32
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
