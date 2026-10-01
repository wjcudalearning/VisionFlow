"""Order-independent (parallel) reconstruction of ``cv2.findContours(RETR_LIST, CHAIN_APPROX_SIMPLE)``.

``tools/contour_reference.py`` ports OpenCV's sequential raster scan + border trace. This module
computes the *same* contours, points and order without any sequential trace, using only per-pixel
local rules, pointer jumping and list ranking, so the algorithm maps onto data-parallel GPU kernels
(the triad idea of Alonso-Jorda et al., "Accelerated border tracking in binary images with GPUs",
J. Supercomputing 2026, adapted to reproduce OpenCV exactly).

Facts the construction relies on (all checked against ``cv2.findContours`` by the tests):

1. A trace step of ``icvFetchContour`` depends only on the state ``(pixel, back direction)`` and
   the zero-ness of the 3x3 neighbourhood, never on marks: from back direction ``b`` the next
   pixel is the first non-zero neighbour scanning ``b+1, b+2, ...`` in the E,NE,N,NW,W,SW,S,SE
   ring, and the new back direction is the reverse of that step. Every border is a cycle of this
   successor function.
2. A state is on some border cycle iff (a) the step that entered it examined a zero (the
   neighbour at ``b+2`` for even ``b``, ``b+1`` for odd ``b``) and (b) the outgoing search
   examines at least one zero before the next foreground neighbour. Restricted to these states the
   successor is a permutation, so cycles can be labelled by pointer jumping.
3. Start events are those of the raster scan: an outer event at every foreground pixel with a zero
   left neighbour and a hole event at every foreground pixel with a zero right neighbour, ordered
   by (row, column, outer-before-hole). OpenCV emits each border cycle exactly once, at its first
   event in raster order, and returns contours in reverse discovery order. A foreground pixel
   without foreground neighbours is a one-point contour at its outer event.
4. ``CHAIN_APPROX_SIMPLE`` keeps the point of a state iff the outgoing step direction differs from
   the incoming one; OpenCV's ``prev_s = s ^ 4`` initialisation makes the start state follow the
   same rule, so the kept points do not depend on where the cycle starts.

``RETR_EXTERNAL`` keeps the outer borders whose outside background is 4-connected to the zero\nframe, in the same order. ``RETR_TREE``/``RETR_CCOMP`` report the same contours in hierarchy order\nand are not reconstructed here. The functions favour clarity over memory: they allocate
per padded pixel and are meant for correctness checks on masks up to a few megapixels.
"""

from __future__ import annotations

import numpy as np

__all__ = ["find_contours", "contour_summaries", "BORDER_TABLE", "NEXT_TABLE"]

_MODES = ("list", "external")


def _check_mode(mode: str) -> bool:
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    return mode == "external"

# (dy, dx) of the OpenCV ring: E, NE, N, NW, W, SW, S, SE.
_RING = np.array(
    [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)], dtype=np.int64
)


def _build_tables():
    fg = lambda nb, k: bool(nb >> (k & 7) & 1)  # noqa: E731
    next_dir = np.full((256, 8), -1, np.int8)
    border = np.zeros((256, 8), bool)
    start_outer = np.full(256, -1, np.int8)
    start_hole = np.full(256, -1, np.int8)
    for nb in range(256):
        for b in range(8):
            if not fg(nb, b):
                continue
            step = next(k for k in range(1, 9) if fg(nb, b + k))
            next_dir[nb, b] = (b + step) & 7
            entered_on_zero = not fg(nb, b + 2) if b % 2 == 0 else not fg(nb, b + 1)
            border[nb, b] = entered_on_zero and step > 1
        for order, table in (((3, 2, 1, 0, 7, 6, 5, 4), start_outer), ((7, 6, 5, 4, 3, 2, 1, 0), start_hole)):
            table[nb] = next((k for k in order if fg(nb, k)), -1)
    return next_dir, border, start_outer, start_hole


NEXT_TABLE, BORDER_TABLE, _START_OUTER, _START_HOLE = _build_tables()


def _neighbour_masks(img: np.ndarray) -> np.ndarray:
    height, width = img.shape
    padded = np.pad(img, 1)
    nb = np.zeros((height, width), np.int64)
    for k, (dy, dx) in enumerate(_RING):
        nb |= padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width].astype(np.int64) << k
    return nb


def _pointer_jump_min(values: np.ndarray, successor: np.ndarray) -> np.ndarray:
    """Minimum of ``values`` over each cycle of the permutation ``successor``."""
    result = values.copy()
    jump = successor.copy()
    # After round r every state holds the minimum over its next 2**r states; ceil(log2(n)) + 1
    # rounds cover the longest possible cycle.
    for _ in range(max(1, int(np.ceil(np.log2(max(successor.size, 2))))) + 1):
        result = np.minimum(result, result[jump])
        jump = jump[jump]
    return result


def _list_rank(successor: np.ndarray, is_head: np.ndarray) -> np.ndarray:
    """Distance of every state from its cycle head, following ``successor``."""
    count = successor.size
    predecessor = np.empty(count, np.int64)
    predecessor[successor] = np.arange(count)
    # Cut each cycle just before its head; ranks then count steps back to the head.
    back = predecessor.copy()
    back[is_head] = -1
    distance = np.where(is_head, 0, 1).astype(np.int64)
    pointer = back.copy()
    while np.any(pointer >= 0):
        active = pointer >= 0
        target = pointer[active]
        distance[active] += distance[target]
        pointer[active] = pointer[target]
    return distance


def _frame_background(zero: np.ndarray) -> np.ndarray:
    reached = np.zeros_like(zero)
    reached[0, :] = zero[0, :]
    reached[-1, :] = zero[-1, :]
    reached[:, 0] = zero[:, 0]
    reached[:, -1] = zero[:, -1]
    while True:
        grown = reached.copy()
        grown[1:, :] |= reached[:-1, :]
        grown[:-1, :] |= reached[1:, :]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= zero
        if np.array_equal(grown, reached):
            return reached
        reached = grown


def _cycles(mask: np.ndarray):
    """Shared construction: border states, successor permutation, heads and order keys."""
    src = np.asarray(mask)
    if src.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {src.shape}")
    height, width = src.shape
    img = np.zeros((height + 2, width + 2), np.uint8)
    img[1:-1, 1:-1] = src != 0
    padded_width = width + 2
    nb = np.zeros_like(img, dtype=np.int64)
    nb[1:-1, 1:-1] = _neighbour_masks(img)[1:-1, 1:-1]
    fg = img.astype(bool)

    # Border states, indexed densely as pixel * 8 + back direction.
    pixel_index = np.arange(img.size, dtype=np.int64).reshape(img.shape)
    state_ids = []
    for b in range(8):
        valid = fg & BORDER_TABLE[nb, b]
        state_ids.append(pixel_index[valid] * 8 + b)
    states = np.sort(np.concatenate(state_ids)) if state_ids else np.zeros(0, np.int64)

    pixels = states // 8
    backs = states % 8
    out_dir = NEXT_TABLE[nb.reshape(-1)[pixels], backs].astype(np.int64)
    offsets = _RING[:, 0] * padded_width + _RING[:, 1]
    next_states = (pixels + offsets[out_dir]) * 8 + ((out_dir + 4) & 7)
    successor = np.searchsorted(states, next_states)
    if states.size and not np.array_equal(states[np.minimum(successor, states.size - 1)], next_states):
        raise AssertionError("border successor left the border-state set")

    # Raster start events. Keys order (row, column, outer-before-hole).
    flat_nb = nb.reshape(-1)
    flat_fg = fg.reshape(-1)
    left_zero = flat_fg & ~np.roll(flat_fg, 1)
    right_zero = flat_fg & ~np.roll(flat_fg, -1)
    event_pixels = []
    event_keys = []
    event_backs = []
    for kind, flags, table in ((0, left_zero, _START_OUTER), (1, right_zero, _START_HOLE)):
        where = np.flatnonzero(flags)
        event_pixels.append(where)
        event_keys.append(where * 2 + kind)
        event_backs.append(table[flat_nb[where]].astype(np.int64))
    event_pixels = np.concatenate(event_pixels)
    event_keys = np.concatenate(event_keys)
    event_backs = np.concatenate(event_backs)

    isolated = (event_backs < 0) & (event_keys % 2 == 0)
    traced = event_backs >= 0
    event_states = event_pixels[traced] * 8 + event_backs[traced]
    event_slots = np.searchsorted(states, event_states)
    if event_states.size and not np.array_equal(states[event_slots], event_states):
        raise AssertionError("a start event is not a border state")

    sentinel = np.iinfo(np.int64).max
    first_key = np.full(states.size, sentinel, np.int64)
    np.minimum.at(first_key, event_slots, event_keys[traced])
    cycle_key = _pointer_jump_min(first_key, successor)
    if np.any(cycle_key == sentinel):
        raise AssertionError("a border cycle has no start event")
    is_head = first_key == cycle_key
    # Background 4-connected to the zero frame (a flood fill; on a device this is a CCL pass).
    frame_background = _frame_background(img == 0)
    return {
        "background_frame_component": frame_background,
        "padded_width": padded_width,
        "pixels": pixels,
        "backs": backs,
        "out_dir": out_dir,
        "successor": successor,
        "cycle_key": cycle_key,
        "is_head": is_head,
        "isolated_keys": event_keys[isolated],
        "isolated_pixels": event_pixels[isolated],
    }


def _ordered_heads(data, external: bool = False):
    keys = np.concatenate([data["cycle_key"][data["is_head"]], data["isolated_keys"]])
    if external:
        keys = keys[_is_top_level_outer(data, keys)]
    return np.sort(keys)[::-1]  # reverse discovery order


def _is_top_level_outer(data, keys: np.ndarray) -> np.ndarray:
    """RETR_EXTERNAL keeps outer borders whose outside background is connected to the frame.

    An outer event's left neighbour is a zero pixel of the background region that surrounds the
    component, so the border is top level iff that pixel is 4-connected to the zero frame.
    """
    background = data["background_frame_component"]
    pixels = keys // 2
    outer = keys % 2 == 0
    left = pixels - 1
    return outer & background.reshape(-1)[left]


def find_contours(mask: np.ndarray, mode: str = "list") -> list[np.ndarray]:
    """``cv2.findContours(mask, RETR_LIST|RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)[0]`` without a sequential trace."""
    external = _check_mode(mode)
    data = _cycles(mask)
    width = data["padded_width"]
    keep = data["out_dir"] != ((data["backs"] + 4) & 7)
    rank = _list_rank(data["successor"], data["is_head"])
    points = {}
    order = np.lexsort((rank, data["cycle_key"]))
    kept = order[keep[order]]
    keys = data["cycle_key"][kept]
    xs = data["pixels"][kept] % width - 1
    ys = data["pixels"][kept] // width - 1
    boundaries = np.flatnonzero(np.diff(keys)) + 1
    for group_keys, group_x, group_y in zip(
        np.split(keys, boundaries), np.split(xs, boundaries), np.split(ys, boundaries)
    ):
        if group_keys.size:
            points[int(group_keys[0])] = np.stack([group_x, group_y], axis=1)
    for key, pixel in zip(data["isolated_keys"], data["isolated_pixels"]):
        points[int(key)] = np.array([[pixel % width - 1, pixel // width - 1]])
    return [
        points[int(key)].astype(np.int32).reshape(-1, 1, 2) for key in _ordered_heads(data, external)
    ]


def contour_summaries(mask: np.ndarray, mode: str = "list") -> list[tuple[int, int, int, int, int, float]]:
    """Per contour ``(x, y, w, h, simple_point_count, contourArea)`` in OpenCV order, by reductions only.

    No point list is materialised: bbox, kept-point count and the shoelace area are segmented
    reductions over each cycle's states, which is what a device summary kernel computes.
    """
    external = _check_mode(mode)
    data = _cycles(mask)
    width = data["padded_width"]
    keys = data["cycle_key"]
    x = data["pixels"] % width - 1
    y = data["pixels"] // width - 1
    next_x = x + _RING[data["out_dir"], 1]
    next_y = y + _RING[data["out_dir"], 0]
    keep = (data["out_dir"] != ((data["backs"] + 4) & 7)).astype(np.int64)
    twice_area = x.astype(np.float64) * next_y - y.astype(np.float64) * next_x
    unique, inverse = np.unique(keys, return_inverse=True)
    stats = {}
    min_x = np.full(unique.size, np.iinfo(np.int64).max); np.minimum.at(min_x, inverse, x)
    max_x = np.full(unique.size, np.iinfo(np.int64).min); np.maximum.at(max_x, inverse, x)
    min_y = np.full(unique.size, np.iinfo(np.int64).max); np.minimum.at(min_y, inverse, y)
    max_y = np.full(unique.size, np.iinfo(np.int64).min); np.maximum.at(max_y, inverse, y)
    kept = np.zeros(unique.size, np.int64); np.add.at(kept, inverse, keep)
    area = np.zeros(unique.size, np.float64); np.add.at(area, inverse, twice_area)
    for index, key in enumerate(unique):
        stats[int(key)] = (
            int(min_x[index]), int(min_y[index]),
            int(max_x[index] - min_x[index] + 1), int(max_y[index] - min_y[index] + 1),
            int(kept[index]), float(abs(area[index] * 0.5)),
        )
    for key, pixel in zip(data["isolated_keys"], data["isolated_pixels"]):
        stats[int(key)] = (int(pixel % width - 1), int(pixel // width - 1), 1, 1, 1, 0.0)
    return [stats[int(key)] for key in _ordered_heads(data, external)]
