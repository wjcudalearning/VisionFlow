from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np


CONTRACT_PATH = Path(__file__).resolve().parents[1] / "gpu" / "equivalence_contract.json"
VALID_LEVELS = {"bit_exact", "decision_exact", "tolerance"}


@dataclass(frozen=True, slots=True)
class ArrayComparison:
    contract_id: str
    level: str
    max_abs_diff: float
    max_mean_abs_diff: float | None
    max_out_of_tolerance_ratio: float


@lru_cache(maxsize=4)
def load_gpu_equivalence_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported GPU equivalence contract schema")
    if set(payload.get("levels", {})) != VALID_LEVELS:
        raise ValueError("GPU equivalence contract must define every supported level")
    for group in ("operators", "detectors"):
        entries = payload.get(group)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"GPU equivalence contract has no {group}")
        for contract_id, entry in entries.items():
            if entry.get("level") not in VALID_LEVELS:
                raise ValueError(f"Invalid equivalence level for {contract_id}")
            golden_tests = entry.get("golden_tests")
            if not isinstance(golden_tests, list) or not golden_tests:
                raise ValueError(f"GPU equivalence contract has no golden test for {contract_id}")
    return payload


def array_comparison(contract_id: str) -> ArrayComparison:
    contract = load_gpu_equivalence_contract()
    try:
        entry = contract["operators"][contract_id]
    except KeyError as exc:
        raise KeyError(f"Unknown GPU equivalence contract: {contract_id}") from exc
    comparison = entry.get("comparison", {})
    if comparison.get("kind") != "array":
        raise ValueError(f"GPU equivalence contract is not array-comparable: {contract_id}")
    return ArrayComparison(
        contract_id=contract_id,
        level=str(entry["level"]),
        max_abs_diff=float(comparison["max_abs_diff"]),
        max_mean_abs_diff=(
            None
            if comparison.get("max_mean_abs_diff") is None
            else float(comparison["max_mean_abs_diff"])
        ),
        max_out_of_tolerance_ratio=float(comparison["max_out_of_tolerance_ratio"]),
    )


def compare_arrays(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    contract_id: str,
) -> dict[str, Any]:
    rule = array_comparison(contract_id)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{name}: shape/dtype mismatch actual={actual.shape}/{actual.dtype}, "
            f"expected={expected.shape}/{expected.dtype}"
        )
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    observed_max = float(delta.max(initial=0.0))
    observed_mean = float(delta.mean()) if delta.size else 0.0
    out_of_tolerance_ratio = float(
        np.count_nonzero(delta > rule.max_abs_diff) / max(delta.size, 1)
    )
    if out_of_tolerance_ratio > rule.max_out_of_tolerance_ratio:
        raise AssertionError(
            f"{name}: contract={contract_id} max_diff={observed_max:g} "
            f"(limit {rule.max_abs_diff:g}), out_of_tolerance_ratio="
            f"{out_of_tolerance_ratio:.6f} "
            f"(limit {rule.max_out_of_tolerance_ratio:.6f})"
        )
    if rule.max_mean_abs_diff is not None and observed_mean > rule.max_mean_abs_diff:
        raise AssertionError(
            f"{name}: contract={contract_id} mean_diff={observed_mean:g} "
            f"(limit {rule.max_mean_abs_diff:g})"
        )
    result = {
        "name": name,
        "contract_id": contract_id,
        "equivalence_level": rule.level,
        "max_diff": round(observed_max, 9),
        "mean_diff": round(observed_mean, 9),
        "mismatch_ratio": round(float(np.count_nonzero(delta) / max(delta.size, 1)), 6),
        "out_of_tolerance_ratio": round(out_of_tolerance_ratio, 6),
    }
    return result
