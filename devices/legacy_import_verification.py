"""Evidence after applying values inferred from a legacy machine program.

Persisted settings, camera hardware readbacks, and physical signal tests are separate claims.
This module never talks to hardware or changes a Recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from devices.ccd_models import CameraRecipeSettings, CcdMachineSettings
from devices.legacy_program_import import ImportFinding


@dataclass(frozen=True)
class ImportVerification:
    key: str
    label: str
    state: str
    detail: str

    def line(self) -> str:
        return f"{self.label}：{self.state}（{self.detail}）"


_CAMERA_READBACKS = {
    "acquisition.length_lines": "CROP",
    "acquisition.exposure_time": "EXP",
    "acquisition.gain": "GAIN",
}


def _stored_value(key: str, machine: CcdMachineSettings, product: CameraRecipeSettings):
    section, field = key.split(".", 1)
    owner = product.acquisition if section == "acquisition" else getattr(machine, section)
    return getattr(owner, field)


def _readback_matches(expected: object, actual: str) -> bool | None:
    if not actual or actual in {"?", "na"}:
        return None
    try:
        return isclose(float(expected), float(actual), rel_tol=0, abs_tol=1e-6)
    except (TypeError, ValueError):
        return None


def verify_import(
    findings: tuple[ImportFinding, ...],
    machine: CcdMachineSettings,
    product: CameraRecipeSettings,
    *,
    applied_camera_settings: tuple | None,
    camera_readbacks: dict[str, str],
    meter_connected: bool,
) -> tuple[ImportVerification, ...]:
    results: list[ImportVerification] = []
    for finding in findings:
        if not finding.applicable:
            continue
        key = finding.key
        try:
            stored = _stored_value(key, machine, product)
        except (AttributeError, ValueError):
            results.append(ImportVerification(key, finding.label, "未確認", "目前版本沒有可核對的儲存欄位"))
            continue
        if stored != finding.value:
            results.append(ImportVerification(key, finding.label, "儲存值不符", f"預期 {finding.display}，目前 {stored}"))
            continue
        if key.startswith(("acquisition.", "connection.")):
            if applied_camera_settings is None or (machine.connection, product.acquisition) != applied_camera_settings[:2]:
                results.append(ImportVerification(key, finding.label, "待寫入", "相機需由操作者連線或重新連線"))
                continue
            readback_key = _CAMERA_READBACKS.get(key)
            actual = camera_readbacks.get(readback_key, "") if readback_key else ""
            matches = _readback_matches(finding.value, actual) if readback_key else None
            if matches is False:
                results.append(ImportVerification(key, finding.label, "讀回不符", f"預期 {finding.display}，硬體讀回 {actual}"))
            elif matches is True:
                results.append(ImportVerification(key, finding.label, "已驗證", f"硬體讀回 {actual}；畫面效果仍需實拍"))
            else:
                results.append(ImportVerification(key, finding.label, "已寫入但未讀回", "硬體未提供可核對的讀回值"))
        elif key.startswith("meter_wheel."):
            state = "已儲存，待硬體確認" if meter_connected else "待寫入"
            results.append(ImportVerification(key, finding.label, state, "米輪設定沒有獨立硬體讀回介面"))
        else:
            results.append(ImportVerification(key, finding.label, "待動作驗證", "儲存值吻合；需在機台確認接線、亮燈或影像效果"))
    return tuple(results)
