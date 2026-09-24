from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping
from pathlib import Path

from devices.ccd_models import DeviceAvailability, DeviceError, SensorRelaySettings
from devices.interfaces import DigitalIo

# ============================================================
# Advantech PCIe-1730 digital I/O through DAQNavi's .NET API (Automation.BDaq4.dll) and pythonnet.
# On the camera machine the Sensor is wired into a DI of this card and a DO of this card is wired to
# the grabber's frame-trigger input (confirmed by the operator 2026-09-24). The driver and assembly
# are installed by DAQNavi on the machine; they are never bundled. Only the file system is probed
# until the card is first opened, so a machine without DAQNavi never loads the .NET runtime for it.
# ============================================================

ASSEMBLY_FILE_NAME = "Automation.BDaq4.dll"
# Older DAQNavi releases ship the same `Automation.BDaq` namespace as Automation.BDaq.dll.
ASSEMBLY_FILE_NAMES = (ASSEMBLY_FILE_NAME, "Automation.BDaq.dll")
ASSEMBLY_NAME = "Automation.BDaq4"
NAMESPACE = "Automation.BDaq"
DLL_PATH_ENV = "VISIONFLOW_BDAQ_DLL"
# DAQNavi installs the assembly into the .NET GAC (4.x and the older 2.x GAC); the SDK keeps copies.
DEFAULT_SEARCH_DIRS = (
    Path(r"C:\Windows\Microsoft.NET\assembly\GAC_MSIL\Automation.BDaq4"),
    Path(r"C:\Windows\Microsoft.NET\assembly\GAC_MSIL\Automation.BDaq"),
    Path(r"C:\Windows\assembly\GAC_MSIL\Automation.BDaq4"),
    Path(r"C:\Windows\assembly\GAC_MSIL\Automation.BDaq"),
    Path(r"C:\Advantech\DAQNavi"),
    Path(r"C:\Program Files\Advantech\DAQNavi"),
    Path(r"C:\Program Files (x86)\Advantech\DAQNavi"),
)


def clean_path_text(text: str) -> str:
    """A pasted path without surrounding blanks or quotes (Explorer's "Copy as path" adds quotes)."""
    value = str(text or "").strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


def _find_in_folder(folder: Path) -> Path | None:
    for name in ASSEMBLY_FILE_NAMES:
        direct = folder / name
        if direct.is_file():
            return direct
    for name in ASSEMBLY_FILE_NAMES:
        for candidate in sorted(folder.rglob(name)):
            return candidate
    return None


def _explicit_candidate(text: str) -> tuple[Path | None, str]:
    """(dll, reason when not found) for a configured file or folder path."""
    path = Path(clean_path_text(text))
    try:
        if path.is_file():
            return path, ""
        if path.is_dir():
            found = _find_in_folder(path)
            if found is not None:
                return found, ""
            return None, f"資料夾「{path}」裡沒有 {'／'.join(ASSEMBLY_FILE_NAMES)}"
    except OSError as exc:
        return None, f"無法讀取「{path}」：{exc}"
    return None, f"「{path}」不存在"


def locate_assembly(
    assembly_path: str = "",
    environ: Mapping[str, str] | None = None,
    search_dirs: tuple[Path, ...] = DEFAULT_SEARCH_DIRS,
) -> Path | None:
    """The configured, environment or installed `Automation.BDaq4.dll`, without loading anything.

    A configured path may be the DLL itself or a folder that contains it, with or without quotes.
    """
    return _search(assembly_path, environ, search_dirs)[0]


def explain_missing(
    assembly_path: str = "",
    environ: Mapping[str, str] | None = None,
    search_dirs: tuple[Path, ...] = DEFAULT_SEARCH_DIRS,
) -> str:
    return _search(assembly_path, environ, search_dirs)[1]


def _search(assembly_path, environ, search_dirs) -> tuple[Path | None, str]:
    env = os.environ if environ is None else environ
    for label, explicit in (("設定的 DLL 位置", assembly_path), (DLL_PATH_ENV, env.get(DLL_PATH_ENV))):
        if clean_path_text(explicit or ""):
            found, reason = _explicit_candidate(explicit)
            return found, "" if found else f"{label}：{reason}"
    for folder in search_dirs:
        try:
            if folder.is_dir():
                found = _find_in_folder(folder)
                if found is not None:
                    return found, ""
        except OSError:
            continue
    return None, "預設安裝位置都找不到"


def _is_success(error_code) -> bool:
    return str(error_code).rsplit(".", 1)[-1] == "Success"


class AdvantechDigitalIo(DigitalIo):
    """PCIe-1730 instant DI/DO. Every call is serialized; the relay thread owns it while running."""

    def __init__(
        self,
        assembly_path: Callable[[], str] | str = "",
        environ: Mapping[str, str] | None = None,
        namespace_loader: Callable[[Path | None], object] | None = None,
    ):
        self._assembly_path = assembly_path
        self._environ = os.environ if environ is None else environ
        self._namespace_loader = namespace_loader or _load_namespace
        self._lock = threading.RLock()
        self._di = None
        self._do = None
        self._device = ""

    def _configured_path(self) -> str:
        source = self._assembly_path
        return str(source() if callable(source) else source or "")

    def availability(self) -> DeviceAvailability:
        if self._namespace_loader is not _load_namespace:
            return DeviceAvailability(True)
        path, reason = _search(self._configured_path(), self._environ, DEFAULT_SEARCH_DIRS)
        if path is not None:
            return DeviceAvailability(True)
        return DeviceAvailability(
            False,
            f"找不到研華 DAQNavi 的 {ASSEMBLY_FILE_NAME}（{reason}）；"
            "請在 Sensor 中繼面板按「瀏覽」選取這個 DLL，或選取它所在的資料夾。",
        )

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._di is not None

    def connect(self, settings: SensorRelaySettings) -> None:
        settings = settings.normalized()
        with self._lock:
            if self._di is not None and self._device == settings.device:
                return
            self.disconnect()
            path = locate_assembly(settings.assembly_path or self._configured_path(), self._environ)
            if path is None and self._namespace_loader is _load_namespace:
                raise DeviceError(self.availability().reason)
            try:
                bdaq = self._namespace_loader(path)
                di = bdaq.InstantDiCtrl()
                di.SelectedDevice = bdaq.DeviceInformation(settings.device)
                do = bdaq.InstantDoCtrl()
                do.SelectedDevice = bdaq.DeviceInformation(settings.device)
            except DeviceError:
                raise
            except Exception as exc:  # noqa: BLE001 - any driver/.NET failure becomes an operator message
                raise DeviceError(
                    f"無法開啟 I/O 卡「{settings.device}」：{type(exc).__name__}: {exc}。"
                    "請用研華 Navigator 確認裝置名稱，並確認原機台程式已關閉。"
                ) from exc
            self._di, self._do, self._device = di, do, settings.device

    def disconnect(self) -> None:
        with self._lock:
            for control in (self._di, self._do):
                if control is not None:
                    try:
                        control.Dispose()
                    except Exception:  # noqa: BLE001 - best-effort release
                        pass
            self._di = self._do = None
            self._device = ""

    def read_bit(self, port: int, bit: int) -> bool:
        with self._lock:
            if self._di is None:
                raise DeviceError("I/O 卡未連線。")
            try:
                result = self._di.ReadBit(int(port), int(bit), 0)
            except Exception as exc:  # noqa: BLE001
                raise DeviceError(f"讀取 DI port {port} bit {bit} 失敗：{type(exc).__name__}: {exc}") from exc
        error_code, value = result if isinstance(result, tuple) else (None, result)
        if error_code is not None and not _is_success(error_code):
            raise DeviceError(f"讀取 DI port {port} bit {bit} 失敗：{error_code}")
        return bool(int(value))

    def write_bit(self, port: int, bit: int, value: bool) -> None:
        with self._lock:
            if self._do is None:
                raise DeviceError("I/O 卡未連線。")
            try:
                error_code = self._do.WriteBit(int(port), int(bit), 1 if value else 0)
            except Exception as exc:  # noqa: BLE001
                raise DeviceError(f"寫入 DO port {port} bit {bit} 失敗：{type(exc).__name__}: {exc}") from exc
        if error_code is not None and not _is_success(error_code):
            raise DeviceError(f"寫入 DO port {port} bit {bit} 失敗：{error_code}")

    def close(self) -> None:
        self.disconnect()


def _load_namespace(path: Path | None):
    from devices.sapera_api import ensure_dotnet_runtime  # noqa: PLC0415 - shared netfx bootstrap

    ensure_dotnet_runtime()
    import importlib  # noqa: PLC0415

    import clr  # noqa: PLC0415

    clr.AddReference(str(path) if path is not None else ASSEMBLY_NAME)
    return importlib.import_module(NAMESPACE)
