from __future__ import annotations

import logging
import os
import re
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from devices.ccd_models import DeviceError

# ============================================================
# Teledyne DALSA Sapera LT access through pythonnet.
#
# This module is the only place that names Sapera .NET members. `SAPERA_API_MANIFEST` lists every
# type, constructor, overload, property, event and enum value that `PythonnetSaperaInterop` uses;
# `check_api()` verifies the manifest by .NET reflection against the machine's own
# DALSA.SaperaLT.SapClassBasic.dll before any hardware access. The field machine runs Sapera LT
# 8.60 and is offline, so a missing member must be reported up front instead of failing mid-connect.
#
# Overloads are always selected explicitly: pythonnet otherwise binds e.g.
# `SapAcquisition.GetParameter(Prm, 0)` to the `(Prm, out Double)` overload and reads 0.
# ============================================================

LOGGER = logging.getLogger(__name__)

SAPERA_NAMESPACE = "DALSA.SaperaLT.SapClassBasic"
ASSEMBLY_FILE_NAME = f"{SAPERA_NAMESPACE}.dll"
DLL_PATH_ENV = "VISIONFLOW_SAPERA_DLL"
SAPERADIR_ENV = "SAPERADIR"
DEFAULT_SAPERA_DIR = r"C:\Program Files\Teledyne DALSA\Sapera"
KNOWN_ASSEMBLY_SUBPATHS = (
    ("Components", "NET", "Bin", ASSEMBLY_FILE_NAME),
    ("Components", "NET", ASSEMBLY_FILE_NAME),
)
NATIVE_RUNTIME_FILE = "corapi.dll"
TARGET_SAPERA_VERSION = "8.60.0.00.2120"
DEFAULT_CCF_SUBDIR = ("CamFiles", "User")
# Buffer classes in preference order. `SapBufferWithTrash` additionally reports frames that landed in
# the trash buffer; plain `SapBuffer` is the documented fallback when that class or its constructor
# is absent. The accepted constructor shape is `(int count, SapAcquisition acq, MemoryType...)` with
# the memory type repeated for every remaining parameter.
BUFFER_CLASS_PREFERENCE = ("SapBufferWithTrash", "SapBuffer")
BUFFER_WITH_TRASH_CLASS = BUFFER_CLASS_PREFERENCE[0]
_SEARCH_MAX_DEPTH = 6
_SAMPLE_DIR_WORDS = ("demo", "example", "sample")


class SaperaError(DeviceError):
    """Sapera failure carrying a short operator code that can be copied by hand from the screen."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        summary = ERROR_MESSAGES.get(code, "Sapera 錯誤")
        super().__init__(f"{code} {summary}" + (f"：{detail}" if detail else ""))


# Codes are grouped by diagnostic step (S1 … S8); docs/sapera-diagnose.md is the operator table.
ERROR_MESSAGES = {
    "E-0101": "pythonnet 未安裝或無法匯入",
    "E-0102": ".NET Framework runtime 載入失敗",
    "E-0103": "需要 64 位元程式",
    "E-0104": "找不到 Sapera LT 安裝目錄",
    "E-0201": "找不到 DALSA.SaperaLT.SapClassBasic.dll",
    "E-0202": "SapClassBasic.dll 載入失敗",
    "E-0203": "Sapera managed DLL 與 runtime 版本不符",
    "E-0301": "Sapera API 缺少必要成員",
    "E-0401": "列舉 Sapera server 失敗",
    "E-0402": "找不到擷取卡（Acq resource）",
    "E-0403": "CCF 檔不存在",
    "E-0404": "尚未選擇 Sapera 擷取卡（server）",
    "E-0501": "SapAcqDevice 建立失敗",
    "E-0502": "SapAcquisition 建立失敗",
    "E-0503": "SapBuffer 建立失敗",
    "E-0504": "SapAcqToBuf 建立失敗",
    "E-0505": "未偵測到相機訊號",
    "E-0506": "找不到可用的 SapBuffer 建構子",
    "E-0601": "相機 Line Rate（AcquisitionLineRate）寫入失敗",
    "E-0602": "Exposure 寫入失敗",
    "E-0603": "Gain 寫入失敗",
    "E-0604": "Length（CROP_HEIGHT）寫入失敗",
    "E-0605": "外部觸發參數寫入失敗",
    "E-0606": "One Frame（EXT_FRAME_TRIGGER_ENABLE）寫入失敗",
    "E-0607": "外部觸發未 arm",
    "E-0608": "板卡內部線觸發（INT_LINE_TRIGGER）寫入失敗",
    "E-0609": "相機 TriggerMode 讀回與要求不符（連續要 Off、外部要 On）",
    "E-0610": "相機 TriggerMode 無法寫入也無法讀回",
    "E-0611": "CCF 影像寬度與相機不符",
    "E-0701": "Snap 啟動失敗",
    "E-0702": "等待影像逾時",
    "E-0703": "影像複製失敗",
    "E-0704": "不支援的像素格式",
    "E-0705": "Grab 啟動失敗",
    "E-0801": "Sapera 物件清理失敗",
    "E-0901": "Sapera 呼叫發生未預期錯誤",
}

# .NET exception types that mean the managed DLL does not match the installed native runtime.
_VERSION_MISMATCH_EXCEPTIONS = (
    "System.IO.FileLoadException",
    "System.EntryPointNotFoundException",
    "System.MissingMethodException",
    "System.BadImageFormatException",
    "System.TypeLoadException",
    "System.DllNotFoundException",
)


def dotnet_exception_name(exc: BaseException) -> str:
    get_type = getattr(exc, "GetType", None)
    if callable(get_type):
        try:
            return str(get_type().FullName)
        except Exception:  # pragma: no cover - defensive
            pass
    return type(exc).__name__


def translate_exception(exc: BaseException, code: str = "E-0901") -> SaperaError:
    """Map a .NET/Python exception raised by a Sapera call to an operator-facing `SaperaError`."""

    if isinstance(exc, SaperaError):
        return exc
    name = dotnet_exception_name(exc)
    if name in _VERSION_MISMATCH_EXCEPTIONS:
        return SaperaError("E-0203", f"{name}: {exc}")
    return SaperaError(code, f"{name}: {exc}")


# ---- API manifest ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ApiMember:
    """One Sapera .NET member. Type names are relative to `SAPERA_NAMESPACE` unless `System.*`."""

    kind: str  # type | ctor | method | property | event | enum
    type_name: str
    name: str = ""
    params: tuple[str, ...] = ()

    def describe(self) -> str:
        if self.kind == "type":
            return self.type_name
        if self.kind == "enum":
            return f"{self.type_name}.{self.name}"
        if self.kind in ("ctor", "method"):
            label = self.type_name if self.kind == "ctor" else f"{self.type_name}.{self.name}"
            return f"{label}({', '.join(self.params)})"
        return f"{self.type_name}.{self.name}"


INT32 = "System.Int32"
INT32_OUT = "System.Int32&"
INT64 = "System.Int64"
STRING = "System.String"
STRING_OUT = "System.String&"
BOOL = "System.Boolean"
INTPTR = "System.IntPtr"
PRM = "SapAcquisition+Prm"
VAL = "SapAcquisition+Val"
CAP = "SapAcquisition+Cap"
RESOURCE_TYPE = "SapManager+ResourceType"
MEMORY_TYPE = "SapBuffer+MemoryType"
BUFFER_PRM = "SapBuffer+Prm"

# Acquisition parameters written or read by `SaperaLineScanCamera`, taken from the xx_ccd paths
# confirmed on the camera machine (PROJECT_HANDOFF.md "Current Parameter Implementation Notes").
ACQ_PARAMETERS = (
    "CROP_HEIGHT",
    "LINE_INTEGRATE_ENABLE",
    "LINE_INTEGRATE_METHOD",
    "LINE_INTEGRATE_DURATION",
    "LINE_INTEGRATE_PULSE0_POLARITY",
    "LINE_INTEGRATE_PULSE1_POLARITY",
    "LINE_TRIGGER_METHOD",
    "LINE_TRIGGER_ENABLE",
    "INT_LINE_TRIGGER_ENABLE",
    "INT_LINE_TRIGGER_FREQ",
    "INT_LINE_TRIGGER_FREQ_MIN",
    "INT_LINE_TRIGGER_FREQ_MAX",
    "CAM_LINE_TRIGGER_FREQ_MIN",
    "CAM_LINE_TRIGGER_FREQ_MAX",
    "EXT_LINE_TRIGGER_ENABLE",
    "EXT_FRAME_TRIGGER_ENABLE",
    "INT_FRAME_TRIGGER_ENABLE",
    "SHAFT_ENCODER_ENABLE",
    "CAM_TRIGGER_ENABLE",
    "EXT_TRIGGER_ENABLE",
)
ACQ_VALUES = ("LINE_INTEGRATE_METHOD_3", "ACTIVE_HIGH", "ACTIVE_LOW", "SIGNAL_NAME_PULSE1")
ACQ_CAPABILITIES = ("LINE_TRIGGER_METHOD",)
EXTERNAL_TRIGGER_EVENTS = ("ExternalTrigger", "ExternalTrigger2")
TRIGGER_TIMING_EVENTS = (
    "ExternalTriggerIgnored",
    "ExternalTriggerTooSlow",
    "ExtLineTriggerTooSlow",
    "LineTriggerTooFast",
)
ACQ_EVENTS = EXTERNAL_TRIGGER_EVENTS + TRIGGER_TIMING_EVENTS

# Attached-camera features written through SapAcqDevice.SetFeatureValue. These are runtime camera
# features, so .NET reflection cannot verify them; they live here with the manifest so that no
# Sapera name is scattered through the binding (Todo P11 "以 8.60 為目標 API"). The candidate lists
# are tried in order because not every camera exposes the same selector or source spelling.
DEVICE_LINE_RATE_FEATURE = "AcquisitionLineRate"
# Camera-side image width, read to compare with the width the CCF gives the board (field: a 640x480
# CCF on a 16384 px Linea). GenICam SFNC order: current AOI width, sensor maximum, sensor width.
DEVICE_WIDTH_FEATURES = ("Width", "WidthMax", "SensorWidth")
DEVICE_GAIN_FEATURE = "Gain"
DEVICE_EXPOSURE_FEATURES = (
    "ExposureTime",
    "ExposureTimeAbs",
    "ExposureTimeRaw",
    "Exposure",
    "LineExposureTime",
    "AcquisitionExposureTime",
    "ShutterTime",
    "ShutterDuration",
)
DEVICE_TRIGGER_SELECTOR_FEATURE = "TriggerSelector"
DEVICE_TRIGGER_MODE_FEATURE = "TriggerMode"
DEVICE_TRIGGER_SOURCE_FEATURE = "TriggerSource"
DEVICE_TRIGGER_MODE_ON = "On"
DEVICE_TRIGGER_MODE_OFF = "Off"
CONTINUOUS_TRIGGER_SELECTORS = ("FrameStart", "LineStart", "AcquisitionStart", "ExposureStart")
EXTERNAL_DISABLED_SELECTORS = ("FrameStart", "AcquisitionStart", "ExposureStart")
EXTERNAL_LINE_SELECTORS = ("LineStart", "LineTrigger", "AcquisitionLine")
EXTERNAL_LINE_SOURCES = (
    "Line1",
    "Input1",
    "CC1",
    "CameraControl1",
    "CameraLinkCC1",
    "CL_CC1",
    "External",
    "ExternalLine",
    "LineTrigger",
)
EXTERNAL_LINE_INTEGRATE_DURATION = 40


def _build_manifest() -> tuple[ApiMember, ...]:
    m: list[ApiMember] = []
    add = m.append
    for type_name in (
        "SapLocation",
        "SapManager",
        "SapFeature",
        "SapAcqDevice",
        "SapAcquisition",
        "SapBuffer",
        "SapAcqToBuf",
        "SapXferPair",
        "SapAcqNotifyEventArgs",
        "SapSignalNotifyEventArgs",
        "SapXferNotifyEventArgs",
    ):
        add(ApiMember("type", type_name))
    # SapLocation / SapManager
    add(ApiMember("ctor", "SapLocation", params=(STRING, INT32)))
    add(ApiMember("property", "SapLocation", "ServerName"))
    add(ApiMember("property", "SapLocation", "ResourceIndex"))
    add(ApiMember("method", "SapManager", "GetServerCount"))
    add(ApiMember("method", "SapManager", "GetServerName", (INT32,)))
    add(ApiMember("method", "SapManager", "GetResourceCount", (STRING, RESOURCE_TYPE)))
    add(ApiMember("method", "SapManager", "GetResourceName", (STRING, RESOURCE_TYPE, INT32)))
    add(ApiMember("enum", RESOURCE_TYPE, "Acq"))
    add(ApiMember("enum", RESOURCE_TYPE, "AcqDevice"))
    # Attached-camera features (SapAcqDevice / SapFeature)
    add(ApiMember("ctor", "SapAcqDevice", params=("SapLocation",)))
    add(ApiMember("ctor", "SapFeature", params=("SapLocation",)))
    for type_name in ("SapAcqDevice", "SapFeature", "SapAcquisition", "SapBuffer", "SapAcqToBuf"):
        add(ApiMember("method", type_name, "Create"))
        add(ApiMember("method", type_name, "Destroy"))
        add(ApiMember("method", type_name, "Dispose"))
        add(ApiMember("property", type_name, "Initialized"))
    add(ApiMember("property", "SapAcqDevice", "Location"))
    add(ApiMember("method", "SapAcqDevice", "IsFeatureAvailable", (STRING,)))
    add(ApiMember("method", "SapAcqDevice", "GetFeatureInfo", (STRING, "SapFeature")))
    add(ApiMember("method", "SapAcqDevice", "SetFeatureValue", (STRING, STRING)))
    add(ApiMember("method", "SapAcqDevice", "SetFeatureValue", (STRING, INT64)))
    add(ApiMember("method", "SapAcqDevice", "GetFeatureValue", (STRING, STRING_OUT)))
    add(ApiMember("method", "SapAcqDevice", "UpdateFeaturesToDevice"))
    add(ApiMember("property", "SapFeature", "DataAccessMode"))
    # Acquisition
    add(ApiMember("ctor", "SapAcquisition", params=("SapLocation", STRING)))
    add(ApiMember("event", "SapAcquisition", "AcqNotify"))
    add(ApiMember("event", "SapAcquisition", "SignalNotify"))
    add(ApiMember("property", "SapAcquisition", "EventType"))
    add(ApiMember("property", "SapAcquisition", "SignalNotifyEnable"))
    add(ApiMember("property", "SapAcquisition", "SignalStatus"))
    add(ApiMember("property", "SapAcquisition", "CamIoControl"))
    add(ApiMember("method", "SapAcquisition", "IsParameterAvailable", (PRM,)))
    add(ApiMember("method", "SapAcquisition", "GetParameter", (PRM, INT32_OUT)))
    add(ApiMember("method", "SapAcquisition", "SetParameter", (PRM, INT32, BOOL)))
    add(ApiMember("method", "SapAcquisition", "SetParameter", (PRM, VAL, BOOL)))
    add(ApiMember("method", "SapAcquisition", "GetCapability", (CAP, INT32_OUT)))
    for name in ACQ_PARAMETERS:
        add(ApiMember("enum", PRM, name))
    for name in ACQ_VALUES:
        add(ApiMember("enum", VAL, name))
    for name in ACQ_CAPABILITIES:
        add(ApiMember("enum", CAP, name))
    for name in ACQ_EVENTS:
        add(ApiMember("enum", "SapAcquisition+AcqEventType", name))
    add(ApiMember("enum", "SapAcquisition+AcqSignalStatus", "None"))
    add(ApiMember("property", "SapAcqNotifyEventArgs", "EventType"))
    add(ApiMember("property", "SapSignalNotifyEventArgs", "SignalStatus"))
    # Buffers and transfer
    add(ApiMember("method", "SapBuffer", "IsBufferTypeSupported", ("SapLocation", MEMORY_TYPE)))
    add(ApiMember("enum", MEMORY_TYPE, "ScatterGather"))
    add(ApiMember("enum", MEMORY_TYPE, "ScatterGatherPhysical"))
    # The buffer class constructor is NOT asserted here: the reference app was compiled against a
    # Sapera build whose `SapBufferWithTrash(Int32, SapAcquisition, SapBuffer+MemoryType)` does not
    # exist on the field machine (8.60 reports it through E-0301), and a missing overload must not
    # make the whole camera unusable. `PythonnetSaperaInterop` probes the available constructors by
    # reflection and falls back to `SapBuffer`; see `BUFFER_CLASS_PREFERENCE`.
    # Buffer members are asserted on the base class: `SapBufferWithTrash` is only a probed
    # enhancement (see BUFFER_CLASS_PREFERENCE), and every member below is inherited from `SapBuffer`.
    add(ApiMember("method", "SapBuffer", "Clear"))
    add(ApiMember("property", "SapBuffer", "Width"))
    add(ApiMember("property", "SapBuffer", "Height"))
    add(ApiMember("method", "SapBuffer", "GetParameter", (BUFFER_PRM, INT32_OUT)))
    add(ApiMember("method", "SapBuffer", "ReadRect", (INT32, INT32, INT32, INT32, INTPTR)))
    add(ApiMember("enum", BUFFER_PRM, "PIXEL_DEPTH"))
    add(ApiMember("enum", BUFFER_PRM, "PITCH"))
    add(ApiMember("ctor", "SapAcqToBuf", params=("SapAcquisition", "SapBuffer")))
    add(ApiMember("property", "SapAcqToBuf", "Pairs"))
    add(ApiMember("event", "SapAcqToBuf", "XferNotify"))
    add(ApiMember("method", "SapAcqToBuf", "Grab"))
    add(ApiMember("method", "SapAcqToBuf", "Snap"))
    add(ApiMember("method", "SapAcqToBuf", "Freeze"))
    add(ApiMember("property", "SapXferPair", "EventType"))
    add(ApiMember("enum", "SapXferPair+XferEventType", "EndOfFrame"))
    add(ApiMember("property", "SapXferNotifyEventArgs", "Trash"))
    return tuple(m)


SAPERA_API_MANIFEST = _build_manifest()


def _full_type_name(name: str) -> str:
    if name.startswith("System.") or name.startswith(f"{SAPERA_NAMESPACE}."):
        return name
    return f"{SAPERA_NAMESPACE}.{name}"


def _parameter_type_names(method) -> tuple[str, ...]:
    return tuple(str(parameter.ParameterType.FullName) for parameter in method.GetParameters())


# Sapera declares the buffer source as `SapXferNode`, the base class of `SapAcquisition`; the
# reference app's `new SapBufferWithTrash(2, _acquisition, ...)` compiles through that conversion.
DEFAULT_BUFFER_SOURCE_TYPES = ("SapAcquisition", "SapXferNode")


def select_buffer_class(
    signatures: Mapping[str, Iterable[Iterable[str]]],
    source_types: Iterable[str] | None = None,
) -> tuple[str, int]:
    """Pick the buffer class and memory-argument count from the constructors `signatures` lists.

    `signatures` maps a class name to its constructor parameter-type lists. A usable constructor takes
    `(System.Int32, <source>, <SapBuffer+MemoryType>...)` where `<source>` is `SapAcquisition` or a
    type it converts to (`source_types`: its base-type chain, as C# overload resolution allows); the
    memory type may repeat (some Sapera builds take a separate trash memory type). Returns `("", 0)`
    when nothing matches. Pure so the field shapes can be tested without .NET.
    """

    accepted = {
        _full_type_name(name) for name in (source_types if source_types is not None else DEFAULT_BUFFER_SOURCE_TYPES)
    }
    accepted.add(_full_type_name("SapAcquisition"))
    memory_name = _full_type_name(MEMORY_TYPE)
    for class_name in BUFFER_CLASS_PREFERENCE:
        for parameters in signatures.get(class_name, ()):
            names = tuple(parameters)
            # At least one memory-type argument is required: dropping it would ignore the
            # ScatterGather/ScatterGatherPhysical choice the binding makes from the board capability.
            if len(names) < 3 or names[0] != "System.Int32" or names[1] not in accepted:
                continue
            if any(name != memory_name for name in names[2:]):
                continue
            return class_name, len(names) - 2
    return "", 0


def describe_buffer_ctors(signatures: Mapping[str, Iterable[Iterable[str]]]) -> str:
    """The constructors this build offers, namespaces stripped, for the E-0506 report line."""

    parts = []
    for class_name in BUFFER_CLASS_PREFERENCE:
        found = [
            f"{class_name}({', '.join(name.rsplit('.', 1)[-1] for name in parameters)})"
            for parameters in signatures.get(class_name, ())
        ]
        parts.append("、".join(found) if found else f"{class_name}（無公開建構子）")
    return "機台提供的建構子：" + "；".join(parts)


def check_api(assembly, manifest: Iterable[ApiMember] = SAPERA_API_MANIFEST) -> tuple[str, ...]:
    """Return a description of every manifest member missing from `assembly` (.NET reflection)."""

    import System  # noqa: PLC0415 - pythonnet namespace, available once the runtime is loaded

    missing: list[str] = []
    types: dict[str, object] = {}

    def resolve(type_name: str):
        if type_name not in types:
            types[type_name] = assembly.GetType(_full_type_name(type_name))
        return types[type_name]

    def parameter_names(method) -> tuple[str, ...]:
        return _parameter_type_names(method)

    def available(member: ApiMember, clr_type) -> tuple[str, ...]:
        """What the assembly actually offers for this member, so a field report is conclusive."""

        try:
            if member.kind == "ctor":
                return tuple(
                    f"{member.type_name}({', '.join(parameter_names(ctor))})"
                    for ctor in clr_type.GetConstructors()
                )
            if member.kind == "method":
                return tuple(
                    f"{member.type_name}.{method.Name}({', '.join(parameter_names(method))})"
                    for method in clr_type.GetMethods()
                    if str(method.Name) == member.name
                )
            if member.kind == "property":
                return tuple(f"{member.type_name}.{prop.Name}" for prop in clr_type.GetProperties())
            if member.kind == "event":
                return tuple(f"{member.type_name}.{event.Name}" for event in clr_type.GetEvents())
            if member.kind == "enum":
                return tuple(str(name) for name in System.Enum.GetNames(clr_type))
        except Exception:  # noqa: BLE001 - reflection detail is best effort only
            return ()
        return ()

    for member in manifest:
        clr_type = resolve(member.type_name)
        if clr_type is None:
            missing.append(member.describe())
            continue
        expected = tuple(_full_type_name(name) for name in member.params)
        if member.kind == "type":
            found = True
        elif member.kind == "ctor":
            found = any(parameter_names(ctor) == expected for ctor in clr_type.GetConstructors())
        elif member.kind == "method":
            found = any(
                str(method.Name) == member.name and parameter_names(method) == expected
                for method in clr_type.GetMethods()
            )
        elif member.kind == "property":
            found = any(str(prop.Name) == member.name for prop in clr_type.GetProperties())
        elif member.kind == "event":
            found = clr_type.GetEvent(member.name) is not None
        elif member.kind == "enum":
            found = bool(clr_type.IsEnum) and member.name in [str(name) for name in System.Enum.GetNames(clr_type)]
        else:  # pragma: no cover - manifest typo
            raise ValueError(f"unknown manifest kind {member.kind}")
        if not found:
            # The field can only copy short text back, so a failure must name what the assembly does
            # expose instead of only what the manifest expected.
            shown = available(member, clr_type)
            if shown:
                listed = "、".join(shown[:6]) + ("…" if len(shown) > 6 else "")
                missing.append(f"{member.describe()}（實際可用：{listed}）")
            else:
                missing.append(member.describe())
    return tuple(missing)


# ---- locating and loading the machine's Sapera ------------------------------------------------

@dataclass(frozen=True)
class AssemblySearch:
    sapera_dir: str | None
    checked: tuple[str, ...]
    found: tuple[str, ...]

    @property
    def chosen(self) -> str | None:
        return self.found[0] if self.found else None


def _rank_candidate(path: str) -> tuple[int, int, str]:
    lowered = path.lower()
    known = any(lowered.endswith(os.path.join(*parts).lower()) for parts in KNOWN_ASSEMBLY_SUBPATHS)
    sample = any(word in part for part in Path(lowered).parts for word in _SAMPLE_DIR_WORDS)
    return (0 if known else 1, 1 if sample else 0, len(path), path)


def _search_tree(root: Path, max_depth: int = _SEARCH_MAX_DEPTH) -> list[str]:
    results: list[str] = []
    root_depth = len(root.parts)
    for current, directories, files in os.walk(root):
        if len(Path(current).parts) - root_depth >= max_depth:
            directories[:] = []
        for name in files:
            if name.lower() == ASSEMBLY_FILE_NAME.lower():
                results.append(str(Path(current) / name))
    return results


def locate_assembly(
    dll_path: str | os.PathLike | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    default_sapera_dir: str = DEFAULT_SAPERA_DIR,
) -> AssemblySearch:
    """Find the machine's SapClassBasic.dll without guessing a version: explicit path, then SAPERADIR."""

    env = os.environ if environ is None else environ
    explicit = dll_path if dll_path is not None else (env.get(DLL_PATH_ENV) or None)
    if explicit is not None:
        path = str(Path(explicit))
        return AssemblySearch(None, (path,), (path,) if Path(path).is_file() else ())

    checked: list[str] = []
    found: list[str] = []
    sapera_dir = env.get(SAPERADIR_ENV) or None
    roots = [root for root in (sapera_dir, default_sapera_dir) if root]
    existing_root = None
    for root in dict.fromkeys(roots):
        root_path = Path(root)
        checked.append(str(root_path))
        if not root_path.is_dir():
            continue
        existing_root = existing_root or str(root_path)
        for parts in KNOWN_ASSEMBLY_SUBPATHS:
            candidate = root_path.joinpath(*parts)
            checked.append(str(candidate))
        found.extend(_search_tree(root_path))
    unique = sorted(dict.fromkeys(found), key=_rank_candidate)
    return AssemblySearch(existing_root, tuple(checked), tuple(unique))


@dataclass(frozen=True)
class SaperaVersions:
    assembly_path: str = ""
    assembly_version: str = ""
    assembly_file_version: str = ""
    native_path: str = ""
    native_file_version: str = ""

    @property
    def mismatch(self) -> bool:
        managed = _major_minor(self.assembly_file_version)
        native = _major_minor(self.native_file_version)
        return managed is not None and native is not None and managed != native

    def summary(self) -> str:
        native = self.native_file_version or "未知"
        return f"managed {self.assembly_file_version or self.assembly_version or '未知'}／runtime {native}"


def _major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"\s*(\d+)\.(\d+)", version or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def _native_runtime_candidates(environ: Mapping[str, str], sapera_dir: str | None) -> list[Path]:
    system_root = environ.get("SystemRoot") or environ.get("SYSTEMROOT") or r"C:\Windows"
    candidates = [Path(system_root) / "System32" / NATIVE_RUNTIME_FILE]
    if sapera_dir:
        candidates.append(Path(sapera_dir) / "Bin" / NATIVE_RUNTIME_FILE)
    return candidates


def ensure_dotnet_runtime() -> None:
    """Load pythonnet on the .NET Framework runtime once per process (Sapera .NET targets netfx)."""

    if struct.calcsize("P") != 8:
        raise SaperaError("E-0103", "Sapera LT x64 需要 64 位元 Python／VisionFlow。")
    try:
        import pythonnet  # noqa: PLC0415
    except ImportError as exc:
        raise SaperaError("E-0101", str(exc)) from exc
    try:
        if pythonnet.get_runtime_info() is None:
            pythonnet.load("netfx")
        import clr  # noqa: F401, PLC0415
    except Exception as exc:  # noqa: BLE001 - any runtime bootstrap failure
        raise SaperaError("E-0102", f"{type(exc).__name__}: {exc}") from exc


@dataclass
class SaperaRuntime:
    """The loaded SapClassBasic assembly and its namespace, ready for the interop layer."""

    assembly: object
    namespace: object
    versions: SaperaVersions
    search: AssemblySearch
    _interop: "PythonnetSaperaInterop | None" = field(default=None, repr=False)

    def check_api(self) -> tuple[str, ...]:
        return check_api(self.assembly)

    def interop(self) -> "PythonnetSaperaInterop":
        if self._interop is None:
            self._interop = PythonnetSaperaInterop(self.namespace)
        return self._interop


def load_runtime(
    dll_path: str | os.PathLike | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    default_sapera_dir: str = DEFAULT_SAPERA_DIR,
) -> SaperaRuntime:
    env = os.environ if environ is None else environ
    # Locate the machine's own assembly first: a machine without Sapera LT must not load pythonnet
    # or the .NET runtime at all, so startup stays cheap and a missing pythonnet is not reported as
    # a missing camera installation.
    search = locate_assembly(dll_path, env, default_sapera_dir=default_sapera_dir)
    if dll_path is None and not env.get(DLL_PATH_ENV) and search.sapera_dir is None:
        raise SaperaError("E-0104", f"未設定 {SAPERADIR_ENV}，且 {default_sapera_dir} 不存在；請確認已安裝 Sapera LT。")
    if search.chosen is None:
        raise SaperaError("E-0201", "已檢查：" + "；".join(search.checked))
    ensure_dotnet_runtime()
    import clr  # noqa: PLC0415
    import System  # noqa: PLC0415

    try:
        assembly = clr.AddReference(search.chosen)
        import importlib  # noqa: PLC0415

        namespace = importlib.import_module(SAPERA_NAMESPACE)
    except Exception as exc:  # noqa: BLE001
        error = translate_exception(exc, "E-0202")
        raise SaperaError(error.code, f"{search.chosen}：{error.detail}") from exc

    def file_version(path: str) -> str:
        try:
            return str(System.Diagnostics.FileVersionInfo.GetVersionInfo(path).FileVersion or "")
        except Exception:  # noqa: BLE001
            return ""

    native_path = next((p for p in _native_runtime_candidates(env, search.sapera_dir) if p.is_file()), None)
    versions = SaperaVersions(
        assembly_path=str(assembly.Location),
        assembly_version=str(assembly.GetName().Version),
        assembly_file_version=file_version(str(assembly.Location)),
        native_path=str(native_path) if native_path else "",
        native_file_version=file_version(str(native_path)) if native_path else "",
    )
    return SaperaRuntime(assembly, namespace, versions, search)


# ---- interop ----------------------------------------------------------------------------------

FrameCallback = Callable[[bool], None]
AcqEventCallback = Callable[[str], None]
SignalCallback = Callable[[bool], None]


@dataclass(frozen=True)
class BufferFormat:
    width: int
    height: int
    pixel_depth: int
    pitch: int


def _guarded(callback, label: str):
    """Sapera invokes handlers on its own threads; an escaping exception must never reach native code."""

    def handler(*args):
        try:
            callback(*args)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Sapera %s handler failed", label)

    return handler


class PythonnetSaperaInterop:
    """Thin, explicit-overload wrapper over SapClassBasic. `SaperaLineScanCamera` uses only this."""

    def __init__(self, namespace):
        import clr  # noqa: PLC0415
        import System  # noqa: PLC0415

        self._sap = namespace
        self._clr = clr
        self._system = System
        sap = namespace
        clr_type = clr.GetClrType
        int32_out = clr_type(System.Int32).MakeByRefType()
        string_out = clr_type(System.String).MakeByRefType()
        self._int64_out = clr_type(System.Int64).MakeByRefType()
        self._prm = sap.SapAcquisition.Prm
        self._val = sap.SapAcquisition.Val
        self._cap = sap.SapAcquisition.Cap
        self._acq_event = sap.SapAcquisition.AcqEventType
        self._resource_type = sap.SapManager.ResourceType
        self._sig_acq_get = (clr_type(sap.SapAcquisition.Prm), int32_out)
        self._sig_acq_set_int = (sap.SapAcquisition.Prm, System.Int32, System.Boolean)
        self._sig_acq_set_val = (sap.SapAcquisition.Prm, sap.SapAcquisition.Val, System.Boolean)
        self._sig_acq_cap = (clr_type(sap.SapAcquisition.Cap), int32_out)
        self._sig_buffer_get = (clr_type(sap.SapBuffer.Prm), int32_out)
        self._sig_set_string = (System.String, System.String)
        self._sig_set_int64 = (System.String, System.Int64)
        self._sig_get_string = (clr_type(System.String), string_out)
        self._sig_resource_count = (System.String, sap.SapManager.ResourceType)
        self._handlers: dict[int, list[tuple[str, object]]] = {}
        self._buffer_class = ""
        self._buffer_memory_args = 0
        self.buffer_with_trash = False
        self.buffer_ctor_signatures: dict[str, tuple[tuple[str, ...], ...]] = {}
        self._select_buffer_ctor()

    def _select_buffer_ctor(self) -> None:
        """Choose the buffer class this Sapera build exposes. Reflection only: no hardware calls.

        A wrong constructor would otherwise fail the whole API self-check (field report E-0301 for
        `SapBufferWithTrash(Int32, SapAcquisition, SapBuffer+MemoryType)` on Sapera LT 8.60), so the
        exact overload is discovered here instead of being asserted in the manifest.
        """

        signatures: dict[str, list[tuple[str, ...]]] = {}
        source_types = list(DEFAULT_BUFFER_SOURCE_TYPES)
        try:
            # Whatever `SapAcquisition` converts to on this build: its own type and every base class.
            node = self._clr.GetClrType(self._sap.SapAcquisition)
            while node is not None and str(node.FullName) != "System.Object":
                source_types.append(str(node.FullName))
                node = node.BaseType
        except Exception:  # noqa: BLE001 - fall back to the documented Sapera names
            LOGGER.debug("SapAcquisition base types unavailable", exc_info=True)
        for class_name in BUFFER_CLASS_PREFERENCE:
            clr_class = getattr(self._sap, class_name, None)
            if clr_class is None:
                continue
            try:
                # pythonnet's class object has no GetConstructors(); reflection needs the CLR type.
                clr_type = self._clr.GetClrType(clr_class)
                signatures[class_name] = [_parameter_type_names(ctor) for ctor in clr_type.GetConstructors()]
            except Exception:  # noqa: BLE001 - treat an unreadable type as unavailable
                continue
        self.buffer_ctor_signatures = {name: tuple(found) for name, found in signatures.items()}
        class_name, memory_args = select_buffer_class(signatures, source_types)
        if class_name:
            self._buffer_class = class_name
            self._buffer_memory_args = memory_args
            self.buffer_with_trash = class_name == BUFFER_WITH_TRASH_CLASS
            return
        LOGGER.warning("Sapera 沒有可用的 SapBuffer 建構子：%s", BUFFER_CLASS_PREFERENCE)
        LOGGER.debug("Sapera buffer 建構子簽名：%s", signatures)

    @property
    def buffer_class(self) -> str:
        """The selected buffer class, or an empty string when this Sapera build offers none."""

        return self._buffer_class

    # enumeration
    def server_count(self) -> int:
        return int(self._sap.SapManager.GetServerCount())

    def server_name(self, index: int) -> str:
        return str(self._sap.SapManager.GetServerName(int(index)))

    def resource_count(self, server_name: str, kind: str) -> int:
        method = self._sap.SapManager.GetResourceCount.Overloads[self._sig_resource_count]
        return int(method(str(server_name), getattr(self._resource_type, kind)))

    def resource_name(self, server_name: str, kind: str, index: int) -> str:
        return str(self._sap.SapManager.GetResourceName(str(server_name), getattr(self._resource_type, kind), int(index)))

    def location(self, server_name: str, resource_index: int):
        return self._sap.SapLocation(str(server_name), int(resource_index))

    # lifecycle shared by every Sapera object
    @staticmethod
    def create(obj) -> bool:
        return bool(obj.Create())

    @staticmethod
    def initialized(obj) -> bool:
        return bool(obj.Initialized)

    @staticmethod
    def destroy(obj) -> bool:
        return bool(obj.Destroy())

    def dispose(self, obj) -> None:
        for event_name, handler in self._handlers.pop(id(obj), []):
            event = getattr(obj, event_name)
            event -= handler
        obj.Dispose()

    # attached-camera features
    def new_acq_device(self, location):
        return self._sap.SapAcqDevice(location)

    @staticmethod
    def feature_available(device, name: str) -> bool:
        return bool(device.IsFeatureAvailable(str(name)))

    def feature_access_mode(self, device, name: str) -> str | None:
        feature = self._sap.SapFeature(device.Location)
        try:
            feature.Create()
            if device.GetFeatureInfo(str(name), feature):
                return str(feature.DataAccessMode)
            return None
        finally:
            try:
                if feature.Initialized:
                    feature.Destroy()
            finally:
                feature.Dispose()

    def feature_int_range(self, device, name: str) -> tuple[int | None, int | None]:
        """Optional probe: `(min, max)` of an integer feature, `None` where this build cannot say.

        Not part of the manifest (like the buffer constructor, a missing member must not fail S3):
        `SapFeature.GetValueMin/Max(out Int64)` is looked up at call time and any failure reads as
        unknown. Field report: Linea 16K rejects AcquisitionLineRate below 300 Hz.
        """

        try:
            feature = self._sap.SapFeature(device.Location)
        except Exception:  # noqa: BLE001 - an unknown range is a valid answer
            return None, None
        try:
            feature.Create()
            if not device.GetFeatureInfo(str(name), feature):
                return None, None
            return self._feature_bound(feature, "GetValueMin"), self._feature_bound(feature, "GetValueMax")
        except Exception:  # noqa: BLE001
            LOGGER.debug("feature range unavailable: %s", name, exc_info=True)
            return None, None
        finally:
            try:
                if feature.Initialized:
                    feature.Destroy()
                feature.Dispose()
            except Exception:  # noqa: BLE001 - probe cleanup is best effort
                LOGGER.debug("SapFeature cleanup failed", exc_info=True)

    def _feature_bound(self, feature, method_name: str) -> int | None:
        method = getattr(feature, method_name, None)
        if method is None:
            return None
        try:
            ok, value = method.Overloads[self._int64_out](0)
        except Exception:  # noqa: BLE001 - overload absent on this build
            return None
        return int(value) if ok else None

    def set_feature_string(self, device, name: str, value: str) -> bool:
        return bool(device.SetFeatureValue.Overloads[self._sig_set_string](str(name), str(value)))

    def set_feature_int64(self, device, name: str, value: int) -> bool:
        return bool(device.SetFeatureValue.Overloads[self._sig_set_int64](str(name), self._system.Int64(int(value))))

    def get_feature_string(self, device, name: str) -> str | None:
        ok, value = device.GetFeatureValue.Overloads[self._sig_get_string](str(name), None)
        return str(value) if ok else None

    @staticmethod
    def update_features(device) -> bool:
        return bool(device.UpdateFeaturesToDevice())

    # acquisition
    def new_acquisition(self, location, config_file: str, on_acq_event: AcqEventCallback, on_signal: SignalCallback):
        sap = self._sap
        acquisition = sap.SapAcquisition(location, str(config_file))
        signal_none = getattr(sap.SapAcquisition.AcqSignalStatus, "None")
        event_members = [(name, getattr(self._acq_event, name)) for name in ACQ_EVENTS]

        def acq_notify(_sender, args):
            event_type = args.EventType
            name = next((label for label, member in event_members if event_type == member), str(event_type))
            on_acq_event(name)

        def signal_notify(_sender, args):
            on_signal(args.SignalStatus != signal_none)

        self._subscribe(acquisition, "AcqNotify", _guarded(acq_notify, "AcqNotify"))
        self._subscribe(acquisition, "SignalNotify", _guarded(signal_notify, "SignalNotify"))
        mask = event_members[0][1]
        for _name, member in event_members[1:]:
            mask = mask | member
        acquisition.EventType = mask
        return acquisition

    def _subscribe(self, obj, event_name: str, handler) -> None:
        event = getattr(obj, event_name)
        event += handler
        self._handlers.setdefault(id(obj), []).append((event_name, handler))

    def acq_param_available(self, acquisition, name: str) -> bool:
        return bool(acquisition.IsParameterAvailable(getattr(self._prm, name)))

    def acq_get_int(self, acquisition, name: str) -> int | None:
        ok, value = acquisition.GetParameter.Overloads[self._sig_acq_get](getattr(self._prm, name), 0)
        return int(value) if ok else None

    def acq_set_int(self, acquisition, name: str, value: int) -> bool:
        method = acquisition.SetParameter.Overloads[self._sig_acq_set_int]
        return bool(method(getattr(self._prm, name), int(value), True))

    def acq_set_val(self, acquisition, name: str, value_name: str) -> bool:
        method = acquisition.SetParameter.Overloads[self._sig_acq_set_val]
        return bool(method(getattr(self._prm, name), getattr(self._val, value_name), True))

    def acq_capability(self, acquisition, name: str) -> int | None:
        ok, value = acquisition.GetCapability.Overloads[self._sig_acq_cap](getattr(self._cap, name), 0)
        return int(value) if ok else None

    def acq_set_cc1(self, acquisition, value_name: str) -> bool:
        controls = acquisition.CamIoControl
        if controls is None or len(controls) == 0 or controls[0] is None:
            return False
        controls[0].Value = int(getattr(self._val, value_name))
        acquisition.CamIoControl = controls
        return True

    @staticmethod
    def acq_read_cc1(acquisition) -> int | None:
        controls = acquisition.CamIoControl
        if controls is None or len(controls) == 0 or controls[0] is None:
            return None
        return int(controls[0].Value)

    def acq_signal_present(self, acquisition) -> bool:
        return acquisition.SignalStatus != getattr(self._sap.SapAcquisition.AcqSignalStatus, "None")

    @staticmethod
    def acq_enable_signal_notify(acquisition) -> None:
        acquisition.SignalNotifyEnable = True

    # buffers and transfer
    def new_buffers(self, acquisition, location, count: int = 2):
        """Create the buffer set with the class chosen by `_select_buffer_ctor`.

        Returns `(buffers, memory_type_name)`; the memory type name carries the class actually used so
        the status text and the diagnosis report show whether trash frames can be reported.
        """

        if not self._buffer_class:
            raise SaperaError("E-0506", describe_buffer_ctors(self.buffer_ctor_signatures))
        memory = self._sap.SapBuffer.MemoryType
        factory = getattr(self._sap, self._buffer_class)
        if self._sap.SapBuffer.IsBufferTypeSupported(location, memory.ScatterGather):
            value, name = memory.ScatterGather, "ScatterGather"
        else:
            value, name = memory.ScatterGatherPhysical, "ScatterGatherPhysical"
        args = [int(count), acquisition] + [value] * self._buffer_memory_args
        return factory(*args), f"{self._buffer_class}/{name}"

    @staticmethod
    def buffer_clear(buffers) -> bool:
        return bool(buffers.Clear())

    def buffer_format(self, buffers) -> BufferFormat | None:
        get = buffers.GetParameter.Overloads[self._sig_buffer_get]
        depth_ok, depth = get(self._sap.SapBuffer.Prm.PIXEL_DEPTH, 0)
        pitch_ok, pitch = get(self._sap.SapBuffer.Prm.PITCH, 0)
        if not (depth_ok and pitch_ok):
            return None
        return BufferFormat(int(buffers.Width), int(buffers.Height), int(depth), int(pitch))

    def buffer_read(self, buffers, destination: np.ndarray, width: int, height: int) -> bool:
        if not destination.flags.c_contiguous:
            raise ValueError("destination must be C-contiguous")
        address = self._system.IntPtr.op_Explicit(self._system.Int64(int(destination.ctypes.data)))
        return bool(buffers.ReadRect(0, 0, int(width), int(height), address))

    def new_transfer(self, acquisition, buffers, on_frame: FrameCallback):
        transfer = self._sap.SapAcqToBuf(acquisition, buffers)
        transfer.Pairs[0].EventType = self._sap.SapXferPair.XferEventType.EndOfFrame

        def xfer_notify(_sender, args):
            on_frame(bool(args.Trash))

        self._subscribe(transfer, "XferNotify", _guarded(xfer_notify, "XferNotify"))
        return transfer

    @staticmethod
    def grab(transfer) -> bool:
        return bool(transfer.Grab())

    @staticmethod
    def snap(transfer) -> bool:
        return bool(transfer.Snap())

    @staticmethod
    def freeze(transfer) -> bool:
        return bool(transfer.Freeze())
