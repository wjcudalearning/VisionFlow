from __future__ import annotations

import base64
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from devices.ccd_models import MultipleRate
from devices.serial_light import describe_bytes, escape_text

# ============================================================
# 「從原機台程式匯入」: read the machine's original C# program (.sln / .csproj / folder) as text and
# pull out the values VisionFlow needs (LSI-8181 meter wheel, PCIe-1730 Sensor relay, Sapera CCF,
# length, exposure, gain). Nothing is compiled or executed.
#
# The original program is object-oriented, so an argument is rarely a literal. Values are traced
# back through: method parameters (every caller, named/positional/default arguments, constructors),
# fields/properties assigned anywhere, constants, enums (explicit or implicit ordinals), config files
# (App.config, .settings, .ini, including reads such as `int.Parse(ini.Read("Sec", "Key"))`), and
# DllImport EntryPoint aliases. Advantech device names stored by the WinForms designer in .resx
# state streams are decoded too. Every value keeps its source file/line. A value that cannot be
# traced is reported as unresolved; several possible values are reported as a conflict. Nothing is
# guessed.
# ============================================================

SKIP_DIRS = frozenset({"bin", "obj", ".vs", ".git", "packages", "node_modules"})
CONFIG_SUFFIXES = (".config", ".ini", ".settings", ".json", ".xml")
MAX_FILE_BYTES = 4_000_000
MAX_DEPTH = 8
TRACE_STEP_BUDGET = 4000

STATUS_READY = "ready"
STATUS_PARTIAL = "partial"
STATUS_CONFLICT = "conflict"
STATUS_UNRESOLVED = "unresolved"
STATUS_INFO = "info"
STATUS_WARNING = "warning"
STATUS_LABELS = {
    STATUS_READY: "可套用",
    STATUS_PARTIAL: "可能只是預設值",
    STATUS_CONFLICT: "多處設定不同",
    STATUS_UNRESOLVED: "無法判定",
    STATUS_INFO: "參考",
    STATUS_WARNING: "注意",
}

# LSI8181_CI_mode_set(card, mode, debounce, rate): rate codes as in devices/lsi8181.py.
MULTIPLE_RATE_FROM_CODE = {0: MultipleRate.X4, 1: MultipleRate.X2, 2: MultipleRate.X1}
DEVICE_NAME = re.compile(r"(PCI[eE]?-\d{4}[A-Z]*(?:,BID#\d+)?)")


class LegacyImportError(ValueError):
    """The selected path cannot be scanned; the message is Traditional Chinese."""


@dataclass(frozen=True)
class SourceHit:
    file: str
    line: int
    method: str
    text: str

    def label(self) -> str:
        where = f"{self.file}:{self.line}"
        return f"{where}（{self.method}）" if self.method else where


@dataclass(frozen=True)
class ImportFinding:
    key: str
    label: str
    status: str
    value: object = None
    display: str = ""
    note: str = ""
    sources: tuple[SourceHit, ...] = ()

    @property
    def applicable(self) -> bool:
        """Can be applied; only `ready` findings are pre-selected in the import dialog."""
        return self.status in (STATUS_READY, STATUS_PARTIAL) and self.value is not None

    @property
    def preselected(self) -> bool:
        return self.status == STATUS_READY and self.value is not None


@dataclass(frozen=True)
class LegacyImportReport:
    root: str
    projects: tuple[str, ...]
    files_scanned: int
    findings: tuple[ImportFinding, ...] = field(default_factory=tuple)

    @property
    def applicable(self) -> tuple[ImportFinding, ...]:
        return tuple(f for f in self.findings if f.applicable)

    def finding(self, key: str) -> ImportFinding | None:
        return next((f for f in self.findings if f.key == key), None)


# ---- source collection ---------------------------------------------------------------------
_SLN_PROJECT = re.compile(r'^Project\("[^"]*"\)\s*=\s*"[^"]*"\s*,\s*"([^"]+\.(?:cs|vb)proj)"', re.MULTILINE)


def _read_text(path: Path) -> str:
    data = path.read_bytes()[:MAX_FILE_BYTES]
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    for encoding in ("utf-8-sig", "cp950"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def project_roots(selected: str | Path) -> tuple[Path, tuple[str, ...]]:
    """(folder shown as the report root, project directories to scan)."""
    path = Path(str(selected).strip().strip("\"'"))
    if not path.exists():
        raise LegacyImportError(f"找不到「{path}」。")
    if path.is_dir():
        return path, (str(path),)
    suffix = path.suffix.lower()
    if suffix in (".csproj", ".vbproj"):
        return path.parent, (str(path.parent),)
    if suffix != ".sln":
        raise LegacyImportError("請選擇原程式的 .sln、.csproj，或它的原始碼資料夾。")
    roots: list[str] = []
    for relative in _SLN_PROJECT.findall(_read_text(path)):
        project = (path.parent / relative.replace("\\", "/")).parent
        if project.is_dir() and str(project) not in roots:
            roots.append(str(project))
    return path.parent, tuple(roots) or (str(path.parent),)


def _walk(root: Path, in_output: bool = False) -> Iterable[tuple[Path, bool]]:
    """(file, inside bin/) pairs; bin/ is walked only for the settings the program saved at run time."""
    try:
        children = sorted(root.iterdir())
    except OSError:
        return
    for child in children:
        if child.is_dir():
            name = child.name.lower()
            if name == "bin":
                yield from _walk(child, True)
            elif name not in SKIP_DIRS:
                yield from _walk(child, in_output)
        else:
            yield child, in_output


def collect_files(roots: Iterable[str]) -> tuple[list[Path], list[Path], list[Path]]:
    """(C# sources, config files including run-time settings under bin/, .resx resources)."""
    code: list[Path] = []
    config: list[Path] = []
    resources: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for path, in_output in _walk(Path(root)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            suffix = path.suffix.lower()
            if in_output:
                if suffix in (".ini", ".config", ".settings", ".json") and not path.name.lower().endswith(
                    (".deps.json", ".runtimeconfig.json", ".exe.config.bak")
                ):
                    config.append(path)
                continue
            if suffix == ".cs":
                code.append(path)
            elif suffix == ".resx":
                resources.append(path)
            elif suffix in CONFIG_SUFFIXES:
                config.append(path)
    return code, config, resources


# ---- C# text helpers -----------------------------------------------------------------------
def strip_comments(text: str) -> str:
    """Blank out // and /* */ comments, keeping strings and every newline (line numbers stay valid)."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
        elif ch == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append("".join("\n" if c == "\n" else " " for c in text[i:end]))
            i = end
        elif ch == "@" and nxt == '"':
            j = i + 2
            while j < n:
                if text[j] == '"':
                    if j + 1 < n and text[j + 1] == '"':
                        j += 2
                        continue
                    break
                j += 1
            out.append(text[i : j + 1])
            i = j + 1
        elif ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def split_arguments(text: str, open_index: int) -> tuple[list[str], int] | None:
    """Top-level comma-separated arguments of the bracket group opening at `open_index`."""
    closing = {"(": ")", "[": "]", "{": "}"}[text[open_index]]
    depth, i, start = 0, open_index, open_index + 1
    args: list[str] = []
    in_string = False
    verbatim = False
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\" and not verbatim:
                i += 2
                continue
            if ch == '"':
                if verbatim and i + 1 < len(text) and text[i + 1] == '"':
                    i += 2
                    continue
                in_string = False
        elif ch == '"':
            in_string, verbatim = True, i > 0 and text[i - 1] == "@"
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                if ch != closing:
                    return None
                args.append(text[start:i].strip())
                return ([] if args == [""] else args), i
        elif ch == "," and depth == 1:
            args.append(text[start:i].strip())
            start = i + 1
        i += 1
    return None


_MODIFIERS = r"(?:(?:public|private|protected|internal|static|async|override|virtual|sealed|unsafe|extern|new|partial|abstract)\s+)"
_METHOD_HEADER = re.compile(
    r"^[ \t]*(?P<mods>" + _MODIFIERS + r"*)(?P<type>[\w<>\[\],.?]+\s+)?(?P<name>\w+)\s*\(", re.MULTILINE
)
_KEYWORDS = frozenset({"if", "for", "foreach", "while", "switch", "using", "lock", "catch", "return", "new", "else", "fixed"})
# A leading word that makes "word Name(" a statement, not a declaration.
_STATEMENT_WORDS = frozenset({"return", "await", "new", "throw", "yield", "else", "case", "goto", "var", "using", "in", "out", "ref"})


@dataclass(frozen=True)
class Method:
    name: str
    params: tuple[tuple[str, str], ...]  # (name, default expression or "")
    start: int  # index of the header in the file text
    body_end: int
    is_declaration_only: bool


@dataclass
class SourceFile:
    path: Path
    relative: str
    text: str
    lines: list[str]
    line_starts: list[int]
    methods: list[Method] = field(default_factory=list)
    method_starts: set[int] = field(default_factory=set)
    class_names: set[str] = field(default_factory=set)

    def line_of(self, index: int) -> int:
        lo, hi = 0, len(self.line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.line_starts[mid] <= index:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    def method_at(self, index: int) -> Method | None:
        best = None
        for method in self.methods:
            if method.start <= index <= method.body_end and (best is None or method.start > best.start):
                best = method
        return best


def _parse_params(text: str) -> tuple[tuple[str, str], ...]:
    params = []
    for raw in text:
        if not raw:
            continue
        default = ""
        head = raw
        if "=" in raw:
            head, default = raw.split("=", 1)
        tokens = re.findall(r"[\w<>\[\].?]+", head.replace("this ", ""))
        tokens = [t for t in tokens if t not in ("ref", "out", "in", "params")]
        if tokens:
            params.append((tokens[-1], default.strip()))
    return tuple(params)


def _index_methods(source: SourceFile) -> None:
    source.class_names = set(re.findall(r"\b(?:class|struct)\s+(\w+)", source.text))
    for match in _METHOD_HEADER.finditer(source.text):
        name = match.group("name")
        declared_type = (match.group("type") or "").strip()
        if name in _KEYWORDS or declared_type in _STATEMENT_WORDS:
            continue
        # Without modifiers or a return type this is a call statement, unless it is a constructor.
        if not match.group("mods") and not declared_type and name not in source.class_names:
            continue
        parsed = split_arguments(source.text, match.end() - 1)
        if parsed is None:
            continue
        args, close = parsed
        rest = source.text[close + 1 : close + 400]
        body = re.match(r"\s*(?::\s*(?:base|this)\s*\([^)]*\)\s*)?(\{|=>|;)", rest)
        if not body:
            continue
        # A header is a declaration: every parameter is "type name" (a call passes expressions).
        if args and not all(re.match(r"^(?:\[[^\]]*\]\s*)?(?:(?:this|ref|out|in|params)\s+)?[\w<>\[\],.?]+\s+\w+(\s*=.*)?$", a) for a in args):
            continue
        token = body.group(1)
        if token == "{":
            open_index = close + 1 + body.end() - 1
            block = split_arguments(source.text, open_index)
            end = block[1] if block else len(source.text)
        elif token == "=>":
            end = source.text.find(";", close)
            end = len(source.text) if end < 0 else end
        else:
            end = close + 1 + body.end()
        source.methods.append(Method(name, _parse_params(args), match.start("name"), end, token == ";"))
        source.method_starts.add(match.start("name"))


def _load(paths: list[Path], root: Path) -> list[SourceFile]:
    files = []
    for path in paths:
        try:
            text = strip_comments(_read_text(path))
        except OSError:
            continue
        starts = [0]
        starts.extend(m.end() for m in re.finditer("\n", text))
        try:
            relative = str(path.relative_to(root))
        except ValueError:
            relative = str(path)
        source = SourceFile(path, relative, text, text.split("\n"), starts)
        _index_methods(source)
        files.append(source)
    return files


@dataclass(frozen=True)
class Call:
    file: SourceFile
    index: int  # position of the call name in the file text
    args: tuple[str, ...]

    @property
    def line(self) -> int:
        return self.file.line_of(self.index)

    def hit(self) -> SourceHit:
        method = self.file.method_at(self.index)
        text = self.file.lines[self.line - 1].strip()
        return SourceHit(self.file.relative, self.line, method.name if method else "", text[:160])


def _is_declaration(source: SourceFile, index: int) -> bool:
    line_start = source.text.rfind("\n", 0, index) + 1
    prefix = source.text[line_start:index]
    if re.search(r"\b(extern|delegate)\b", prefix):
        return True
    return index in source.method_starts


def find_calls(files: list[SourceFile], names: Iterable[str], member: bool = False) -> list[Call]:
    """Calls to any of `names`; `member=True` requires a preceding '.' (instance/static member call)."""
    alternatives = "|".join(re.escape(n) for n in names)
    regex = re.compile((r"\.\s*" if member else r"(?<![\w.])(?:[\w.]+\.)?") + rf"({alternatives})\s*\(")
    calls = []
    for source in files:
        for match in regex.finditer(source.text):
            index = match.start(1)
            if _is_declaration(source, index):
                continue
            parsed = split_arguments(source.text, match.end() - 1)
            if parsed is not None:
                calls.append(Call(source, index, tuple(parsed[0])))
    return calls


def dll_aliases(files: list[SourceFile], exported: str) -> set[str]:
    """C# names bound to `exported` through [DllImport(..., EntryPoint = "exported")]."""
    names = {exported}
    pattern = re.compile(
        rf'EntryPoint\s*=\s*"{re.escape(exported)}"[^\]]*\]\s*(?:\[[^\]]*\]\s*)*' + _MODIFIERS + r"*[\w<>\[\].]+\s+(\w+)\s*\("
    )
    for source in files:
        names.update(pattern.findall(source.text))
    return names


# ---- value resolution ----------------------------------------------------------------------
_CAST = re.compile(
    r"^\(\s*(?:byte|sbyte|short|ushort|int|uint|long|ulong|float|double|decimal|bool|string|"
    r"Byte|UInt16|Int16|Int32|UInt32|Double|Single|Boolean|String)\s*\)\s*"
)
_NUMBER = re.compile(r"^[-+]?(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?)(?:[uUlLfFdDmM]{0,2})$")
_STRING = re.compile(r'^\$?@?"((?:[^"\\]|\\.|"")*)"$')
_CONVERSION = re.compile(
    r"^(?:[\w.]+\.)?(?:Parse|ToInt16|ToInt32|ToInt64|ToUInt16|ToUInt32|ToByte|ToDouble|ToSingle|ToDecimal|ToBoolean|ToString|GetBytes)\s*\("
)
_ASSIGNMENT_START = re.compile(r"(?<![\w.])((?:\w+\.)*)([A-Za-z_]\w*)\s*=(?![=>])\s*")
_BYTE_ARRAY = re.compile(r"^new\s+byte\s*\[\s*\w*\s*\]\s*\{([^{}]*)\}$", re.IGNORECASE)
# WinForms control properties hold operator input: always a run-time value.
_UI_PROPERTIES = frozenset({"SelectedIndex", "SelectedItem", "SelectedValue", "Value", "Text", "Checked", "CheckState"})


def _owner(chain: str) -> str:
    """Normalized owner of a member chain: `_settings.X` and `this.settings.X` both give `settings`."""
    parts = [p for p in chain.split(".") if p and p not in ("this", "base")]
    return parts[-2].lstrip("_").lower() if len(parts) >= 2 else ""


def _expression_at(text: str, start: int) -> str:
    """The assigned expression starting at `start`, up to a top-level ';', ',' or closing bracket."""
    depth, i = 0, start
    in_string = False
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif ch in ";," and depth == 0:
            break
        elif ch == "\n" and depth == 0 and text[start:i].strip().endswith(("{", ")")):
            break
        i += 1
    return text[start:i]
_PROPERTY_INIT = re.compile(r"\b(\w+)\s*\{\s*get;[^}]*\}\s*=\s*([^;]+);")
_PROPERTY_ARROW = re.compile(r"\b(\w+)\s*=>\s*([^;]+);")
_PROPERTY_GET_RETURN = re.compile(r"\b(\w+)\s*\{\s*get\s*\{\s*return\s+([^;]+);")
_ENUM = re.compile(r"\benum\s+(\w+)\s*(?::\s*\w+\s*)?\{([^}]*)\}")


def _split_ternary(text: str) -> tuple[str, str] | None:
    """(when-true, when-false) of a top-level `a ? b : c`, ignoring `?.`, `??` and nested brackets."""
    depth, question, in_string = 0, -1, False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            in_string = ch != '"'
        elif ch == '"':
            in_string = True
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and ch == "?" and question < 0:
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt in ".?" or (i > 0 and text[i - 1] == "?"):
                i += 2 if nxt in ".?" else 1
                continue
            question = i
        elif depth == 0 and ch == ":" and question >= 0:
            return text[question + 1 : i].strip(), text[i + 1 :].strip()
        i += 1
    return None


def _literal(text: str):
    if text in ("true", "false"):
        return text == "true"
    if _NUMBER.match(text):
        body = text.rstrip("uUlLfFdDmM")
        if body.lower().lstrip("+-").startswith("0x"):
            return int(body, 16)
        return float(body) if "." in body else int(body)
    match = _STRING.match(text)
    if match and "{" not in match.group(1):
        raw = match.group(1)
        if text.lstrip("$").startswith("@"):
            return raw.replace('""', '"')
        return _CS_ESCAPE.sub(_unescape_cs, raw)
    return None


_CS_ESCAPE = re.compile(r"\\(u[0-9a-fA-F]{4}|x[0-9a-fA-F]{1,4}|.)")
_CS_SIMPLE = {"r": "\r", "n": "\n", "t": "\t", "0": "\0", "a": "\a", "b": "\b", "f": "\f", "v": "\v"}


def _unescape_cs(match: re.Match) -> str:
    code = match.group(1)
    if code[0] in "ux" and len(code) > 1:
        return chr(int(code[1:], 16))
    return _CS_SIMPLE.get(code, code)


def _config_literal(text: str):
    value = text.strip().strip('"')
    parsed = _literal(value) if value else None
    if parsed is not None:
        return parsed
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    return value or None


def read_config_values(paths: list[Path]) -> dict[str, set]:
    """key -> values from *.ini (k=v), App.config <add key value>, and .settings/<setting> values."""
    values: dict[str, set] = {}
    xml_patterns = (
        re.compile(r'<add\s+key="([^"]+)"\s+value="([^"]*)"', re.IGNORECASE),
        re.compile(r'<Setting\s+Name="([^"]+)"[^>]*>\s*<Value[^>]*>([^<]*)</Value>', re.IGNORECASE),
        re.compile(r'<setting\s+name="([^"]+)"[^>]*>\s*<value>([^<]*)</value>', re.IGNORECASE),
        re.compile(r'"([A-Za-z_]\w*)"\s*:\s*("?[^",\r\n}]*"?)'),
    )
    ini_pattern = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*=\s*([^;\r\n]*?)\s*$", re.MULTILINE)
    for path in paths:
        try:
            text = _read_text(path)
        except OSError:
            continue
        regexes = (ini_pattern,) if path.suffix.lower() == ".ini" else xml_patterns
        for regex in regexes:
            for key, raw in regex.findall(text):
                value = _config_literal(raw)
                if value is not None:
                    values.setdefault(key, set()).add(value)
    return values


class Resolver:
    """Resolve C# expressions to the set of constant values they can take (empty set = unknown)."""

    def __init__(self, files: list[SourceFile], config: dict[str, set]):
        self.files = files
        self.config = config
        self.symbols: dict[str, list[tuple[str, SourceFile, int, str]]] = {}
        self.enum_members: dict[str, set[int]] = {}
        self._callers: dict[str, list[Call]] = {}
        self._memo: dict = {}
        self._steps = 0
        self._exhausted = False
        self._caller_arguments: dict = {}
        self._methods_by_name: dict[str, list[tuple[SourceFile, Method]]] = {}
        for source in files:
            for method in source.methods:
                if not method.is_declaration_only:
                    self._methods_by_name.setdefault(method.name, []).append((source, method))
        for source in files:
            for match in _ASSIGNMENT_START.finditer(source.text):
                expression = _expression_at(source.text, match.end())
                self._add_symbol(match.group(2), expression, source, match.end(), _owner(match.group(1) + match.group(2)))
            for regex in (_PROPERTY_INIT, _PROPERTY_ARROW, _PROPERTY_GET_RETURN):
                for match in regex.finditer(source.text):
                    self._add_symbol(match.group(1), match.group(2), source, match.start(2))
            for enum_match in _ENUM.finditer(source.text):
                ordinal = -1
                for member in (m.strip() for m in enum_match.group(2).split(",")):
                    if not member:
                        continue
                    member = re.sub(r"^\[[^\]]*\]\s*", "", member)
                    if "=" in member:
                        name, value = (p.strip() for p in member.split("=", 1))
                        parsed = _literal(value)
                        ordinal = parsed if isinstance(parsed, int) else ordinal + 1
                    else:
                        name, ordinal = member, ordinal + 1
                    self.enum_members.setdefault(name, set()).add(ordinal)

    def _add_symbol(self, name: str, expression: str, source: SourceFile, index: int, owner: str = "") -> None:
        expression = expression.strip()
        if not expression or name in _KEYWORDS or expression.startswith("{"):
            return
        if expression.startswith("new ") and not _BYTE_ARRAY.match(expression):
            return
        self.symbols.setdefault(name, []).append((expression, source, index, owner))

    # -- callers of a method (for parameter tracing) --------------------------------------
    def callers(self, method: Method, source: SourceFile) -> list[Call]:
        """Calls (including `new Class(...)`) whose argument count fits this overload."""
        if method.name not in self._callers:
            self._callers[method.name] = find_calls(self.files, [method.name])
        required = sum(1 for _name, default in method.params if not default)
        return [c for c in self._callers[method.name] if required <= len(c.args) <= len(method.params)]

    def trace(self, expression: str, source: SourceFile, index: int) -> tuple[set, tuple[str, ...]]:
        """(possible values, runtime-only sources met on the way such as UI input or file reads)."""
        self._gaps: list[str] = []
        self._steps = 0
        values = self.values(expression, source, index)
        return values, tuple(dict.fromkeys(self._gaps))

    def _gap(self, text: str) -> set:
        gaps = getattr(self, "_gaps", None)
        if gaps is not None and len(gaps) < 20:
            gaps.append(text[:80])
        return set()

    def values(self, expression: str, source: SourceFile, index: int, depth: int = 0, seen=None, bindings=None) -> set:
        if depth == 0:
            self._steps = 0
            self._exhausted = False
        return self._lookup(expression, source, index, depth, seen, bindings)

    def _lookup(self, expression: str, source: SourceFile, index: int, depth: int, seen, bindings) -> set:
        if self._exhausted:
            return set()
        seen = set() if seen is None else seen
        text = expression.strip()
        if not text or depth > MAX_DEPTH or (text, id(source), index) in seen:
            return set()
        # Common names (value, index…) are assigned in hundreds of places; without a memo and a step
        # budget, tracing them grows exponentially on a real program and freezes the scan.
        bound = tuple(sorted((k, v[0], id(v[1]), v[2]) for k, v in (bindings or {}).items()))
        # Depth is part of the key: a result cut short by MAX_DEPTH must not be reused nearer the root.
        key = (text, id(source), index, bound, depth)
        cached = self._memo.get(key)
        if cached is not None:
            self._steps += 1
            if self._steps > TRACE_STEP_BUDGET:
                self._exhausted = True
                return set()
            result, gaps = cached
            for gap in gaps:
                self._gap(gap)
            return set(result)
        # Each top-level lookup gets its own budget; nested lookups share it. Past the budget no new
        # expression is expanded, but values already found are kept (and flagged by the gap).
        self._steps += 1
        if self._steps > TRACE_STEP_BUDGET:
            self._exhausted = True
            self._gap("（追蹤太複雜，已停止）")
            return set()
        outer_gaps = getattr(self, "_gaps", None)
        self._gaps = []
        seen = seen | {(text, id(source), index)}
        if re.match(r"^(?:out|ref)\s", text):
            result = self._gap(text)
        else:
            text_in = re.sub(r"^in\s+", "", text)
            result = self._values(text_in, source, index, depth, seen, bindings or {})
            if not result:
                self._gap(text_in)
        own_gaps = tuple(self._gaps)
        # A budget-limited result is incomplete. A later, independent trace must be able to
        # explore this expression again rather than inheriting a truncated answer.
        if not self._exhausted:
            self._memo[key] = (frozenset(result), own_gaps)
        if outer_gaps is not None:
            outer_gaps.extend(g for g in own_gaps if len(outer_gaps) < 20)
        self._gaps = outer_gaps
        return set(result)

    def _values(self, text: str, source: SourceFile, index: int, depth: int, seen, bindings: dict) -> set:
        def again(expression, where=source, at=index, bound=bindings):
            return self.values(expression, where, at, depth + 1, seen, bound)

        # Peel casts, parentheses and conversions such as int.Parse(x) / Convert.ToInt32(x).
        while True:
            before = text
            text = _CAST.sub("", text).strip()
            if text.startswith("(") and split_arguments(text, 0) == ([text[1:-1].strip()], len(text) - 1):
                text = text[1:-1].strip()
            conversion = _CONVERSION.match(text)
            if conversion:
                parsed = split_arguments(text, conversion.end() - 1)
                if parsed and parsed[1] == len(text) - 1 and parsed[0]:
                    text = parsed[0][0]
            text = re.sub(r"\.Value$", "", text)
            if text == before:
                break
        literal = _literal(text)
        if literal is not None:
            return {literal}
        byte_array = _BYTE_ARRAY.match(text)
        if byte_array:
            items = [self.values(item, source, index, depth + 1, seen) for item in byte_array.group(1).split(",") if item.strip()]
            if items and all(len(v) == 1 and isinstance(next(iter(v)), int) for v in items):
                return {bytes(next(iter(v)) & 0xFF for v in items)}
            return set()
        if "??" in text:
            left, right = text.split("??", 1)
            return again(left) | again(right)
        ternary = _split_ternary(text)
        if ternary:
            return again(ternary[0]) | again(ternary[1])
        config_keys = [k for k in re.findall(r'"(\w[\w.]*)"', text) if k in self.config]
        if config_keys:
            return set(self.config[config_keys[-1]])
        call = re.match(r"^(?:new\s+)?(?:[\w.]+\.)?(\w+)\s*\(", text)
        if call and call.group(1) not in _KEYWORDS:
            parsed = split_arguments(text, call.end() - 1)
            if parsed and parsed[1] == len(text) - 1 and parsed[0]:
                return self._return_values(call.group(1), tuple(parsed[0]), source, index, depth, seen, bindings)
        chain = re.fullmatch(r"(?:this\.|base\.)?([\w.]+?)(?:\(\s*\))?", text)
        if not chain:
            return set()
        name = chain.group(1).split(".")[-1]
        results: set = set()
        simple = "." not in chain.group(1)
        if simple and name in bindings:
            expression_text, where, at = bindings[name]
            return again(expression_text, where, at, {})
        method = source.method_at(index)
        if method is not None and simple:
            for position, (param, default) in enumerate(method.params):
                if param != name:
                    continue
                for argument, caller in self._arguments_for(method, source, position, param, default):
                    if self._exhausted:
                        break
                    if argument:
                        results |= again(argument, caller.file, caller.index, {})
                if results:
                    return results
        if not simple and name in _UI_PROPERTIES:
            # `comboRate.SelectedIndex`, `numericIncrement.Value`: whatever the operator chose at run time.
            self._gap(text)
            return set()
        candidates = list(self.symbols.get(name, ()))
        if simple and method is not None:
            # A name assigned inside the same method is a local: ignore same-named symbols elsewhere.
            local = [c for c in candidates if c[1] is source and method.start <= c[2] <= method.body_end and c[2] < index]
            members = [c for c in candidates if c[3] in ("", "this")]
            # An unqualified field belongs to its own class: prefer assignments in the same file.
            candidates = local or [c for c in members if c[1] is source] or members
        elif not simple:
            # `_settings.X` only follows `settings.X = …` or an unqualified member `X = …` (object initializer).
            owner = _owner(chain.group(1))
            candidates = [c for c in candidates if c[3] in ("", "this", owner)]
        for expression_text, where, at, _owner_name in candidates:
            if self._exhausted:
                break
            results |= again(expression_text, where, at)
        if name in self.enum_members:
            results |= self.enum_members[name]
        if not results and name in self.config:
            results |= self.config[name]
        if not results and not candidates and name not in self.enum_members:
            self._gap(text)
        return results

    def _return_values(self, name, args, source, index, depth, seen, bindings) -> set:
        """Values returned by a called method, with its parameters bound to this call's arguments."""
        results: set = set()
        for method_source, method in self._methods_by_name.get(name, ()):
            if self._exhausted:
                break
            required = sum(1 for _n, default in method.params if not default)
            if not required <= len(args) <= len(method.params):
                continue
            bound = {}
            for position, (param, default) in enumerate(method.params):
                argument = self._argument(args, position, param, "")
                if argument:
                    bound[param] = (argument, source, index)
                elif default:
                    bound[param] = (default, method_source, method.start)
            for expression_text, at in self._returns(method_source, method, bound, depth, seen):
                if self._exhausted:
                    break
                results |= self.values(expression_text, method_source, at, depth + 1, seen, bound)
        return results

    def _returns(self, source: SourceFile, method: Method, bound: dict, depth, seen) -> list[tuple[str, int]]:
        body = source.text[method.start : method.body_end + 1]
        arrow = re.search(r"\)\s*=>\s*", body)
        if arrow and "{" not in body[: arrow.start()].split("(", 1)[-1]:
            return [(body[arrow.end() :].rstrip(";").strip(), method.start + arrow.end())]
        found = []
        for match in re.finditer(r"\breturn\s+", body):
            start = method.start + match.end()
            expression = _expression_at(source.text, start)
            found.append((expression, start, self._case_label(source.text, method.start, start)))
        switch = re.search(r"\bswitch\s*\(\s*(\w+)\s*\)", body)
        if switch and switch.group(1) in bound and any(label for _e, _s, label in found):
            subject_expression, where, at = bound[switch.group(1)]
            subject = self.values(subject_expression, where, at, depth + 1, seen)
            if len(subject) == 1:
                value = next(iter(subject))
                chosen = [(e, s) for e, s, label in found if label and label != "default" and value in self.values(label, source, s, depth + 1, seen)]
                if not chosen:
                    chosen = [(e, s) for e, s, label in found if label == "default"]
                if chosen:
                    return chosen
        return [(e, s) for e, s, _label in found]

    @staticmethod
    def _case_label(text: str, method_start: int, position: int) -> str:
        segment = text[method_start:position]
        labels = list(re.finditer(r"\bcase\s+([^:]+?)\s*:|\bdefault\s*:", segment))
        if not labels:
            return ""
        last = labels[-1]
        return last.group(1) if last.group(1) else "default"

    def _arguments_for(self, method: Method, source: SourceFile, position: int, param: str, default: str):
        """(argument expression, call) for one parameter over every caller, literals first; cached."""
        key = (id(source), method.start, position)
        if key not in self._caller_arguments:
            pairs = [(self._argument(c.args, position, param, default), c) for c in self.callers(method, source)]
            pairs.sort(key=lambda item: _literal(item[0].strip()) is None)
            self._caller_arguments[key] = pairs
        return self._caller_arguments[key]

    @staticmethod
    def _argument(args: tuple[str, ...], position: int, param: str, default: str) -> str:
        for arg in args:
            named = re.match(rf"^{re.escape(param)}\s*:\s*(.+)$", arg)
            if named:
                return named.group(1)
        positional = [a for a in args if not re.match(r"^\w+\s*:(?!:)", a)]
        return positional[position] if position < len(positional) else default


# ---- findings ------------------------------------------------------------------------------
class Collected(dict):
    """{value: [hits]} plus the runtime-only sources met while tracing (`gaps`)."""

    gaps: tuple[str, ...] = ()


def _collect(calls: list[Call], index: int, resolver: Resolver) -> tuple[Collected, list[SourceHit]]:
    """{value: [hits]} for argument `index`, plus hits whose argument could not be traced."""
    resolved = Collected()
    unresolved: list[SourceHit] = []
    gaps: list[str] = []
    for call in calls:
        if index >= len(call.args):
            continue
        values, call_gaps = resolver.trace(call.args[index], call.file, call.index)
        if not values:
            unresolved.append(call.hit())
        elif call_gaps:
            gaps.extend(call_gaps)
        for value in values:
            resolved.setdefault(value, []).append(call.hit())
    resolved.gaps = tuple(dict.fromkeys(gaps))
    return resolved, unresolved


def _finding(key, label, resolved, unresolved, convert=lambda v: v, show=str, note="") -> ImportFinding | None:
    if not resolved and not unresolved:
        return None
    gaps = getattr(resolved, "gaps", ())
    hits = tuple(dict.fromkeys([h for group in resolved.values() for h in group] + list(unresolved)))
    converted: dict = {}
    for raw in resolved:
        try:
            converted.setdefault(convert(raw), raw)
        except (KeyError, TypeError, ValueError):
            converted.setdefault(("?", raw), raw)
    if not converted:
        expressions = "；".join(h.text for h in unresolved[:3])
        return ImportFinding(key, label, STATUS_UNRESOLVED, None, "（追不到固定數值）", f"請手動查看：{expressions}", hits)
    if len(converted) > 1:
        shown = "、".join(show(v) if not isinstance(v, tuple) else str(v[1]) for v in converted)
        return ImportFinding(key, label, STATUS_CONFLICT, None, shown, "原程式在不同地方用了不同的值，請確認實際使用哪一個。", hits)
    value = next(iter(converted))
    if isinstance(value, tuple):
        return ImportFinding(key, label, STATUS_UNRESOLVED, None, str(value[1]), "這個值 VisionFlow 無法對應。", hits)
    extra = f"另有 {len(unresolved)} 處追不到固定數值，請確認。" if unresolved else ""
    if gaps:
        runtime = "；".join(gaps[:3])
        text = f"值可能在執行時才決定（例如畫面輸入或讀設定檔：{runtime}），{show(value)} 可能只是預設值；確認後再套用。"
        return ImportFinding(key, label, STATUS_PARTIAL, value, show(value), " ".join(x for x in (note, text, extra) if x), hits)
    return ImportFinding(key, label, STATUS_READY, value, show(value), " ".join(x for x in (note, extra) if x), hits)


def _warning(key, label, text, calls: list[Call]) -> ImportFinding:
    return ImportFinding(key, label, STATUS_WARNING, None, text, "", tuple(c.hit() for c in calls))


def _bool_text(value: bool) -> str:
    return "是" if value else "否"


def _single(resolver: Resolver, call: Call, position: int):
    if position >= len(call.args):
        return None
    values = resolver.values(call.args[position], call.file, call.index)
    return next(iter(values)) if len(values) == 1 else None


class LegacyProgramAnalyzer:
    def __init__(self, files: list[SourceFile], resolver: Resolver, resources: list[Path], root: Path):
        self.files = files
        self.resolver = resolver
        self.resources = resources
        self.root = root

    def lsi(self, exported: str) -> list[Call]:
        return find_calls(self.files, dll_aliases(self.files, exported))

    def run(self) -> list[ImportFinding]:
        found: list[ImportFinding | None] = []
        found += self._meter_wheel()
        found += self._sensor_relay()
        found += self._sapera()
        found += self._light()
        return [f for f in found if f is not None]

    # --- LSI-8181 meter wheel ---------------------------------------------------------
    def _meter_wheel(self) -> list[ImportFinding | None]:
        r = self.resolver
        increment = self.lsi("LSI8181_compare_increment_set")
        ci_mode = self.lsi("LSI8181_CI_mode_set")
        cmp_out = self.lsi("LSI8181_compare_CMP_OUT_set")
        polarity = self.lsi("LSI8181_CIO_polarity_set")
        compare_mode = self.lsi("LSI8181_compare_mode_set")
        card_calls = increment + ci_mode + cmp_out + self.lsi("LSI8181_counter_set") + self.lsi("LSI8181_compare_value_set")
        out: list[ImportFinding | None] = [
            _finding("meter_wheel.card_id", "米輪卡片 ID", *_collect(card_calls, 0, r), convert=int),
            _finding("meter_wheel.compare_increment", "自動遞增（幾格拍一行）", *_collect(increment, 1, r), convert=int),
            _finding(
                "meter_wheel.multiple_rate",
                "倍頻",
                *_collect(ci_mode, 3, r),
                convert=lambda v: MULTIPLE_RATE_FROM_CODE[int(v)],
                show=lambda rate: rate.name,
            ),
            _finding("meter_wheel.cmp_out_width", "CMP Out Width", *_collect(cmp_out, 3, r), convert=int),
            _finding(
                "meter_wheel.reverse_direction",
                "反向計數",
                *_collect(polarity, 1, r),
                convert=lambda v: bool(int(v) & 1),
                show=_bool_text,
                note="依 CIO 極性 bit 0（A 相）判斷。",
            ),
        ]
        for call in ci_mode:
            mode, debounce = _single(r, call, 1), _single(r, call, 2)
            if (mode is not None and mode != 0) or (debounce is not None and debounce != 1):
                out.append(_warning("warn.ci_mode", "米輪計數模式", f"原程式計數模式 {mode}、防抖 {debounce}；VisionFlow 固定為 0（正交）與 1（1 µs）。", [call]))
                break
        for call in cmp_out:
            pol, mode = _single(r, call, 1), _single(r, call, 2)
            if (pol is not None and pol != 0) or (mode is not None and mode != 1):
                out.append(_warning("warn.cmp_out", "CMP OUT 輸出方式", f"原程式 CMP OUT 極性 {pol}、輸出模式 {mode}；VisionFlow 固定為 0 與 1（脈衝）。", [call]))
                break
        for call in compare_mode:
            mode = _single(r, call, 1)
            if mode is not None and mode != 2:
                out.append(_warning("warn.compare_mode", "Compare 模式", f"原程式 Compare 模式為 {mode}；VisionFlow 固定為 2（自動遞增）。", [call]))
                break
        out.append(self._timing("info.compare_value", "Compare 寫入時機", self.lsi("LSI8181_compare_value_set")))
        out.append(self._timing("info.encoder_value", "Encoder 寫入時機", self.lsi("LSI8181_counter_set")))
        return out

    def _timing(self, key: str, label: str, calls: list[Call]) -> ImportFinding | None:
        if not calls:
            return None
        parts = []
        for call in calls[:6]:
            hit = call.hit()
            values = self.resolver.values(call.args[1], call.file, call.index) if len(call.args) > 1 else set()
            shown = "、".join(str(v) for v in sorted(values, key=str)) if values else (call.args[1] if len(call.args) > 1 else "?")
            parts.append(f"{hit.method or hit.file} 寫入 {shown}")
        note = "若是在觸發或開始取像的方法裡寫入，對應 VisionFlow「外部觸發時自動寫入已存 Compare／Encoder 值」。"
        return ImportFinding(key, label, STATUS_INFO, None, "；".join(parts), note, tuple(c.hit() for c in calls))

    # --- PCIe-1730 Sensor relay -------------------------------------------------------
    def _sensor_relay(self) -> list[ImportFinding | None]:
        r = self.resolver
        devices = [c for c in find_calls(self.files, ["DeviceInformation"]) if "new" in c.file.text[max(0, c.index - 12) : c.index]]
        device = _finding("sensor_relay.device", "I/O 卡裝置", *_collect(devices, 0, r), convert=_device_name)
        if device is None or device.status != STATUS_READY:
            device = self._designer_device() or device
        reads = find_calls(self.files, ["ReadBit"], member=True)
        writes = find_calls(self.files, ["WriteBit"], member=True)
        out: list[ImportFinding | None] = [
            device,
            _finding("sensor_relay.di_port", "Sensor DI port", *_collect(reads, 0, r), convert=int),
            _finding("sensor_relay.di_bit", "Sensor DI bit", *_collect(reads, 1, r), convert=int),
            _finding("sensor_relay.do_port", "擷取卡 DO port", *_collect(writes, 0, r), convert=int),
            _finding("sensor_relay.do_bit", "擷取卡 DO bit", *_collect(writes, 1, r), convert=int),
        ]
        out += self._do_pulse(writes)
        if not reads:
            whole = [c for c in find_calls(self.files, ["Read"], member=True) if re.search(r"\bInstantDi", c.file.text)]
            if whole:
                out.append(_warning("warn.di_read", "Sensor DI 讀取方式", "原程式整個 port 一起讀（Read），bit 要看後面的位元判斷，請手動確認 DI port／bit。", whole[:3]))
        interrupts = find_calls(self.files, ["SnapStart"], member=True) + [
            Call(s, m.start(), ()) for s in self.files for m in re.finditer(r"\bDiintChannels\b|\.Interrupt\s*\+=", s.text)
        ]
        if interrupts:
            out.append(_warning("warn.di_interrupt", "Sensor 偵測方式", "原程式用 DI 中斷偵測 Sensor；VisionFlow 用定時讀取，延遲約為「DI 讀取間隔」。", interrupts[:3]))
        return out

    def _designer_device(self) -> ImportFinding | None:
        """Device names the WinForms designer serialized into .resx state streams (base64)."""
        names: dict[str, list[SourceHit]] = {}
        for path in self.resources:
            try:
                text = _read_text(path)
            except OSError:
                continue
            relative = str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)
            for blob in re.findall(r"<value>\s*([A-Za-z0-9+/=\s]{40,})\s*</value>", text):
                try:
                    data = base64.b64decode(re.sub(r"\s+", "", blob), validate=False)
                except ValueError:
                    continue
                for decoded in (data.decode("latin-1"), data.decode("utf-16-le", errors="ignore")):
                    for name in DEVICE_NAME.findall(decoded):
                        names.setdefault(name, []).append(SourceHit(relative, 0, "", "表單設計工具儲存的裝置設定"))
        if not names:
            return None
        return _finding("sensor_relay.device", "I/O 卡裝置", names, [], convert=_device_name, note="從表單資源（.resx）解出。")

    def _do_pulse(self, writes: list[Call]) -> list[ImportFinding | None]:
        """Thread.Sleep/Task.Delay between WriteBit calls of one method: pulse width and active level."""
        by_method: dict[tuple[int, int], list[Call]] = {}
        for call in writes:
            method = call.file.method_at(call.index)
            by_method.setdefault((id(call.file), method.start if method else -1), []).append(call)
        widths: dict = {}
        width_unresolved: list[SourceHit] = []
        levels: dict = {}
        for calls in by_method.values():
            if len(calls) < 2:
                continue
            calls.sort(key=lambda c: c.index)
            first, last = calls[0], calls[-1]
            for sleep in find_calls([first.file], ["Sleep", "Delay"]):
                if first.index < sleep.index < last.index and sleep.args:
                    values = self.resolver.values(sleep.args[0], sleep.file, sleep.index)
                    if not values:
                        width_unresolved.append(sleep.hit())
                    for value in values:
                        widths.setdefault(value, []).append(sleep.hit())
            start_level, end_level = _single(self.resolver, first, 2), _single(self.resolver, last, 2)
            if start_level is not None and end_level is not None and bool(start_level) != bool(end_level):
                levels.setdefault(not bool(start_level), []).append(first.hit())
        return [
            _finding("sensor_relay.pulse_ms", "DO 脈寬（ms）", widths, width_unresolved, convert=float, show=lambda v: f"{v:g}"),
            _finding(
                "sensor_relay.do_active_low",
                "DO 低電位有效",
                levels,
                [],
                convert=bool,
                show=_bool_text,
                note="依原程式先寫 1 再寫 0（高電位有效）或相反判斷。",
            ),
        ]

    # --- Sapera ----------------------------------------------------------------------
    def _sapera(self) -> list[ImportFinding | None]:
        r = self.resolver
        ccf: dict = {}
        for source in self.files:
            for match in re.finditer(r'@?"([^"\n]*\.ccf)"', source.text, re.IGNORECASE):
                value = _literal(match.group(0))
                if value and not any(c in value for c in "*?"):
                    ccf.setdefault(value, []).append(Call(source, match.start(), ()).hit())
        for key, values in r.config.items():
            for value in values:
                if isinstance(value, str) and value.lower().endswith(".ccf"):
                    ccf.setdefault(value, []).append(SourceHit("設定檔", 0, key, f"{key}={value}"))
        parameters = find_calls(self.files, ["SetParameter"], member=True)
        crop = [c for c in parameters if c.args and "CROP_HEIGHT" in c.args[0]]
        features = find_calls(self.files, ["SetFeatureValue"], member=True)
        exposure = [c for c in features if c.args and re.search(r'"ExposureTime(?:Abs)?"', c.args[0])]
        gain = [c for c in features if c.args and re.search(r'"Gain(?:Raw|Abs)?"', c.args[0])]
        out: list[ImportFinding | None] = [
            _finding("connection.config_file_path", "CCF 檔案", ccf, [], convert=str,
                     note="若只有檔名，請確認相機機台上的完整位置。"),
            _finding("acquisition.length_lines", "影像長度（線）", *_collect(crop, 1, r), convert=int),
            _finding("acquisition.exposure_time", "曝光時間", *_collect(exposure, 1, r), convert=float, show=lambda v: f"{v:g}"),
            _finding("acquisition.gain", "增益", *_collect(gain, 1, r), convert=float, show=lambda v: f"{v:g}"),
        ]
        shaft = [c for c in parameters if c.args and "SHAFT_ENCODER" in c.args[0]]
        if shaft:
            text = "；".join(
                f"{c.args[0].split('.')[-1]} = {_single(r, c, 1) if _single(r, c, 1) is not None else (c.args[1] if len(c.args) > 1 else '?')}"
                for c in shaft[:4]
            )
            out.append(
                _warning("warn.shaft_encoder", "擷取卡 Shaft Encoder", f"原程式在擷取卡做了除頻／倍頻（{text}），影像比例也受它影響；VisionFlow 請用同一份 CCF。", shaft[:4])
            )
        return out


    # --- RS-232 light (System.IO.Ports.SerialPort) ------------------------------------
    def _light(self) -> list[ImportFinding | None]:
        names: set[str] = set()
        for source in self.files:
            names.update(re.findall(r"\bSerialPort\s+(\w+)\s*[=;]", source.text))
            names.update(re.findall(r"\b(\w+)\s*=\s*new\s+SerialPort\b", source.text))
        constructors = find_calls(self.files, ["SerialPort"])
        constructors = [c for c in constructors if "new" in c.file.text[max(0, c.index - 12) : c.index] and c.args]
        if not names and not constructors:
            return []
        r = self.resolver
        out: list[ImportFinding | None] = []

        properties: dict[str, list[Call]] = {}
        for source in self.files:
            for name in names:
                pattern = rf"\b(?:this\.)?{re.escape(name)}\.(PortName|BaudRate|Parity|DataBits|StopBits|NewLine)\s*=(?!=)\s*"
                for match in re.finditer(pattern, source.text):
                    expression = _expression_at(source.text, match.end())
                    properties.setdefault(match.group(1), []).append(Call(source, match.end(), (expression,)))
        ctor_positions = {"PortName": 0, "BaudRate": 1, "Parity": 2, "DataBits": 3, "StopBits": 4}
        for prop, position in ctor_positions.items():
            for call in constructors:
                if position < len(call.args):
                    properties.setdefault(prop, []).append(Call(call.file, call.index, (call.args[position],)))

        def enum_member(call: Call, enum: str):
            match = re.search(rf"\b{enum}\.(\w+)", call.args[0])
            return match.group(1) if match else None

        out.append(_finding("light.port", "光源 COM port", *_collect(properties.get("PortName", []), 0, r),
                            convert=lambda v: _com_port(v)))
        out.append(_finding("light.baud_rate", "光源 Baud rate", *_collect(properties.get("BaudRate", []), 0, r), convert=int))
        out.append(_finding("light.data_bits", "光源 Data bits", *_collect(properties.get("DataBits", []), 0, r), convert=int))
        for key, label, enum, mapping in (
            ("light.parity", "光源 Parity", "Parity", {"None": "none", "Odd": "odd", "Even": "even", "Mark": "mark", "Space": "space"}),
            ("light.stop_bits", "光源 Stop bits", "StopBits", {"One": "one", "OnePointFive": "one_point_five", "Two": "two"}),
        ):
            members: dict = {}
            unknown: list[SourceHit] = []
            for call in properties.get(enum if enum == "Parity" else "StopBits", []):
                member = enum_member(call, enum)
                if member in mapping:
                    members.setdefault(mapping[member], []).append(call.hit())
                else:
                    unknown.append(call.hit())
            out.append(_finding(key, label, members, unknown))

        writes = find_calls(self.files, ["Write", "WriteLine"], member=True)
        writes = [c for c in writes if _receiver(c) in names and c.args]
        uses_line = {c for c in writes if c.file.text[c.index : c.index + 9] == "WriteLine"}
        line_endings: dict = {}
        if writes and len(uses_line) == len(writes):
            newline = properties.get("NewLine", [])
            if newline:
                values, _unresolved = _collect(newline, 0, r)
                line_endings = {v: hits for v, hits in values.items() if isinstance(v, str)}
            else:
                line_endings = {"\n": [c.hit() for c in writes]}  # .NET SerialPort.NewLine default
        elif writes and not uses_line:
            line_endings = {"": [c.hit() for c in writes]}
        if line_endings:
            out.append(_finding("light.line_ending", "光源指令結尾", line_endings, [], convert=_line_ending,
                                show=lambda v: {"": "無", "\r": "CR", "\n": "LF", "\r\n": "CR+LF"}.get(v, repr(v))))
        out += self._light_commands(writes)
        return out

    def _light_commands(self, writes: list[Call]) -> list[ImportFinding | None]:
        on: dict[str, list[SourceHit]] = {}
        off: dict[str, list[SourceHit]] = {}
        unclear: list[SourceHit] = []
        writer_methods: set[str] = set()
        for call in writes:
            method = call.file.method_at(call.index)
            if method:
                writer_methods.add(method.name)
            values, _gaps = self.resolver.trace(call.args[0], call.file, call.index)
            for value in values:
                if not isinstance(value, (str, bytes)):
                    continue
                text = describe_bytes(value if isinstance(value, bytes) else value.encode("latin-1", errors="replace"))
                owner = self._literal_owner(value) or (method.name if method else "")
                kind = _command_kind(owner, text)
                hit = SourceHit(call.hit().file, call.hit().line, owner, call.hit().text)
                if kind == "on":
                    on.setdefault(text, []).append(hit)
                elif kind == "off":
                    off.setdefault(text, []).append(hit)
                else:
                    unclear.append(hit)
        out: list[ImportFinding | None] = []
        for key, label, found in (("light.on_commands", "光源開燈指令", on), ("light.off_commands", "光源關燈指令", off)):
            if found:
                hits = tuple(h for group in found.values() for h in group)
                commands = tuple(found)
                out.append(
                    ImportFinding(key, label, STATUS_PARTIAL, commands, "；".join(commands),
                                  "依所在方法名稱分類為開燈／關燈，請確認順序與內容。", hits)
                )
        if unclear:
            out.append(
                ImportFinding("info.light_commands", "其他光源指令", STATUS_INFO, None,
                              "；".join(sorted({h.method for h in unclear if h.method})) or "（無法分類）",
                              "這些固定指令無法判斷是開燈或關燈，請手動確認。", tuple(unclear))
            )
        out.append(self._brightness_template(writer_methods))
        return out

    def _literal_owner(self, value) -> str:
        """Method whose source contains this literal command (the caller of a Send(cmd) wrapper)."""
        if not isinstance(value, str) or not value:
            return ""
        for source in self.files:
            for match in re.finditer(r'@?"(?:[^"\\\n]|\\.|"")*"', source.text):
                if _literal(match.group(0)) == value:
                    method = source.method_at(match.start())
                    return method.name if method else ""
        return ""

    def _brightness_template(self, writer_methods: set[str]) -> ImportFinding | None:
        """Interpolated/format strings in methods that write to the port or call one that does."""
        if not writer_methods:
            return None
        callers = set(writer_methods)
        for source in self.files:
            for method in source.methods:
                body = source.text[method.start : method.body_end]
                if any(re.search(rf"\b{re.escape(name)}\s*\(", body) for name in writer_methods):
                    callers.add(method.name)
        templates: dict[str, list[SourceHit]] = {}
        notes: set[str] = set()
        for source in self.files:
            for match in re.finditer(r'\$@?"((?:[^"\\]|\\.|"")*)"|\b[Ss]tring\.Format\s*\(', source.text):
                method = source.method_at(match.start())
                if method is None or method.name not in callers:
                    continue
                if match.group(0).startswith("$"):
                    converted = _convert_interpolated(match.group(1))
                else:
                    parsed = split_arguments(source.text, match.end() - 1)
                    converted = _convert_format(parsed[0]) if parsed and parsed[0] else None
                if converted is None:
                    continue
                template, note = converted
                if "{value" not in template:
                    continue
                templates.setdefault(template, []).append(Call(source, match.start(), ()).hit())
                if note:
                    notes.add(note)
        if not templates:
            return None
        status = STATUS_PARTIAL if len(templates) == 1 else STATUS_CONFLICT
        value = next(iter(templates)) if len(templates) == 1 else None
        note = "依變數名稱判斷通道與亮度欄位，請用「送出」測試確認。" + " ".join(sorted(notes))
        hits = tuple(h for group in templates.values() for h in group)
        return ImportFinding("light.brightness_template", "光源亮度指令範本", status, value, "；".join(templates), note, hits)


def _receiver(call: Call) -> str:
    before = call.file.text[max(0, call.index - 80) : call.index].rstrip()
    before = before[:-1] if before.endswith(".") else before
    match = re.search(r"(\w+)\s*$", before)
    return match.group(1) if match else ""


def _com_port(value) -> str:
    text = str(value).strip().upper()
    if not re.fullmatch(r"COM\d+", text):
        raise ValueError(text)
    return text


def _line_ending(value) -> str:
    if value not in ("", "\r", "\n", "\r\n"):
        raise ValueError(value)
    return value


_OFF_WORDS = ("off", "close", "stop", "dispose", "shutdown", "exit", "disable", "關")
_ON_WORDS = ("on", "open", "start", "init", "enable", "connect", "load", "開")


def _command_kind(method: str, command: str) -> str:
    name = method.lower()
    if any(word in name for word in _OFF_WORDS):
        return "off"
    if any(word in name for word in _ON_WORDS):
        return "on"
    return ""


def _csharp_spec(spec: str) -> tuple[str, str]:
    """C# numeric format ("000", "D3", "X2") as a Python format spec, with a note when unsure."""
    spec = spec.strip()
    if not spec:
        return "", ""
    if re.fullmatch(r"0+", spec):
        return f"0{len(spec)}", ""
    match = re.fullmatch(r"([DdXx])(\d*)", spec)
    if match:
        kind, width = match.groups()
        suffix = "" if kind in "Dd" else kind
        return (f"0{width}{suffix}" if width else suffix or ""), ""
    return "", f"格式「{spec}」無法轉換，已省略。"


def _field_name(expression: str) -> str | None:
    lowered = expression.lower()
    if re.search(r"sum|check|crc|bcc|xor", lowered):
        return "xor" if "xor" in lowered else "checksum"
    if re.search(r"ch(an(nel)?)?\b|ch\d|channel|通道|\bch", lowered):
        return "channel"
    if re.search(r"val|bright|level|lum|intens|power|亮", lowered):
        return "value"
    return None


def _convert_interpolated(body: str) -> tuple[str, str] | None:
    out, notes = [], []
    position = 0
    for match in re.finditer(r"\{([^{}:]+)(?::([^{}]*))?\}", body):
        out.append(body[position : match.start()])
        name = _field_name(match.group(1))
        if name is None:
            return None
        spec, note = _csharp_spec(match.group(2) or "")
        notes.append(note)
        out.append("{" + name + (f":{spec}" if spec and name not in ("checksum", "xor") else "") + "}")
        position = match.end()
    out.append(body[position:])
    return "".join(out), " ".join(n for n in notes if n)


def _convert_format(args: list[str]) -> tuple[str, str] | None:
    format_text = _literal(args[0].strip())
    if not isinstance(format_text, str):
        return None
    fields = [_field_name(a) for a in args[1:]]
    out, notes = [], []
    position = 0
    for match in re.finditer(r"\{(\d+)(?::([^{}]*))?\}", format_text):
        out.append(escape_text(format_text[position : match.start()]))
        index = int(match.group(1))
        name = fields[index] if index < len(fields) else None
        if name is None:
            return None
        spec, note = _csharp_spec(match.group(2) or "")
        notes.append(note)
        out.append("{" + name + (f":{spec}" if spec and name not in ("checksum", "xor") else "") + "}")
        position = match.end()
    out.append(escape_text(format_text[position:]))
    return "".join(out), " ".join(n for n in notes if n)


def _device_name(value) -> str:
    text = str(value).strip()
    if isinstance(value, bool) or not text:
        raise ValueError(text)
    return text


def scan_legacy_program(selected: str | Path) -> LegacyImportReport:
    root, roots = project_roots(selected)
    code_paths, config_paths, resources = collect_files(roots)
    if not code_paths:
        raise LegacyImportError(f"「{root}」底下找不到 C# 原始碼（.cs）。只有 EXE 時，可先用 ILSpy 匯出原始碼再選。")
    files = _load(code_paths, root)
    resolver = Resolver(files, read_config_values(config_paths))
    findings = LegacyProgramAnalyzer(files, resolver, resources, root).run()
    order = {STATUS_READY: 0, STATUS_PARTIAL: 1, STATUS_CONFLICT: 2, STATUS_UNRESOLVED: 3, STATUS_WARNING: 4, STATUS_INFO: 5}
    findings.sort(key=lambda f: order.get(f.status, 9))
    return LegacyImportReport(str(root), tuple(roots), len(code_paths), tuple(findings))
