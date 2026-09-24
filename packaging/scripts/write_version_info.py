"""Write a PyInstaller PE version resource for a packaged executable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
VERSION_PATTERN = re.compile(r"^\d+(?:\.\d+){0,3}$")


def _version_parts(value: str) -> tuple[int, int, int, int]:
    if not VERSION_PATTERN.fullmatch(value):
        raise ValueError(f"Version must contain one to four numeric parts: {value}")
    parts = [int(part) for part in value.split(".")]
    if any(part > 65535 for part in parts):
        raise ValueError("Version parts must be between 0 and 65535")
    return tuple((parts + [0, 0, 0, 0])[:4])


def _commit_id() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--product-name", required=True)
    parser.add_argument("--executable-name", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    file_version = _version_parts(args.version)
    commit = _commit_id()
    product_version = f"{args.version}+{commit}" if commit != "unknown" else args.version
    description = args.product_name
    string_values = {
        "CompanyName": "VisionFlow",
        "FileDescription": description,
        "FileVersion": args.version,
        "InternalName": args.product_name,
        "LegalCopyright": "VisionFlow contributors",
        "OriginalFilename": args.executable_name,
        "ProductName": args.product_name,
        "ProductVersion": product_version,
    }
    strings = ",\n         ".join(
        f"StringStruct({_quote(key)}, {_quote(value)})"
        for key, value in string_values.items()
    )
    file_version_literal = ", ".join(str(part) for part in file_version)
    source = f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({file_version_literal}),
    prodvers=({file_version_literal}),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        u'040904B0',
        [{strings}])
    ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
