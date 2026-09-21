# -*- mode: python ; coding: utf-8 -*-
#
# This spec lives in packaging\specs. PyInstaller resolves relative paths against
# the spec directory, so every source path is derived from the repository root.
from pathlib import Path

SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent.parent
ENTRY_POINT = 'contour_preprocess_tool/launcher.py'
VERSION_INFO = 'contour_preprocess_tool/version_info.txt'

a = Analysis(
    [str(ROOT / ENTRY_POINT)],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
# API-set forwarders and UCRT are operating-system components. The build host
# can also expose unrelated Poppler ICU/OpenSSL DLLs through PATH; collecting
# those beside QtCore can make Qt load an incompatible dependency first.
_excluded_runtime_names = {
    'icudt78.dll',
    'icuuc.dll',
    'libcrypto-3-x64.dll',
    'libssl-3-x64.dll',
    'ucrtbase.dll',
}
_system_runtime_prefixes = ('api-ms-win-', 'ext-ms-win-')
a.binaries = [
    entry
    for entry in a.binaries
    if Path(entry[0]).name.lower() not in _excluded_runtime_names
    and not Path(entry[0]).name.lower().startswith(_system_runtime_prefixes)
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Traditional CV Tuning Tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=str(ROOT / VERSION_INFO),
)
