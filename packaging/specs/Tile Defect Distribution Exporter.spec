# -*- mode: python ; coding: utf-8 -*-
#
# This spec lives in packaging\specs. PyInstaller resolves relative paths against
# the spec directory, so every source path is derived from the repository root.
from pathlib import Path

SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent.parent
ENTRY_POINT = 'tools/export_tile_defect_distribution.py'
VERSION_INFO = ROOT / 'build' / 'version_info' / 'Tile Defect Distribution Exporter.txt'

a = Analysis(
    [str(ROOT / ENTRY_POINT)],
    pathex=[],
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
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='export_tile_defect_distribution',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=str(VERSION_INFO),
)
