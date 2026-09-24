# -*- mode: python ; coding: utf-8 -*-
#
# This spec lives in packaging\specs. PyInstaller resolves relative paths against
# the spec directory, so every source path is derived from the repository root.
from pathlib import Path

SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent.parent
VERSION_INFO = ROOT / 'build' / 'version_info' / 'VisionFlow AOI.txt'

cuda_dll = ROOT / 'gpu' / 'visionflow_cuda.dll'
cuda_binaries = [(str(cuda_dll), 'gpu')] if cuda_dll.exists() else []
# The large-template Pattern Match path uses the FFT kernels inside visionflow_cuda.dll, so no
# external FFT runtime is packaged. A cufft64_*.dll left in gpu/ from an earlier build is ignored.

a = Analysis(
    [str(ROOT / 'gui_launcher.py')],
    pathex=[],
    binaries=cuda_binaries,
    datas=[
        (str(ROOT / 'recipes'), 'recipes'),
        (str(ROOT / 'models' / 'yolox'), 'models/yolox'),
        (str(ROOT / 'build_provenance.json'), '.'),
    ],
    # pythonnet loads the camera machine's own Sapera LT SapClassBasic.dll at runtime on the .NET
    # Framework runtime. The managed hooks collect clr.pyd/Python.Runtime.dll; the vendor DLL and
    # the LSI-8181 driver stay on the machine and must never be bundled.
    # `devices.ccd_settings_import` has no caller yet (its GUI action is a pending Todo item), so
    # PyInstaller would drop it; bundling it keeps the packaged app matching the documented CCD
    # capability set, so wiring that action later cannot fail on the offline camera machine.
    # Detectors exported by the traditional-CV tuning tool import contour_preprocess_tool.engine at
    # runtime; keep the engine and exporter bundled so a newly registered detector and the packaged
    # tuned-detector smoke both work.
    hiddenimports=[
        'pythonnet',
        'clr_loader',
        'devices.ccd_settings_import',
        'contour_preprocess_tool.engine',
        'contour_preprocess_tool.detector_export',
    ],
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
    [],
    exclude_binaries=True,
    name='VisionFlow AOI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=str(VERSION_INFO),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='VisionFlow AOI',
)
