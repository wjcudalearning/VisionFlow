from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "packaging" / "specs"
BUILD_DIR = ROOT / "packaging" / "scripts"


class UtilityPackagingContractTests(unittest.TestCase):
    def test_every_standalone_tool_has_a_one_file_spec_and_build_script(self):
        contracts = {
            "NG Tile Area Tool.spec": (
                "tools/export_ng_tiles_by_area.py",
                "build_ng_tile_area_tool.ps1",
            ),
            "Pattern Grid Tile Exporter.spec": (
                "tools/export_pattern_grid_tiles.py",
                "build_pattern_grid_tile_exporter.ps1",
            ),
            "Matrix Summary Exporter.spec": (
                "tools/export_matrix_summary.py",
                "build_matrix_summary_exporter.ps1",
            ),
            "Scatter Plot Exporter.spec": (
                "tools/export_scatter_plots.py",
                "build_scatter_plot_exporter.ps1",
            ),
            "Tile Defect Distribution Exporter.spec": (
                "tools/export_tile_defect_distribution.py",
                "build_tile_defect_distribution_exporter.ps1",
            ),
        }
        for spec_name, (entry_point, build_name) in contracts.items():
            with self.subTest(spec=spec_name):
                spec = (SPEC_DIR / spec_name).read_text(encoding="utf-8")
                build = (BUILD_DIR / build_name).read_text(encoding="utf-8")
                self.assertIn(f"'{entry_point}'", spec)
                self.assertIn("str(ROOT / ENTRY_POINT)", spec)
                self.assertIn("exe = EXE(", spec)
                self.assertIn("console=False", spec)
                self.assertIn("Invoke-PyInstallerBuild @buildArguments", build)
                self.assertIn(spec_name, build)

    def test_specs_and_build_scripts_stay_inside_packaging(self):
        self.assertEqual([], sorted(path.name for path in ROOT.glob("*.ps1")))
        self.assertEqual([], sorted(path.name for path in ROOT.glob("*.spec")))
        self.assertTrue(SPEC_DIR.is_dir())
        self.assertTrue(BUILD_DIR.is_dir())

    def test_every_spec_resolves_the_repository_root_from_specpath(self):
        specs = sorted(SPEC_DIR.glob("*.spec"))
        self.assertTrue(specs)
        for spec in specs:
            with self.subTest(spec=spec.name):
                source = spec.read_text(encoding="utf-8")
                self.assertIn("SPEC_DIR = Path(SPECPATH).resolve()", source)
                self.assertIn("ROOT = SPEC_DIR.parent.parent", source)

    def test_every_spec_embeds_version_metadata_and_disables_upx(self):
        specs = sorted(SPEC_DIR.glob("*.spec"))
        for spec in specs:
            with self.subTest(spec=spec.name):
                source = spec.read_text(encoding="utf-8")
                self.assertIn("VERSION_INFO = ROOT / 'build' / 'version_info'", source)
                self.assertIn("version=str(VERSION_INFO)", source)
                self.assertIn("upx=False", source)
        helper = (BUILD_DIR / "pyinstaller_build.ps1").read_text(encoding="ascii")
        self.assertIn("write_version_info.py", helper)
        self.assertIn("Invoke-WithCleanBuildPath {", helper)
        self.assertIn("-LiteralPath", helper)

    def test_every_spec_is_referenced_by_a_build_script(self):
        build_sources = {
            path.name: path.read_text(encoding="utf-8")
            for path in BUILD_DIR.glob("*.ps1")
        }
        self.assertTrue(build_sources)
        for spec in sorted(SPEC_DIR.glob("*.spec")):
            with self.subTest(spec=spec.name):
                self.assertTrue(
                    any(spec.name in source for source in build_sources.values()),
                    f"no build script references {spec.name}",
                )

    def test_build_scripts_locate_the_repository_root_from_their_own_location(self):
        for path in sorted(BUILD_DIR.glob("build_*.ps1")):
            with self.subTest(script=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertIn(
                    '[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\\.."))',
                    source,
                )
                self.assertNotIn('Join-Path $PSScriptRoot "env', source)

    def test_all_tools_expose_noninteractive_smoke_mode(self):
        for entry_point in (
            "tools/export_ng_tiles_by_area.py",
            "tools/export_pattern_grid_tiles.py",
            "tools/export_matrix_summary.py",
            "tools/export_scatter_plots.py",
            "tools/export_tile_defect_distribution.py",
        ):
            with self.subTest(entry_point=entry_point):
                source = (ROOT / entry_point).read_text(encoding="utf-8")
                self.assertIn("--smoke-test", source)
                self.assertIn("TOOL_VERSION", source)

    def test_bundle_builder_refuses_overwrite_and_keeps_cpu_only_scope(self):
        build = (BUILD_DIR / "build_utility_tools.ps1").read_text(encoding="utf-8")
        readme = (ROOT / "docs" / "packaging" / "UTILITY_TOOLS_README.txt").read_text(
            encoding="utf-8"
        )

        self.assertIn("[ValidatePattern('^\\d+\\.\\d+\\.\\d+$')]", build)
        self.assertIn("Release ZIP already exists", build)
        self.assertIn("Compress-Archive -LiteralPath $bundleRoot", build)
        self.assertIn("release_artifacts", build)
        self.assertIn(r"docs\packaging\UTILITY_TOOLS_README.txt", build)
        for name in (
            "NG-Tile-Area-Tool.exe",
            "Pattern-Grid-Tile-Exporter.exe",
            "Matrix-Summary-Exporter.exe",
            "Scatter-Plot-Exporter.exe",
            "Tile-Defect-Distribution-Exporter.exe",
        ):
            self.assertIn(name, build)
        self.assertIn("CPU-only", readme)
        self.assertIn("未進行程式碼簽章", readme)
        self.assertTrue((ROOT / "release_artifacts" / "README.md").is_file())

    def test_ng_tile_builder_uses_the_packaging_document_source(self):
        build = (BUILD_DIR / "build_ng_tile_area_tool.ps1").read_text(encoding="utf-8")

        self.assertIn(r"docs\packaging\NG_TILE_AREA_TOOL_README.txt", build)
        self.assertTrue(
            (ROOT / "docs" / "packaging" / "NG_TILE_AREA_TOOL_README.txt").is_file()
        )


if __name__ == "__main__":
    unittest.main()
