from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class UtilityPackagingContractTests(unittest.TestCase):
    def test_every_standalone_tool_has_a_one_file_spec_and_build_script(self):
        contracts = {
            "NG Tile Area Tool.spec": (
                "export_ng_tiles_by_area.py",
                "build_ng_tile_area_tool.ps1",
            ),
            "Pattern Grid Tile Exporter.spec": (
                "export_pattern_grid_tiles.py",
                "build_pattern_grid_tile_exporter.ps1",
            ),
            "Matrix Summary Exporter.spec": (
                "export_matrix_summary.py",
                "build_matrix_summary_exporter.ps1",
            ),
            "Scatter Plot Exporter.spec": (
                "export_scatter_plots.py",
                "build_scatter_plot_exporter.ps1",
            ),
        }
        for spec_name, (entry_point, build_name) in contracts.items():
            with self.subTest(spec=spec_name):
                spec = (ROOT / spec_name).read_text(encoding="utf-8")
                build = (ROOT / build_name).read_text(encoding="utf-8")
                self.assertIn(f"['{entry_point}']", spec)
                self.assertIn("exe = EXE(", spec)
                self.assertIn("console=False", spec)
                self.assertIn("-m PyInstaller", build)
                self.assertIn(spec_name, build)

    def test_all_tools_expose_noninteractive_smoke_mode(self):
        for entry_point in (
            "export_ng_tiles_by_area.py",
            "export_pattern_grid_tiles.py",
            "export_matrix_summary.py",
            "export_scatter_plots.py",
        ):
            with self.subTest(entry_point=entry_point):
                source = (ROOT / entry_point).read_text(encoding="utf-8")
                self.assertIn("--smoke-test", source)
                self.assertIn("TOOL_VERSION", source)

    def test_bundle_builder_refuses_overwrite_and_keeps_cpu_only_scope(self):
        build = (ROOT / "build_utility_tools.ps1").read_text(encoding="utf-8")
        readme = (ROOT / "UTILITY_TOOLS_README.txt").read_text(encoding="utf-8")

        self.assertIn("[ValidatePattern('^\\d+\\.\\d+\\.\\d+$')]", build)
        self.assertIn("Release ZIP already exists", build)
        self.assertIn("Compress-Archive -LiteralPath $bundleRoot", build)
        for name in (
            "NG-Tile-Area-Tool.exe",
            "Pattern-Grid-Tile-Exporter.exe",
            "Matrix-Summary-Exporter.exe",
            "Scatter-Plot-Exporter.exe",
        ):
            self.assertIn(name, build)
        self.assertIn("CPU-only", readme)
        self.assertIn("未進行程式碼簽章", readme)


if __name__ == "__main__":
    unittest.main()
