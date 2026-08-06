from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.csv_summary import CsvSummaryExporter
from core.logging_system import configure_logging, get_logger
from core.pipeline import AOIPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AOI CV inspection pipeline.")
    parser.add_argument("--gui", action="store_true", help="Start the PySide6 GUI.")
    parser.add_argument("--image", help="Path to input image.")
    parser.add_argument("--recipe", help="Path to YAML recipe.")
    parser.add_argument("--output", default="outputs", help="Output directory.")
    parser.add_argument("--debug", action="store_true", help="Save extra debug artifacts when available.")
    parser.add_argument("--log-level", default=None, help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    parser.add_argument("--log-dir", default=None, help="Directory for rotating AOI log files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_dir = Path(args.log_dir) if args.log_dir else Path(args.output) / "logs"
    configure_logging(log_dir=log_dir, level=args.log_level)
    logger = get_logger("main")
    logger.info("AOI application started")
    if args.gui:
        from gui.main_window import run_app

        logger.info("Launching GUI")
        return run_app()

    if not args.image or not args.recipe:
        logger.error("CLI run missing required --image or --recipe argument")
        raise SystemExit("--image and --recipe are required unless --gui is used.")

    logger.info("CLI inspection requested: image=%s recipe=%s output=%s", args.image, args.recipe, args.output)
    pipeline = AOIPipeline(
        recipe_path=Path(args.recipe),
        output_dir=Path(args.output),
        debug=args.debug,
    )
    result = pipeline.run(Path(args.image))
    CsvSummaryExporter.finalize_result(Path(args.output), result)

    summary = {
        "image_name": result["image_name"],
        "recipe_name": result["recipe_name"],
        "final_result": result["final_result"],
        "ng_count": result["summary"]["ng_count"],
        "defect_count": result["summary"]["defect_count"],
        "duration_sec": result.get("duration_sec", 0),
        "outputs": result["outputs"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result["final_result"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
