"""Lanza por separado scout, reconstrucción, cierre o auditoría BP front V2.2.

Reenvía los argumentos a cada script sin encadenar fases ni modificar datos.
"""
import argparse
import subprocess
import sys
from pathlib import Path


BP_DIR = Path(__file__).resolve().parents[1]
ROOT = BP_DIR.parents[1]
SCRIPTS = {
    "scout": Path(__file__).resolve().parent / "scout_tracking_cases_front_bp.py",
    "rebuild": Path(__file__).resolve().parent / "rebuild_tracking_front_bp_from_raw.py",
    "audit": Path(__file__).resolve().parent / "audit_bar_tracking_front_bp.py",
    "finalize": BP_DIR / "finalize_segmentation_front_bp.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("stage", choices=SCRIPTS)
    args, restantes = parser.parse_known_args()
    return subprocess.call([sys.executable, str(SCRIPTS[args.stage]), *restantes], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
