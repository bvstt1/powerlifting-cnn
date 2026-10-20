"""Lanza prototipo, test, validación o verificación BP front V3.

Reenvía argumentos a una etapa concreta sin ejecutarlas todas ni alterar
por sí mismo el algoritmo o los datos existentes.
"""
import argparse
import subprocess
import sys
from pathlib import Path


BP_DIR = Path(__file__).resolve().parents[1]
ROOT = BP_DIR.parents[1]
SCRIPTS = {
    "prototype": Path(__file__).resolve().parent / "rebuild_bar_tracking_v3_from_raw.py",
    "test": Path(__file__).resolve().parent / "test_bar_tracking_v3_prototype.py",
    "validate": Path(__file__).resolve().parent / "validate_bar_tracking_v3_76.py",
    "verify": Path(__file__).resolve().parent / "verify_bar_tracking_v3_validation76.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("stage", choices=SCRIPTS)
    args, restantes = parser.parse_known_args()
    return subprocess.call([sys.executable, str(SCRIPTS[args.stage]), *restantes], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
