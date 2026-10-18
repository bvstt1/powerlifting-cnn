"""Entrada monovista al evaluador congelado; rechaza repetir inferencia guardada."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from training.bp_lateral_final_test_v1.run import main

if __name__ == "__main__":
    if "--view" in sys.argv:
        raise SystemExit("Esta entrada corresponde exclusivamente a LEFT")
    sys.argv.extend(["--view","left"])
    main()
