"""Vista previa manual de las 3 cámaras con la sincronía medida.

    python -m judge.capture.preview              # abre las cámaras del config.yaml
    python -m judge.capture.preview --probe      # lista los índices de cámara disponibles

ESC cierra la ventana. No procesa ni guarda nada.
"""

import argparse

import cv2
import numpy as np

from judge.capture.camera import probe_cameras
from judge.capture.multi_camera import MultiCameraCapture
from judge.config import VIEWS, load_config

TILE_HEIGHT = 360


def tile(image, label):
    height = TILE_HEIGHT
    width = max(1, round(image.shape[1] * height / image.shape[0]))
    out = cv2.resize(image, (width, height))
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="ruta de config.yaml (por defecto judge/config.yaml)")
    parser.add_argument("--probe", action="store_true", help="listar índices de cámara disponibles y salir")
    args = parser.parse_args()
    config = load_config(args.config, check_files=False)

    if args.probe:
        found = probe_cameras(config.capture)
        for camera in found:
            print(f"Cámara {camera['index']}: {camera['width']}x{camera['height']} @ {camera['fps']:.0f} FPS")
        print(f"{len(found)} cámaras disponibles; configuradas: {config.capture.cameras}")
        return

    with MultiCameraCapture(config.capture) as capture:
        print("ESC para salir")
        while True:
            frame_set = capture.read(timeout=1.0)
            if frame_set is None:
                print("Sin conjunto sincronizado en 1 s")
                continue
            mosaic = np.hstack([tile(frame_set.frames[v].image, v) for v in VIEWS])
            cv2.putText(mosaic, f"skew {frame_set.skew_ms:.1f} ms  descartados {capture.dropped_frames}",
                        (10, TILE_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Captura 3 vistas", mosaic)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
