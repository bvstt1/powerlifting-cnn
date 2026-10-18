"""Inspección reproducible sin GUI del mismo tracker integrado en el visor.

Ejemplos (desde raíz):
 python display/sq/bar_tracking_demo.py --video sq_007 --frame 8 --inspect-only
 python display/sq/bar_tracking_demo.py --video sq_007 --frame 8 --roi 500 500 40 45
Las ROI se eligen visualmente, nunca se infieren de una etiqueta de validez.
"""
import argparse
from pathlib import Path

import cv2

from bar_tracker import BarTracker, VideoViewport


ROOT = Path(__file__).resolve().parents[2]


def run(video, start, roi, output, max_frames, inspect_only=False):
    path = ROOT / "dataset" / "sq" / "left" / f"{video}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir {path}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Frame {start} no disponible: {path}")
        if inspect_only:
            target = output / f"{video}_frame_{start:04d}_select.jpg"
            cv2.imwrite(str(target), frame)
            print(f"Inspección: {target}, dimensiones={frame.shape[1]}x{frame.shape[0]}")
            return
        if roi is None:
            raise ValueError("La prueba requiere ROI manual (--roi x y ancho alto)")
        tracker = BarTracker(fps)
        tracker.set_enabled(True)
        tracker.initialize(frame, start, roi)
        print(f"{video}: CSRT inicializado en frame {start}, ROI={roi} (píxeles originales)")
        samples = []

        def snapshot(image, number):
            viewport = VideoViewport(image.shape[1], image.shape[0], 0, 0,
                                     image.shape[1], image.shape[0])
            canvas = image.copy()
            tracker.draw(canvas, viewport, number)
            cv2.putText(canvas, f"{video} t={number} {tracker.state}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (210, 60, 255), 2)
            scale = min(360/image.shape[1], 500/image.shape[0])
            samples.append(cv2.resize(canvas, (int(image.shape[1]*scale), int(image.shape[0]*scale))))

        snapshot(frame, start)
        for offset in range(1, max_frames):
            ok, frame = capture.read()
            if not ok:
                break
            idx = start + offset
            ok = tracker.update(frame, idx)
            if offset % max(1, max_frames//8) == 0 or not ok:
                snapshot(frame, idx)
            if not ok:
                print(f"{video}: tracking perdido en frame {idx}")
                break
        csv_path = tracker.csv_export(output / f"{video}_bar_tracking.csv")
        if samples:
            import numpy as np
            # Todos los thumbnails comparten las mismas dimensiones.
            contact = np.concatenate(samples, axis=1)
            cv2.imwrite(str(output / f"{video}_bar_preview.jpg"), contact)
        observed = [r for r in tracker.records if r["status"] == "siguiendo"]
        if observed:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot([r["time_s"] for r in observed],
                    [r["center_y"] for r in observed], color="magenta", linewidth=1.5,
                    label="y original (sin suavizar)")
            ax.invert_yaxis()  # y de imagen crece hacia abajo
            ax.set(xlabel="Tiempo (s)", ylabel="y (px; hacia abajo +)",
                   title=f"{video} - barra: diagnóstico, no dato de entrenamiento")
            ax.grid(alpha=.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / f"{video}_bar_y_raw.png", dpi=140)
            plt.close(fig)
            bottom = max(observed, key=lambda row: row["center_y"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, bottom["frame"])
            ok, bottom_frame = capture.read()
            if ok:
                x, y = int(bottom["box_x"]), int(bottom["box_y"])
                w, h = int(bottom["box_w"]), int(bottom["box_h"])
                cv2.rectangle(bottom_frame, (x,y), (x+w,y+h), (210,60,255), 3)
                cv2.circle(bottom_frame,
                           (int(bottom["center_x"]), int(bottom["center_y"])),
                           7, (210,60,255), -1)
                cv2.imwrite(str(output / f"{video}_bar_bottom_frame_{bottom['frame']:04d}.jpg"),
                            bottom_frame)
        print(f"{video}: {len(observed)} frames observados, "
              f"{len(tracker.records)-len(observed)} fallos; CSV={csv_path}; "
              f"preview={output / (video + '_bar_preview.jpg')}")
    finally:
        capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="ID sq_NNN de desarrollo")
    parser.add_argument("--frame", type=int, default=0, help="Frame original de selección")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X","Y","W","H"))
    parser.add_argument("--max-frames", type=int, default=140)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "display" / "sq" / "bar_diagnostics")
    args = parser.parse_args()
    run(args.video, args.frame, tuple(args.roi) if args.roi else None,
        args.output, args.max_frames, args.inspect_only)
