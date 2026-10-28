"""Visor manual de la segmentación del press de banca sobre un video o una cámara.

    python -m judge.segmentation.preview --view left --video ruta/video.mp4
    python -m judge.segmentation.preview --view front --camera 0 --skeleton

Con --skeleton se dibuja también el esqueleto de la misma vista (es lo que verá el sistema).
`q` o ESC cierran la ventana. En CPU cada frame tarda varios segundos.
"""

import argparse
import time

import cv2

from judge.config import VIEWS, load_config
from judge.keypoints import KeypointStage
from judge.segmentation import SegmentationStage


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", required=True, choices=VIEWS)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="ruta de un video")
    source.add_argument("--camera", type=int, help="índice de cámara")
    parser.add_argument("--skeleton", action="store_true", help="dibujar también el esqueleto")
    args = parser.parse_args()

    config = load_config(exercise="bench", check_files=False)
    capture = cv2.VideoCapture(args.video if args.video else args.camera)
    if not capture.isOpened():
        raise SystemExit("No se pudo abrir la fuente de video")

    keypoints = KeypointStage(config) if args.skeleton else None
    with SegmentationStage(config) as segmentation:
        model = config.view(args.view).segmentation
        print(f"bench/{args.view}: {model.name} ({model.model.name})")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            image = {args.view: frame}
            if keypoints:
                image = keypoints.annotate(image, keypoints.process(image, {args.view: started}))
            results = segmentation.process({args.view: frame})
            annotated = segmentation.annotate(image, results)[args.view]
            counts = {name: len(results[args.view].by_class(name)) for name in ("bar", "bench", "person")}
            cv2.putText(annotated, f"{counts}  {time.perf_counter() - started:.2f} s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow("Segmentacion", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    if keypoints:
        keypoints.close()
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
