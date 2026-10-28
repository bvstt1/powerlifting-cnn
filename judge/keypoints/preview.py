"""Visor manual: extrae y dibuja los keypoints de una vista sobre un video o una cámara.

    python -m judge.keypoints.preview --exercise squat --view left --video ruta/video.mp4
    python -m judge.keypoints.preview --exercise bench --view front --camera 0

`q` o ESC cierran la ventana. Con CPU cada frame tarda varios segundos (imgsz=960): el
video avanza frame a frame, sin reproducirse a tiempo real.
"""

import argparse
import time

import cv2

from judge.config import VIEWS, load_config
from judge.keypoints import KeypointStage


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exercise", required=True, choices=("squat", "bench", "deadlift"))
    parser.add_argument("--view", required=True, choices=VIEWS)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="ruta de un video")
    source.add_argument("--camera", type=int, help="índice de cámara")
    parser.add_argument("--labels", action="store_true", help="mostrar nombre y confianza de cada punto")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()

    config = load_config(exercise=args.exercise, check_files=False)
    capture = cv2.VideoCapture(args.video if args.video else args.camera)
    if not capture.isOpened():
        raise SystemExit("No se pudo abrir la fuente de video")

    with KeypointStage(config) as stage:
        skeleton = config.view(args.view).skeleton
        print(f"{args.exercise}/{args.view}: {skeleton.name} ({skeleton.backend}, {skeleton.num_points} puntos)")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            results = stage.process({args.view: frame}, {args.view: time.perf_counter()})
            elapsed = time.perf_counter() - started
            annotated = stage.annotate({args.view: frame}, results, args.min_confidence, args.labels)[args.view]
            visible = int(results[args.view].visible(args.min_confidence).sum())
            cv2.putText(annotated, f"{visible}/{skeleton.num_points} puntos  {elapsed:.2f} s",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Keypoints", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
