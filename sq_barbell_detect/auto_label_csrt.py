
"""Auto-etiquetado YOLO de barra SQ lateral usando CSRT.
Un click por video (frame 0) -> tracking automático -> labels YOLO.
Guarda en sq_barbell_detect/labels/ listo para split y entrenar.
"""
import sys
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from display.sq.bar_tracker import BarTracker, create_csrt


VIDEOS_DIR = Path(r"C:\Users\basti\MediapipePythonProjects\dataset\sq\left")
IMAGES_DIR = Path(r"C:\Users\basti\MediapipePythonProjects\sq_barbell_detect\images")
LABELS_DIR = Path(r"C:\Users\basti\MediapipePythonProjects\sq_barbell_detect\labels")
LABELS_DIR.mkdir(exist_ok=True)

STRIDE = 30  # debe coincidir con extract_frames_for_annotation.py
MAX_FRAMES_PER_VIDEO = 5


def select_roi_first_frame(video_path):
    """Abre video, muestra frame 0, devuelve ROI (x,y,w,h) en pixeles originales."""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    roi = cv2.selectROI(f"Inicializa BARRA - {video_path.name} (ENTER/SPACE confirm, C cancel)", frame, False)
    cv2.destroyAllWindows()
    if roi == (0, 0, 0, 0):
        return None
    return roi  # (x, y, w, h)


def frames_to_label(video_path):
    """Genera indices de frames que tienen imagen extraida (coincide con extract_frames_for_annotation)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return list(range(0, total, STRIDE))[:MAX_FRAMES_PER_VIDEO]


def main():
    video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
    print(f"Videos a procesar: {len(video_files)}")

    for video_path in video_files:
        print(f"\n--- {video_path.name} ---")

        target_frames = frames_to_label(video_path)
        if not target_frames:
            print("  Sin frames objetivo")
            continue

        roi = select_roi_first_frame(video_path)
        if roi is None:
            print("  Cancelado por usuario")
            continue

        x, y, w, h = roi
        print(f"  ROI inicial: x={x} y={y} w={w} h={h}")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        tracker = BarTracker(fps, tracker_factory=create_csrt)
        tracker.set_enabled(True)

        frame_idx = 0
        target_set = set(target_frames)
        labeled = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx == 0:
                ok = tracker.initialize(frame, 0, (float(x), float(y), float(w), float(h)))
                if not ok:
                    print("  Fallo al inicializar CSRT")
                    break
            else:
                tracker.update(frame, frame_idx)

            if frame_idx in target_set:
                center = tracker.current_center(frame_idx)
                if center:
                    cx, cy = center
                    h_img, w_img = frame.shape[:2]
                    label_line = f"0 {cx/w_img:.6f} {cy/h_img:.6f} {w/w_img:.6f} {h/h_img:.6f}\n"
                    img_name = f"{video_path.stem}_f{frame_idx:05d}.jpg"
                    lbl_path = LABELS_DIR / f"{video_path.stem}_f{frame_idx:05d}.txt"
                    lbl_path.write_text(label_line)
                    labeled += 1
                else:
                    print(f"  Frame {frame_idx}: tracking perdido")

            frame_idx += 1

        cap.release()
        print(f"  Etiquetados: {labeled}/{len(target_frames)} frames")

    print(f"\nListo. Labels en: {LABELS_DIR}")
    print(f"Total archivos .txt: {len(list(LABELS_DIR.glob('*.txt')))}")


if __name__ == "__main__":
    main()