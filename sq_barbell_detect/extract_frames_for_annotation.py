"""Extrae frames de videos SQ left para anotar barras con LabelImg/CVAT."""
import cv2
from pathlib import Path

DATASET_VIDEOS = Path(r"C:\Users\basti\MediapipePythonProjects\dataset\sq\left")
OUTPUT_DIR = Path(r"C:\Users\basti\MediapipePythonProjects\sq_barbell_detect\images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRIDE = 30  # 1 frame cada 30 (~1 por segundo a 30fps)
MAX_FRAMES_PER_VIDEO = 5

for video_path in sorted(DATASET_VIDEOS.glob("*.mp4")):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{video_path.name}: {total} frames")

    saved = 0
    for frame_idx in range(0, total, STRIDE):
        if saved >= MAX_FRAMES_PER_VIDEO:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        out_name = f"{video_path.stem}_f{frame_idx:05d}.jpg"
        cv2.imwrite(str(OUTPUT_DIR / out_name), frame)
        saved += 1
    cap.release()
    print(f"  -> {saved} frames guardados en {OUTPUT_DIR}")

print(f"\nTotal frames: {len(list(OUTPUT_DIR.glob('*.jpg')))}")
print("Siguiente paso: abre LabelImg, carga carpeta 'images', anota clase 'barbell'")
print("Guarda YOLO format -> genera labels/ con .txt por imagen")