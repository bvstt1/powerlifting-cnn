import numpy as np
from pathlib import Path
from extract_body_keypoints import create_landmarker, extract_keypoints_from_video
from extract_bar_features import extract_bar_features


DATASET_ROOT = Path("dataset")
OUTPUT_ROOT = Path("processed")

CAMERAS = {
    "front": "cam_front.mp4",
    "left": "cam_left.mp4",
    "right": "cam_right.mp4"
}


def process_dataset():

    for exercise_dir in sorted(DATASET_ROOT.iterdir()):
        if not exercise_dir.is_dir():
            continue

        exercise_name = exercise_dir.name
        print(f"\n=== PROCESANDO {exercise_name.upper()} ===")

        for attempt_dir in sorted(exercise_dir.iterdir()):
            if not attempt_dir.is_dir():
                continue

            attempt_name = attempt_dir.name
            print(f"\n→ Intento: {attempt_name}")

            out_dir = OUTPUT_ROOT / exercise_name / attempt_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for cam, filename in CAMERAS.items():
                video_path = attempt_dir / filename

                if not video_path.exists():
                    print(f"  ⚠ {filename} no existe, se omite")
                    continue

                print(f"  Procesando cámara: {cam}")

                # 🔥 CREAR LANDMARKER NUEVO POR VIDEO
                landmarker = create_landmarker()

                keypoints = extract_keypoints_from_video(video_path, landmarker)

                landmarker.close()

                np.save(out_dir / f"{cam}_body.npy", keypoints)
                print(f"    ✔ {cam}_body.npy {keypoints.shape}")

                # Barra solo para cámara frontal
                if cam == "front":
                    bar_features = extract_bar_features(video_path)
                    np.save(out_dir / "front_bar.npy", bar_features)
                    print(f"    ✔ front_bar.npy {bar_features.shape}")

    print("\nProcesamiento completo.")


if __name__ == "__main__":
    process_dataset()