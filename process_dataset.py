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
    landmarker = create_landmarker()

    for exercise_dir in DATASET_ROOT.iterdir():
        if not exercise_dir.is_dir():
            continue

        print(f"\n=== {exercise_dir.name.upper()} ===")

        for attempt_dir in exercise_dir.iterdir():
            if not attempt_dir.is_dir():
                continue

            print(f"→ {attempt_dir.name}")
            out_dir = OUTPUT_ROOT / exercise_dir.name / attempt_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for cam, filename in CAMERAS.items():
                video_path = attempt_dir / filename

                keypoints = extract_keypoints_from_video(video_path, landmarker)
                np.save(out_dir / f"{cam}_body.npy", keypoints)

                print(f"  ✔ {cam}_body {keypoints.shape}")

                if cam == "front":
                    bar_features = extract_bar_features(video_path)
                    np.save(out_dir / "front_bar.npy", bar_features)
                    print(f"  ✔ front_bar {bar_features.shape}")

    landmarker.close()


if __name__ == "__main__":
    process_dataset()
