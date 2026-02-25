import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# -----------------------------
# Configuración
# -----------------------------
MODEL_PATH = "../../models/pose_landmarker_heavy.task"
VIDEO_PATH = r"C:\Users\basti\Desktop\AQP17xjrbDFEnDWJVVDm00TVKDOx6MWKXSae5G9KnJv2eGq0xT9v5d-Clajmj-Ftrh7YAfXGDntQ5pNfmxa9r0TTRm6MZiW-yoNZN18.mp4"

CONF_THRESHOLD = 0.5
DESCENT_THRESHOLD = 0.003
ASCENT_THRESHOLD = -0.003
STABLE_FRAMES = 5


# -----------------------------
# Landmarker
# -----------------------------
def create_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )
    return vision.PoseLandmarker.create_from_options(options)


# -----------------------------
# Selección landmarks
# -----------------------------
def select_hip_knee(landmarks):
    left = [23, 25]
    right = [24, 26]

    left_vis = np.mean([landmarks[i].visibility for i in left])
    right_vis = np.mean([landmarks[i].visibility for i in right])

    idx = left if left_vis > right_vis else right
    return landmarks[idx[0]], landmarks[idx[1]]


# -----------------------------
# Máquina de estados
# -----------------------------
class SquatStateMachine:
    def __init__(self):
        self.state = "STANDING"
        self.prev_hip_y = None
        self.bottom_hip_y = None
        self.valid_rep = False
        self.stable_counter = 0

    def update(self, hip_y, knee_y):

        if self.prev_hip_y is None:
            self.prev_hip_y = hip_y
            return self.state, None

        velocity = hip_y - self.prev_hip_y

        # Transiciones
        if self.state == "STANDING":
            if velocity > DESCENT_THRESHOLD:
                self.state = "DESCENDING"
                self.valid_rep = False

        elif self.state == "DESCENDING":
            if velocity < ASCENT_THRESHOLD:
                self.state = "BOTTOM"
                self.bottom_hip_y = hip_y

        elif self.state == "BOTTOM":
            # Validación profundidad
            if hip_y > knee_y:
                self.valid_rep = True

            if velocity < ASCENT_THRESHOLD:
                self.state = "ASCENDING"

        elif self.state == "ASCENDING":
            if abs(velocity) < 0.001:
                self.stable_counter += 1
            else:
                self.stable_counter = 0

            if self.stable_counter > STABLE_FRAMES:
                result = self.valid_rep
                self.state = "STANDING"
                self.stable_counter = 0
                self.prev_hip_y = hip_y
                return self.state, result

        self.prev_hip_y = hip_y
        return self.state, None


# -----------------------------
# Ejecutar
# -----------------------------
def run(video_path):

    landmarker = create_landmarker()
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    squat_machine = SquatStateMachine()

    last_result = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx * 1000 / max(cap.get(cv2.CAP_PROP_FPS),1))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:

            hip, knee = select_hip_knee(result.pose_landmarks[0])

            if hip.visibility > CONF_THRESHOLD and knee.visibility > CONF_THRESHOLD:

                hip_y = hip.y
                knee_y = knee.y

                state, rep_result = squat_machine.update(hip_y, knee_y)

                if rep_result is not None:
                    last_result = rep_result

                h, w, _ = frame.shape

                # Dibujar puntos
                hip_px = int(hip.x * w)
                hip_py = int(hip_y * h)

                knee_px = int(knee.x * w)
                knee_py = int(knee_y * h)

                cv2.circle(frame, (hip_px, hip_py), 8, (0,255,0), -1)
                cv2.circle(frame, (knee_px, knee_py), 8, (255,0,0), -1)

                # Línea horizontal en rodilla
                cv2.line(frame, (0, knee_py), (w, knee_py), (255,0,0), 2)

                # Mostrar estado
                cv2.putText(frame, f"State: {state}",
                            (30,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255,255,255),
                            2)

                if last_result is not None:
                    color = (0,255,0) if last_result else (0,0,255)
                    text = "VALID" if last_result else "NO REP"

                    cv2.putText(frame, text,
                                (30,90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                color,
                                3)

        cv2.imshow("Squat Depth Validation", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_idx += 1

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(VIDEO_PATH)