import cv2

cap = cv2.VideoCapture("Inputs/videos/2024 Powerlifting America Classic Open Nationals - Day 1, Session A.mp4")

orig_fps = cap.get(cv2.CAP_PROP_FPS)
target_fps = 25
step = round(orig_fps / target_fps)

frames = []
i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if i % step == 0:
        frames.append(frame)

    i += 1

cap.release()

print(f"Frames originales aprox: {i}")
print(f"Frames finales: {len(frames)}")
print(f"FPS original: {orig_fps}")
print(f"FPS efectivo aprox: {orig_fps / step}")
