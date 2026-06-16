from pathlib import Path
import shutil

# Carpeta raíz donde están los dl_150, dl_151, ..., dl_259
ROOT = Path(r"D:")

# Carpetas destino
front_dir = ROOT / "front"
left_dir = ROOT / "left"
right_dir = ROOT / "right"

front_dir.mkdir(exist_ok=True)
left_dir.mkdir(exist_ok=True)
right_dir.mkdir(exist_ok=True)

for i in range(150, 260):
    folder = ROOT / f"dl_{i}"

    if not folder.exists():
        print(f"No existe: {folder}")
        continue

    for file in folder.iterdir():
        if not file.is_file():
            continue

        name = file.stem.lower()
        ext = file.suffix

        if name == "cam_front":
            dst = front_dir / f"dl_{i}{ext}"
            shutil.move(str(file), str(dst))

        elif name == "cam_left":
            dst = left_dir / f"dl_{i}{ext}"
            shutil.move(str(file), str(dst))

        elif name == "cam_right":
            dst = right_dir / f"dl_{i}{ext}"
            shutil.move(str(file), str(dst))

    print(f"Procesado dl_{i}")

print("Terminado.")