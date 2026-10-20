"""Migración única de los experimentos BP de la raíz de training/.

Ejecutar desde la raíz: python -B -m training.legacy.relocate_bp_experiments
Usar --apply solo después de revisar el inventario. No modifica resultados
congelados ni sus manifests: esas verificaciones pertenecen a las rutas viejas.
"""

import argparse
import hashlib
import re
import shutil
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[2]
TRAINING = RAIZ / "training"
DESTINO = TRAINING / "bp" / "experiments"
NOMBRES = (
    "bp_front_cross_validation_5fold_v1", "bp_front_final_test_v1",
    "bp_front_fusion_v1", "bp_front_posecnn_common_v1", "bp_front_segcnn_v1",
    "bp_lateral_final_test_v1", "bp_left_cv_v1", "bp_left_final_test_v1",
    "bp_left_fusion_v1", "bp_left_posecnn_compatibility_v1",
    "bp_left_segcnn_v1", "bp_posecnn_common_v1", "bp_right_cv_v1",
    "bp_right_final_test_v1", "bp_right_fusion_v1",
    "bp_right_posecnn_common_v1", "bp_right_segcnn_v1", "bp_side_cv_v1",
    "bp_side_fusion_v1", "bp_side_segcnn_v1",
)


def digest(ruta):
    sha = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha.update(bloque)
    return sha.digest()


def actualizar_python(texto, movido):
    # Cambiar referencias exactas, sin tocar otros paquetes training.bp.*.
    nombres = "|".join(map(re.escape, NOMBRES))
    texto = re.sub(r"training\.(" + nombres + r")\b",
                   r"training.bp.experiments.\1", texto)
    texto = re.sub(r"training/(" + nombres + r")\b",
                   r"training/bp/experiments/\1", texto)
    texto = texto.replace('ROOT/"training"/f"bp_', 'ROOT/"training/bp/experiments"/f"bp_')
    texto = texto.replace('ROOT / "training" / f"bp_',
                          'ROOT / "training/bp/experiments" / f"bp_')
    if movido:
        texto = texto.replace('Path(__file__).resolve().parents[2]',
                              'Path(__file__).resolve().parents[4]')
        texto = texto.replace('FUSION_DIR.parents[1]', 'FUSION_DIR.parents[3]')
        texto = texto.replace('SALIDA.parents[1]', 'SALIDA.parents[3]')
        texto = texto.replace('OUT.parents[1]', 'OUT.parents[3]')
        texto = texto.replace('p["left_out"].parent/"common/train_utils.py"',
                              'ROOT/"training/common/train_utils.py"')
        texto = texto.replace('p["left_out"].parent/"bp/train_posecnn.py"',
                              'ROOT/"training/bp/train_posecnn.py"')
    return texto


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--docs-only", action="store_true",
                        help="Actualizar índices actuales sin tocar informes históricos")
    args = parser.parse_args()
    assert TRAINING.is_dir() and TRAINING / "bp" == DESTINO.parent
    if args.docs_only:
        for nombre in ("EXPERIMENTOS.md", "INVENTARIO_REDES.md"):
            ruta = TRAINING / "docs" / nombre
            texto = ruta.read_text(encoding="utf-8")
            nuevo = actualizar_python(texto, False)
            if args.apply and nuevo != texto:
                ruta.write_text(nuevo, encoding="utf-8")
            print(f"{nombre}: {'actualizado' if args.apply else 'revisado'}")
        return
    pendientes = [nombre for nombre in NOMBRES if (TRAINING / nombre).is_dir()]
    ocupados = [nombre for nombre in NOMBRES if (DESTINO / nombre).exists()]
    if ocupados or len(pendientes) != len(NOMBRES):
        raise RuntimeError(f"Origen/destino inesperados: {pendientes=}, {ocupados=}")
    cambios = []
    for ruta in TRAINING.rglob("*.py"):
        if ruta == Path(__file__) or "__pycache__" in ruta.parts:
            continue
        # Antes de mover, identificar si el archivo está en una carpeta migrada.
        movido = ruta.relative_to(TRAINING).parts[0] in NOMBRES
        original = ruta.read_text(encoding="utf-8")
        nuevo = actualizar_python(original, movido)
        if original != nuevo:
            compile(nuevo, str(ruta), "exec")
            cambios.append((ruta, nuevo))
    print(f"Carpetas completas: {len(pendientes)}; scripts a actualizar: {len(cambios)}")
    if not args.apply:
        return
    DESTINO.mkdir(parents=True, exist_ok=False)
    # Comprobar que cada byte migrado sobrevive igual antes de alterar código.
    originales = {}
    for nombre in pendientes:
        for ruta in (TRAINING / nombre).rglob("*"):
            if ruta.is_file():
                originales[(nombre, ruta.relative_to(TRAINING / nombre))] = digest(ruta)
        shutil.move(str(TRAINING / nombre), str(DESTINO / nombre))
    for (nombre, relativo), sha in originales.items():
        if digest(DESTINO / nombre / relativo) != sha:
            raise RuntimeError(f"El movimiento alteró bytes: {nombre}/{relativo}")
    for ruta, nuevo in cambios:
        movida = ruta.relative_to(TRAINING)
        destino = (DESTINO / movida) if movida.parts[0] in NOMBRES else ruta
        destino.write_text(nuevo, encoding="utf-8", newline="")
    print(f"Migrados {len(pendientes)} directorios y {len(originales)} archivos sin pérdida de bytes; "
          f"{len(cambios)} scripts adaptados. Manifiestos históricos intactos.")


if __name__ == "__main__":
    main()
