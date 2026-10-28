"""Juez virtual de powerlifting: punto de entrada.

Con cámaras (3 vistas, el juez controla el levantamiento con el teclado):
    python -m judge.main --exercise squat
Con 3 videos ya grabados (todo el video es un levantamiento):
    python -m judge.main --exercise bench --videos frontal.mp4 izquierda.mp4 derecha.mp4 --stride 15
Utilidades:
    python -m judge.main --list-cameras

Teclas (configurables en config.yaml, sección controls): espacio inicia y termina el levantamiento,
c lo cancela, q sale.
"""

import argparse
import sys

import cv2

from judge.attempt_log import AttemptLog
from judge.capture import CameraError, MultiCameraCapture
from judge.capture.video_source import VideoSource
from judge.config import EXERCISES, VIEWS, ConfigError, load_config
from judge.evaluation import LiftEvaluator
from judge.fusion import FusionRule
from judge.keypoints import KeypointStage
from judge.pipeline import JudgePipeline
from judge.segmentation import SegmentationStage
from judge.ui import compose_mosaic, status_text

WINDOW = "Juez virtual"
EXIT_OK, EXIT_CONFIG, EXIT_CAPTURE = 0, 2, 3


def warnings_for(config):
    """Avisos de arranque sobre redes que no pueden decidir."""
    notes = []
    for view, item in config.views().items():
        if not item.evaluator.verified and not config.allow_unverified:
            notes.append(f"AVISO: la red de la vista {view} no está verificada; esa vista responderá "
                         f"'sin datos suficientes'. Motivo: {item.evaluator.verification_note}")
    return notes


def build_pipeline(config, device="cpu", segmentation=True, threaded=False, min_interval=0.0, log=True):
    """Pipeline con los modelos reales del ejercicio activo."""
    attempt_log = AttemptLog(config.pipeline.log_directory, config) if log else None
    use_segmentation = segmentation and config.exercise == "bench"
    return JudgePipeline(config, KeypointStage(config), LiftEvaluator(config, device), FusionRule.from_config(config),
                         SegmentationStage(config) if use_segmentation else None, attempt_log, threaded, min_interval)


def announce(attempt, out=print):
    out("")
    out(attempt.decision.summary())
    out(f"  frames evaluados: {attempt.recording.frames} | duración: {attempt.recording.duration_s:.1f} s"
        + (" | TRUNCADO" if attempt.recording.truncated else "")
        + (f" | frames con error de extracción: {attempt.errors}" if attempt.errors else ""))
    if attempt.path:
        out(f"  registro: {attempt.path}")
    out("")


def run_batch(pipeline, source, out=print):
    """Evalúa videos grabados: todo el video es un único levantamiento."""
    pipeline.start_lift()
    sets = 0
    while True:
        frame_set = source.read()
        if frame_set is None:
            break
        pipeline.offer(frame_set, force=True)
        sets += 1
        out(f"  conjunto {sets}: {pipeline.frame_counts()}")
    attempt = pipeline.finish_lift()
    announce(attempt, out)
    return attempt


def run_interactive(config, pipeline, source, show=cv2.imshow, wait_key=cv2.waitKey, out=print, max_iterations=None):
    """Ventana con las 3 vistas; el juez inicia, termina o cancela el levantamiento con el teclado."""
    codes = config.key_codes()
    raw, last_attempt, mosaic = {}, None, None
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        iterations += 1
        frame_set = source.read(0.1)
        if frame_set is not None:
            raw = frame_set.images()
            pipeline.offer(frame_set)
        images = pipeline.last_annotated if (pipeline.last_annotated and (pipeline.recording or config.pipeline.preview_while_idle)) else raw
        status = status_text(config.exercise, pipeline.state, pipeline.frame_counts(), config.controls, pipeline.skipped)
        if images:
            mosaic = compose_mosaic(images, status, pipeline.state, last_attempt.decision if last_attempt else None)
            show(WINDOW, mosaic)
        key = wait_key(1) & 0xFF
        if key == codes["quit"]:
            if pipeline.recording:
                pipeline.cancel_lift()
            break
        if key == codes["cancel"] and pipeline.recording:
            pipeline.cancel_lift()
            out("Levantamiento cancelado.")
        elif key == codes["start_stop"]:
            if pipeline.recording:
                last_attempt = pipeline.finish_lift(mosaic)
                announce(last_attempt, out)
            else:
                last_attempt = None
                pipeline.last_annotated = None
                pipeline.start_lift()
                out("Levantamiento iniciado.")
    return last_attempt


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="ruta de config.yaml (por defecto judge/config.yaml)")
    parser.add_argument("--exercise", choices=EXERCISES, help="ejercicio (por defecto el de config.yaml)")
    parser.add_argument("--videos", nargs=3, metavar=("FRONTAL", "IZQUIERDA", "DERECHA"),
                        help="evaluar 3 videos grabados en lugar de usar cámaras")
    parser.add_argument("--stride", type=int, default=1, help="con --videos, procesa 1 de cada N frames")
    parser.add_argument("--no-segmentation", action="store_true", help="no segmentar la barra (más rápido)")
    parser.add_argument("--no-log", action="store_true", help="no guardar el registro del intento")
    parser.add_argument("--device", default="cpu", help="dispositivo de PyTorch para las redes de evaluación")
    parser.add_argument("--list-cameras", action="store_true", help="listar los índices de cámara y salir")
    return parser.parse_args(argv)


def main(argv=None, out=print):
    args = parse_args(argv)
    try:
        config = load_config(args.config, exercise=args.exercise)
    except (ConfigError, OSError) as error:
        out(f"Configuración inválida:\n{error}")
        return EXIT_CONFIG

    if args.list_cameras:
        from judge.capture.camera import probe_cameras
        found = probe_cameras(config.capture)
        for camera in found:
            out(f"Cámara {camera['index']}: {camera['width']}x{camera['height']} @ {camera['fps']:.0f} FPS")
        out(f"{len(found)} cámaras disponibles; configuradas: {config.capture.cameras}")
        return EXIT_OK

    for note in warnings_for(config):
        out(note)
    out(f"Ejercicio: {config.exercise}")
    batch = args.videos is not None
    interval = 0.0 if batch else 1.0 / config.capture.process_fps
    try:
        source = (VideoSource(dict(zip(VIEWS, args.videos)), args.stride) if batch
                  else MultiCameraCapture(config.capture))
        with source:
            with build_pipeline(config, args.device, not args.no_segmentation, threaded=not batch,
                                min_interval=interval, log=not args.no_log) as pipeline:
                if batch:
                    run_batch(pipeline, source, out)
                else:
                    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                    try:
                        run_interactive(config, pipeline, source, out=out)
                    finally:
                        cv2.destroyAllWindows()
    except CameraError as error:
        out(f"Error de captura: {error}")
        return EXIT_CAPTURE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

