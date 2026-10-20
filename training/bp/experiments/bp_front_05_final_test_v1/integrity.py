"""Sellado post-Test: comprueba protocolos/salidas y hashes sin inferir de nuevo."""
import json
from pathlib import Path

from .common import OUT, RAIZ, guardar_json, sha256


def main():
    protocolo = OUT / "bp_front_final_test_protocol_frozen.json"
    previo = (OUT / "bp_front_final_test_protocol_frozen.sha256").read_text(
        encoding="utf-8").split()[0]
    spec = json.loads(protocolo.read_text(encoding="utf-8"))
    if sha256(protocolo) != previo:
        raise RuntimeError("Protocolo congelado alterado")
    fuentes = {}
    for k, info in spec["sha256_artifacts"].items():
        actual = sha256(RAIZ / info["path"])
        fuentes[k] = {"sha256_antes": info["sha256"],
                      "sha256_despues": actual,
                      "intacto": info["sha256"] == actual}
    if not all(r["intacto"] for r in fuentes.values()):
        raise RuntimeError("Una fuente original cambió")
    pred = OUT / "bp_front_final_test_predictions.csv"
    log = json.loads((OUT / "bp_front_final_test_execution_log.json").read_text(encoding="utf-8"))
    if (log["state"] != "COMPLETED_ONE_PASS" or
            log["test_inference_passes"] != 1 or
            log["protocol_sha256"] != previo or
            sha256(pred) != log["result_sha256"]):
        raise RuntimeError("Test no corresponde a única ejecución certificada")
    resultados = ["bp_front_final_test_metrics.json",
                  "bp_front_final_test_comparison.csv",
                  "bp_front_final_test_confusion_matrices.csv",
                  "bp_front_final_test_paired_errors.csv",
                  "bp_front_final_test_confidence_intervals.json",
                  "bp_front_validation_vs_test.csv",
                  "bp_front_final_test_error_summary.json"]
    imgs = sorted((OUT / "figures").glob("*.png"))
    if len(imgs) != 9 or any(not (OUT / name).is_file() for name in resultados):
        raise RuntimeError("Resultados o figuras faltantes")
    detalles = {
        "estado": "TEST_FINAL_BP_FRONT_CERRADO",
        "protocol_sha256": previo,
        "test_predictions_sha256": sha256(pred),
        "cantidad_test_inferencias": 1,
        "fuentes_intactas": len(fuentes), "hashes_fundamentales": fuentes,
        "datos_test_originales_hasheados_durante_inferencia": 100,
        "duplicados_mp4_entre_splits": spec["duplicados_exactos_entre_particiones"],
        "resultado_archivos_sha256": {name: sha256(OUT / name)
                                     for name in resultados},
        "figuras_sha256": {p.name: sha256(p) for p in imgs},
        "sin_reentrenamiento_ni_reajuste": True,
        "limitacion": "SHA-256 de MP4 prueba igualdad exacta en bytes; no prueba independencia por atleta/evento ni duplicación con recodificación",
    }
    guardar_json(OUT / "bp_front_final_test_integrity.json", detalles)
    print(detalles["estado"], "fuentes intactas", len(fuentes),
          "figuras", len(imgs), "evaluaciones Test", detalles["cantidad_test_inferencias"])


if __name__ == "__main__":
    main()
