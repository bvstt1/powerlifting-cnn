"""Resume las secuencias temporales BP left/right ya congeladas.

Genera informes de calidad, cobertura y procedencia a partir de los
artefactos de extracción; no examina etiquetas ni entrena clasificadores.
"""
import csv
import json
from pathlib import Path

from .run import ROOT, CODE, codigo_sha, paths, sha, write_csv, write_json
from .schema import esquema, NAMES


def detalle(view):
    p=paths(view);out=p["out"]
    s=json.loads((out/"integrity.json").read_text(encoding="utf-8"))
    pilot=json.loads((out/"pilot_audit.json").read_text(encoding="utf-8"))
    audit=json.loads((out/"source_audit.json").read_text(encoding="utf-8"))
    frozen=json.loads((out/"frozen_config.json").read_text(encoding="utf-8"))
    if frozen["code_sha256"]!=codigo_sha() or s["errors"] or pilot["failures"]:
        raise RuntimeError("No hay datos auditados y congelados")
    manifest=list(csv.DictReader(open(out/"features_manifest.csv",encoding="utf-8-sig",newline="")))
    original=list(csv.DictReader(open(out/"source_audit.csv",encoding="utf-8-sig",newline="")))
    if len(manifest)!=s["videos"]:
        raise RuntimeError("Manifest incompleto")
    write_csv(out/"quality.csv",manifest,list(manifest[0]))
    write_json(out/f"bp_{view}_features_schema.json",esquema())
    byid={r["video_id"]:r for r in original}
    verify_rows=[{"video_id":r["video_id"],"status":r["status"],"T":r["T"],
                  "raw_sha256":r["raw_sha256"],"pose_sha256":r["pose_sha256"],
                  "features_sha256":r["out_sha256"],"errors":r["error"]} for r in manifest]
    write_csv(out/"integrity_per_video.csv",verify_rows,list(verify_rows[0]))
    sync=[{"video_id":r["video_id"],"T_raw":r["T"],"pose_shape":r["pose_shape"],"fps":r["fps"],
           "status":r["sync_status"],"historical_alignment":"unverified"} for r in original]
    write_csv(out/"sync_verification.csv",sync,list(sync[0]))
    disponibilidad=[{"index":n,"feature":f,"available_percent":round(100*s["availability_by_feature"][f],4),
                     "missing_percent":round(100*(1-s["availability_by_feature"][f]),4)}
                    for n,f in enumerate(NAMES)]
    write_csv(out/"availability_by_feature.csv",disponibilidad,list(disponibilidad[0]))
    warnings=[]
    for r in manifest:
        t=int(r["T"]); reason=[]
        if int(r["bar_ambiguous"])>=.2*t:reason.append("bar_ambiguous_ge20pct")
        if int(r["bar_corrob"])<.2*t:reason.append("bar_wrist_corrob_lt20pct")
        if int(r["profile"])<.2*t:reason.append("torso_profile_lt20pct")
        if int(byid[r["video_id"]]["missing_bench_frames"])>=.5*t:reason.append("bench_missing_ge50pct")
        if reason:
            warnings.append({"video_id":r["video_id"],"split":r["split"],"T":t,
                             "reasons":"|".join(reason),"bar_corrob":r["bar_corrob"],
                             "bar_ambiguous":r["bar_ambiguous"],"profile":r["profile"],
                             "missing_bench":byid[r["video_id"]]["missing_bench_frames"]})
    write_csv(out/"attention.csv",warnings,list(warnings[0]) if warnings else
              ["video_id","split","T","reasons","bar_corrob","bar_ambiguous","profile","missing_bench"])
    scen={"many_bar":any(int(r["multiple_bar_frames"])>0 for r in original if r["video_id"] in pilot["pilot_ids"]),
          "person_absent":any(int(r["missing_person_frames"])>0 for r in original if r["video_id"] in pilot["pilot_ids"]),
          "bench_absent":any(int(r["missing_bench_frames"])>0 for r in original if r["video_id"] in pilot["pilot_ids"]),
          "pose_torso_unavailable":any(int(r["missing_torso_pose_frames"])>0 for r in original if r["video_id"] in pilot["pilot_ids"]),
          "bench_fallback_observed":sum(int(r["profile_bench"]) for r in pilot["metrics"].values()),
          "short_projection":sum(int(r["ordinary"]) for r in pilot["metrics"].values()),
          "far_projection":sum(int(r["far"]) for r in pilot["metrics"].values()),
          "unevaluable":sum(int(r["unevaluable"]) for r in pilot["metrics"].values())}
    pilot["scenario_coverage"]=scen
    write_json(out/"pilot_audit.json",pilot)

    detail="\n".join(f'| {n} | {round(100*s["availability_by_feature"][n],2)}% | {esquema()["features"][i]["branch"]} | {esquema()["features"][i]["formula"]} |'
                      for i,n in enumerate(NAMES))
    attention_dev=[x for x in warnings if x["split"] in ("train","validation")]
    top=sorted(attention_dev,key=lambda r:(-len(r["reasons"].split("|")),r["video_id"]))[:12]
    report=f"""# Reporte BP_{view.upper()} Temporal Features V1

Estado: **{view.upper()}_FEATURES_READY_FOR_TRAINING** (secuencias extraídas y verificadas; imputación/escalado solo en Train durante el entrenamiento posterior).

## Cohorte y fuentes

- Videos y secuencias: **{s['videos']}/{s['videos']}**; frames originales **{s['frames']}**; forma por video **(T,{s['F']})** sin remuestreo. Porcentaje global de valores faltantes (incluye QUALITY siempre observable): **{s['missing_percent']:.4f}%** ({s['missing_values']} de {s['frames']*s['F']}). No hay Inf, los NaN mantienen `availability=0` y `provenance=0`.
- RAW `../raw/`, pose YOLO `keypoints/bp/yolo/{view}/`, MP4 `dataset/bp/{view}/`; checkpoint original SHA-256 `{audit['checkpoint_sha256']}`. No se reejecutó YOLO. Configuración y SHA-256 del extractor en `frozen_config.json`; hashes por muestra en `features_manifest.csv` e `integrity_per_video.csv`.
- `source_audit.csv` y `sync_verification.csv`: por cada video, pose `(T,8,3)`, índices RAW 0..T-1, 1920×1080, 30 fps y `timestamp=frame_index/fps`. La sincronía histórica no se certifica solo con coincidencia T/FPS. Orden operativo de pose: 0 rodilla, 1 tobillo, 2 cadera, 3 hombro, 4 mano, 5 codo, 6 talón, 7 punta. Sin anotaciones YOLO de entrenamiento originales para certificación anatómica retrospectiva.
- Se completó una primera pasada y, al revisar el contrato del código, se corrigió la dependencia innecesaria de BAR para medir el perfil PERSON guiado por pose. Se repitieron pruebas y pilotos y se regeneraron **todas** las secuencias derivadas con el hash congelado final; ninguna secuencia de la primera pasada integra esta cohorte. No se modificaron RAW, pose, splits ni etiquetas.
- Split PoseCNN histórico: {s['splits']}. Cohorte supervisada común con label/pose/RAW/features: **{s['supervised_cohort']}**. Sin asignación: **{', '.join(s['unassigned_ids'])}**; mantienen features descriptivas pero quedan fuera de entrenamiento. No se cambió ningún split.
- Duplicados MP4 SHA-256 verificados: {audit['exact_video_duplicates']}. Ninguno cruza particiones supervisadas. Retirados *solo de cohorte supervisada derivada*, preservando RAW, etiquetas y split: {s['duplicates_removed_from_supervised']}.
- Identidad física entre LEFT/RIGHT: **no verificada**; mismo número no establece correspondencia ni agrupamiento.

## Resultados geométricos

| Estado | Frames |
|---|---:|
| BAR RAW ausente / PERSON RAW ausente / BENCH RAW ausente | {s['raw_missing']['bar']} / {s['raw_missing']['person']} / {s['raw_missing']['bench']} |
| BAR seleccionada provisionalmente | {s['bar_selected']} |
| De ellas, muñeca corroborada / identidad no verificada | {s['bar_wrist_corrob']} / {s['bar_selected_unverified']} |
| BAR detectada sin selección / múltiples plausibles ambiguas | {s['bar_unverified_unselected']} / {s['bar_ambiguous']} |
| Perfil torso de pose / estimado por banco | {s['profile_pose']} / {s['profile_bench_estimate']} |
| Separación observada directamente en máscara / proyección ordinaria / lejana | {s['relation_direct_observed']} / {s['relation_ordinary_projected']} / {s['relation_far_projected']} |
| Relación no evaluable | {s['relation_unevaluable']} |
| Velocidad con asociación local provisional respaldada | {s['velocity']} |
| Referencia temporal BENCH utilizada para perfil (sin inventar detección) | {s['bench_reference_used']} |

La región se construye entre 45–75% de cadera→hombro y toma el borde superior de la máscara PERSON; **no** es una segmentación anatómica del pecho. El extremo proximal procede de la franja extrema visible BAR; el grosor no es radio físico. `sep_signed_h=(y_silueta-y_barra)/H` (positivo = barra proyectada por encima). Directa solo si BAR tiene píxeles en x_torso. Proyectada: `y_barra=y_extremo+grosor_aprox+a*(x_torso-x_extremo)`. D/L≤2 ordinaria; 2<D/L≤4 lejana; D/L>4 abstención. Sin contacto físico certificado. La pendiente conserva su signo según orientación real; no flips. Con máscara/pose insuficiente, se conserva NaN. Dinámica solo con barras únicas corroboradas en dos frames consecutivos y gating bbox IoU/centro/área; no son tracks persistentes.

## Piloto de Train y revisión

- Videos de desarrollo expuestos: **{', '.join(pilot['pilot_ids'])}**; `pilot_exposure.csv`. Sin Test ni etiquetas usadas para diseñar o seleccionar entidades. Verificación de reconstrucción/índices/procedencia, pruebas sintéticas y repetición determinista de extracción. Errores críticos: **{len(pilot['failures'])}**.
- Casos observados en piloto: múltiples BAR={scen['many_bar']}, PERSON ausente={scen['person_absent']}, BENCH ausente={scen['bench_absent']}, pose torso ausente={scen['pose_torso_unavailable']}. Respaldo torso por banco={scen['bench_fallback_observed']} frames; proyección ordinaria={scen['short_projection']}, lejana={scen['far_projection']}, no evaluable={scen['unevaluable']}. Los casos que no ocurrieron en el piloto se cubrieron sintéticamente o quedaron como limitación de representatividad.
- Previews: `previews/<video>/frame_XXXXX.jpg` (original a la izquierda; máscaras RAW, 8 puntos pose, barra, perfil, extremo, eje y separación a la derecha). Las inferencias del visor con `retina_masks=False` pueden diferir en píxeles de los RAW `retina_masks=True`.
- Atención descriptiva en Train/Validation, hasta 12 ejemplos: {[(r['video_id'],r['reasons']) for r in top]}. Lista completa **{len(warnings)}** videos en `attention.csv`, con split. Estos avisos no borran muestras ni alteran labels.

## Contrato de 62 canales

`bp_{view}_features_schema.json` contiene fórmula, índice y dependencia de pose de cada canal. Una medición BAR calculada tras selección por muñeca es **POSE-ASSISTED**, no modalidad de segmentación pura. `availability` y `provenance` se guardan por canal; `bar_state`, `profile_state`, `relation_state`, `raw_instance_ids` y `raw_class_counts` por frame. Procedencia: 0 faltante, 1 máscara/caja, 2 asistido por pose, 3 estimación banco, 4/5 proyección ordinaria/lejana, 6 referencia banco, 7 asociación local, 8 calidad (puede depender de pose). Para ablación SEG-only filtrar por dependencia de pose y rediseñar selección de BAR sobre Train, sin llamar a esta rama independiente de pose.

| Canal | Disponible | Rama | Fórmula |
|---|---:|---|---|
{detail}

## Recomendación

Es posible preparar SegCNN **{view.upper()}** como modelo monovista separado, con máscaras `availability`, señales de calidad y procedencia; imputador y scaler estadístico se ajustarán solo en Train. Proteger evaluación Test de ajustes futuros. La selección descriptiva no confirma identidad sostenida; alta tasa de proyecciones no significa alta precisión de juicio. Medir una futura ablación SEG-only/POSE-ASSISTED y desempeño por calidad, exclusivamente con development antes de test.
"""
    (out/f"Reporte_BP_{view.upper()}_Temporal_Features_V1.md").write_text(report,encoding="utf-8")
    return s,scen,len(warnings)


def main():
    left,sl,wl=detalle("left")
    right,sr,wr=detalle("right")
    summary=f"""# Resumen documental BP lateral Temporal Features V1

Sistemas y cohortes **LEFT y RIGHT independientes**. No se creó dataset ni modelo SIDE. No se entrenó CNN ni se reejecutó YOLO.

| Vista | Secuencias | Frames | F | NaN global | BAR corroborada | BAR seleccionada no verificada | BAR ambigua | Separación directa / proyectada ordinaria / lejana | Velocidad | Cohorte supervisada |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| LEFT | {left['videos']} | {left['frames']} | {left['F']} | {left['missing_percent']:.2f}% | {left['bar_wrist_corrob']} | {left['bar_selected_unverified']} | {left['bar_ambiguous']} | {left['relation_direct_observed']} / {left['relation_ordinary_projected']} / {left['relation_far_projected']} | {left['velocity']} | {left['supervised_cohort']} |
| RIGHT | {right['videos']} | {right['frames']} | {right['F']} | {right['missing_percent']:.2f}% | {right['bar_wrist_corrob']} | {right['bar_selected_unverified']} | {right['bar_ambiguous']} | {right['relation_direct_observed']} / {right['relation_ordinary_projected']} / {right['relation_far_projected']} | {right['velocity']} | {right['supervised_cohort']} |

LEFT sin split {len(left['unassigned_ids'])}; RIGHT sin split {len(right['unassigned_ids'])}. Duplicado exacto entre dos videos Train RIGHT: bp_354 excluido solo en cohorte derivada como réplica de bp_027. No hay duplicados exactos entre particiones dentro de una vista. Identidad multivista no verificada. Ambos pilotos Train de 10 videos pasaron; atención descriptiva LEFT {wl}, RIGHT {wr} videos; ver reportes monovista y CSV respectivos. La geometría barra–torso se basa principalmente en extrapolación 2D, no en observación directa ni en contacto anatómico. Las velocidades se abstienen cuando la identidad no está corroborada. Los canales laterales asistidos por pose no constituyen modalidad completamente independiente.

Tras una revisión del código se repitieron pilotos y se regeneraron por completo ambas cohortes derivadas para que el perfil PERSON sea medible sin BAR seleccionada cuando pose y máscara lo permitan. Los informes reflejan únicamente el hash final congelado.

Estados: **LEFT_FEATURES_READY_FOR_TRAINING**; **RIGHT_FEATURES_READY_FOR_TRAINING**. El entrenamiento posterior debe respetar las cohortes monovista y ajustar imputación/escalado únicamente con Train. No se tomó decisión con Test para alterar la configuración congelada.
"""
    (ROOT/"segmentation/bp/Reporte_BP_Laterales_Temporal_Features_Resumen.md").write_text(summary,encoding="utf-8")
    print("Reportes emitidos; LEFT",left["videos"],"RIGHT",right["videos"])


if __name__=="__main__":main()
