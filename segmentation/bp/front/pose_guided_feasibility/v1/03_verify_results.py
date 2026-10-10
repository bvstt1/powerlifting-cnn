"""Verificación reproducible de resultados y reporte; fuentes en modo lectura."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def read_csv(name):
    with (HERE / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fmt(value):
    try:
        return '%.1f' % float(value)
    except (ValueError, TypeError):
        return 'N/D'


def score(value):
    try:
        return '%.4f' % float(value)
    except (ValueError, TypeError):
        return 'N/D'


def verify():
    handoffs = read_csv('bp_front_pose_guided_handoff_metrics.csv')
    audit = read_csv('bp_front_pose_guided_sync_audit.csv')
    identity = read_csv('bp_front_pose_guided_track_identity_metrics.csv')
    manifest_path = HERE / 'bp_front_pose_guided_experiment_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if len(audit) != 76 or len(handoffs) != 96 or len(identity) == 0:
        raise AssertionError('Población de Validation76 no coincide')
    selected = [r for r in handoffs if r['selected_by_v3'] == '1']
    rejected = [r for r in handoffs if r['selected_by_v3'] == '0']
    correct = [r for r in selected if r['visual_review_status'] == 'VISUALLY_CONFIRMED' and r['false_merge_detected'] == '0']
    false = [r for r in selected if r['visual_review_status'] == 'VISUAL_PROBLEM' and r['false_merge_detected'] == '1']
    assert (len(selected), len(rejected), len(correct), len(false)) == (79, 17, 77, 2)
    assert {r['video_id'] for r in false} == {'bp_663'}
    assert all(r['sync_status'] == 'SEQUENCE_CONSISTENT_OVERLAY_PENDING' for r in audit)
    assert all(r['visual_review_status'] == 'NOT_APPLICABLE' for r in rejected)
    for r in handoffs:
        if r['review_evidence'] and not (ROOT/r['review_evidence']).is_file():
            raise AssertionError('Review visual inexistente: '+r['review_evidence'])
    for vid in ('bp_663', 'bp_662', 'bp_182', 'bp_238', 'bp_538', 'bp_540', 'bp_544', 'bp_102', 'bp_103'):
        if not list((HERE/'previews'/vid).glob('*.jpg')):
            raise AssertionError('Falta preview '+vid)
    after = {relative: sha(ROOT/relative) for relative in manifest['hashes_before']}
    differences = [name for name, value in after.items() if manifest['hashes_before'][name] != value]
    if differences or manifest['missing_sources']:
        raise AssertionError('Integridad fallida: '+repr(differences)+' / ausentes '+repr(manifest['missing_sources']))
    manifest['hashes_after'] = after
    manifest['source_hashes_match'] = True
    manifest['status'] = 'PROMISING_EXPLORATORY_NOT_APPROVED_FOR_TRACKER'
    manifest['scope'] = 'Validation76 solamente; no inferencia ni entrenamiento; heurística evaluada sobre los mismos casos explorados'
    manifest['preview_count'] = len(list((HERE/'previews').rglob('*.jpg')))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    def counts(rows):
        counter = Counter(r['pose_guided_assessment'] for r in rows)
        return [counter['POSE_SUPPORTS_SAME_BAR'], counter['POSE_SUPPORTS_DIFFERENT_OBJECT'],
                counter['POSE_INSUFFICIENT_EVIDENCE']]

    result = {'correct_accepted': counts(correct), 'false_accepted': counts(false),
              'not_selected_unreviewed': counts(rejected)}
    total_frames = sum(int(r['video_frames']) for r in audit)
    both_frames = sum(int(r['pose_both_valid_frames']) for r in audit)
    summary = [
        {'measure': 'status', 'value': 'PROMISING (exploratorio; no aprobación V3.1)'},
        {'measure': 'videos_validation76', 'value': len(audit)},
        {'measure': 'video_frames', 'value': total_frames},
        {'measure': 'both_wrist_valid_frames', 'value': both_frames},
        {'measure': 'both_wrist_coverage', 'value': '%.6f' % (both_frames/total_frames)},
        {'measure': 'selected_v3', 'value': len(selected)},
        {'measure': 'not_selected_v3_without_visual_reference', 'value': len(rejected)},
        {'measure': 'correct_same', 'value': result['correct_accepted'][0]},
        {'measure': 'correct_different_false_reject_risk', 'value': result['correct_accepted'][1]},
        {'measure': 'correct_indeterminate', 'value': result['correct_accepted'][2]},
        {'measure': 'false_merges_identified', 'value': result['false_accepted'][1]},
        {'measure': 'files_hashed_before_after', 'value': len(after)},
        {'measure': 'source_hashes_match', 'value': True},
        {'measure': 'generated_previews', 'value': manifest['preview_count']},
    ]
    with (HERE/'bp_front_pose_guided_feasibility_summary.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['measure', 'value']); w.writeheader(); w.writerows(summary)

    def row(vid, source, target):
        return next(r for r in handoffs if r['video_id'] == vid and r['source_track'] == str(source) and r['target_track'] == str(target))

    critical = [row('bp_663', 0, 4), row('bp_663', 4, 5), row('bp_662', 0, 4), row('bp_662', 4, 5)]
    cases = []
    for r in critical:
        cases.append('| %s %s→%s @%s | %s | %s/%s→%s/%s | %s/%s→%s/%s | %s / %s | %s/%s | %s | %s | %s |' % (
            r['video_id'], r['source_track'], r['target_track'], r['transition_frame'], score(r['total_edge_score']),
            fmt(r['source_left_wrist_bar_distance']), fmt(r['source_right_wrist_bar_distance']),
            fmt(r['left_wrist_bar_distance']), fmt(r['right_wrist_bar_distance']),
            fmt(r['source_left_axis_distance']), fmt(r['source_right_axis_distance']),
            fmt(r['left_wrist_axis_distance']), fmt(r['right_wrist_axis_distance']),
            fmt(r['left_signed_y_jump']), fmt(r['right_signed_y_jump']),
            score(r['left_wrist_confidence']), score(r['right_wrist_confidence']),
            r['temporal_motion_compatibility'], r['selected_by_v3'], r['pose_guided_assessment']))
    controls = []
    for r in correct:
        if r['video_id'] in {'bp_182', 'bp_238', 'bp_538', 'bp_540', 'bp_544'}:
            controls.append('| %s %s→%s @%s | %s | %s / %s | %s | %s |' % (
                r['video_id'], r['source_track'], r['target_track'], r['transition_frame'],
                score(r['total_edge_score']), fmt(r['left_signed_y_jump']), fmt(r['right_signed_y_jump']),
                r['pose_guided_assessment'], r['review_evidence']))
    remaining = []
    for r in rejected:
        remaining.append('| %s %s→%s @%s | %s | %s | %s |' % (
            r['video_id'], r['source_track'], r['target_track'], r['transition_frame'],
            score(r['total_edge_score']), r['pose_guided_assessment'], r['visual_review_status']))
    id_lines = []
    for r in identity:
        if r['video_id'] in {'bp_662', 'bp_663'}:
            id_lines.append('| %s | %s | %s | %s | %s | %s |' % (
                r['video_id'], r['track_id'], r['is_v2_main'], r['track_total_frames'],
                r['bar_above_wrist_frames'], r['bar_below_wrist_frames']))
    report = '''# BP_FRONT: viabilidad Pose-Guided Bar Tracking (exploratorio)

## 1. Objetivo e hipótesis
Investigar si las muñecas YOLO Pose RAW aportan evidencia **adicional** para distinguir la barra sostenida de estructuras paralelas detectadas como `bar`. Estado: **PROMISING, exclusivamente exploratorio**; no se implementó ni aprobó V3.1.

## 2. Estado V3 y referencias
Validation76: 72 `FULLY_RECOVERED`, 1 `PARTIALLY_RECOVERED`, 3 `UNCHANGED_CONSERVATIVE`, 0 `REGRESSED`; 1641/1680 frames recuperados. De 79 handoffs aceptados, 77 visualmente correctos y 2 falsos merges en `bp_663` (ambos `suspicious=0`); 17 propuestas no seleccionadas, **sin etiqueta visual de verdad**. Los estados visuales fueron usados solo en la evaluación posterior; nunca entran en las métricas ni en la decisión Pose.

## 3–5. Fuentes, auditoría RAW y sincronización
- Videos: `dataset/bp/front/bp_XXX.mp4` (hash idéntico al `video_sha256` de cada RAW).
- Pose RAW: `keypoints/bp/front/bp_XXX.npy`, `(T,6,3)` float, `x,y` normalizados y confidence por punto; índices `[0:left_elbow,1:left_shoulder,2:right_shoulder,3:right_elbow,4:right_wrist,5:left_wrist]`. Extractor: `extract_keypoints/bp/extract_keypoints_front_bp.py`, YOLO Pose `models/bp/bp_front_skeleton_v8.pt`; frame decodificado secuencialmente; `conf>=0.5` y NaN en coordenadas no fiables. **No** se utilizaron secuencias T=170 para PoseCNN.
- Segmentación: `segmentation/bp/front/raw/bp_XXX_segmentation.npz`, `frame_index`, `timestamp_sec`, `frame_image_shape_hw`, `frame_instance_offsets`, clases, confidence, bboxes píxel, `mask_bits`/`mask_byte_offsets`/`instance_mask_origin_xy`/`instance_mask_shape_hw`; máscaras recuperadas con `np.unpackbits(bitorder='little')`. Modelo de procedencia en metadata: `models/bp/bp_front_seg_v2.pt`.
- Tracks: `segmentation/bp/front/tracking_v2/raw/bp_XXX_segmentation.npz` (`instance_track_id` de V2, mismos índices/máscaras de instancia RAW); `segmentation/bp/front/tracking_v2/features/`; V3: `segmentation/bp/front/tracking_v3/validation_76/selections/` (índices seleccionados y track IDs), `audits/` (propuestas), `manifests/*handoffs.csv` y `*visual_review.csv`. Los IDs no son automáticamente identidades físicas.
- Comprobación automática en `bp_front_pose_guided_sync_audit.csv`: {nvid} videos, {frames} frames, {both} con ambas muñecas válidas ({pct:.2f}%); FPS=30, 1920×1080 en estos 76, base cero; índices consecutivos en RAW/V2/V3, shapes por frame, timestamps `i/fps`, hash del MP4, recuento exacto de frames Pose y geometría normalizada `[0,1]`; transformación Pose a píxeles `(x·W,y·H)`. Los JPG originales son BGR al leer OpenCV y se escriben conservando colores; no hay recorte/letterbox persistido en los artefactos. Superposición visual comprobada en los nueve videos prioritarios, en frames separados: manos y máscaras corresponden al video y al mismo instante observable.
- **Límite de procedencia:** Pose `.npy` no incluye timestamp, hash del video ni frame IDs internos: igualdad de T + código extractor secuencial + coherencia de movimiento y superposición visual proporcionan evidencia de sincronía, no una certificación criptográfica del procesamiento Pose original en TODOS los frames. No se corrigieron offsets ni se inventaron frames. Gaps RAW siguen vacíos. Verificación independiente de provenance Pose sigue pendiente.

## 6. Definiciones geométricas y procedimiento exploratorio
Para máscara binaria `M_t`, muñeca `w_t=(xW,yH)` y centro `c_t`:
- `d_mask(w,M)=min_{{p∈M}}||w-p||₂` (transformada euclídea, coordenada de muñeca redondeada al píxel; error de discretización ≤0.71 px). Referencia física `S=||w_L-w_R||₂` solo si ambas muñecas son fiables; bilateral = `max(d_L,d_R)/S`; conservar valores separados y confidence. La distancia sola falla si una máscara está parcialmente recortada o si dos barras flanquean la mano.
- PCA sobre los píxeles de la máscara: autovector principal `u`; `d_eje=|(w-c)×u|`, extensión = exceso de proyección `(w-c)·u` fuera del intervalo proyectado de píxeles de M. Colinealidad no determina identidad; máscara corta puede aumentar muchísimo `d_mask` por extensión sin cambiar `d_eje`.
- Desplazamiento **firmado** `q(w,M)=c_y-w_y` (y crece hacia abajo). En el borde de un handoff se comparan medianas de los **dos últimos frames reales** de origen y **dos primeros frames reales** de destino; contexto adicional: hasta seis frames reales por lado. `Δq_L, Δq_R` detecta salto inferior/superior aunque `d_mask` absoluto apenas cambie. En ventanas se computa mediana de desplazamiento euclídeo barra frente a promedio de muñecas, solo en pares consecutivos con ambas fiables; `PAUSE_OR_INSUFFICIENT` no es fallo de identidad.
- Para clasificar evidencia descriptiva: mínimo 3 observaciones bilaterales en cada ventana de seis y dos por lado del borde, `conf>=0.5` ya exigido por extractor. `DIFFERENT_OBJECT` si los dos `|Δq|>max(45 px, 0.08·min(S_pre,S_post))`, mismo signo y anchuras de las máscaras comparables (ratio >0.65). `SAME_BAR` si ambos `|Δq|<max(30 px,0.045·min(S))` y distancias al eje de ambas muñecas, antes/después, <0.15·min(S). En otros casos `INSUFFICIENT_EVIDENCE`. Estos **márgenes son exploratorios**, definidos después de inspeccionar el caso y controles; NO son parámetros de V3 ni una regla lista para desplegar. Medianas de seis y dos frames se publican juntas para exponer cambios de identidad DENTRO de un track.

## 7–8. Casos críticos (píxeles; distancias, ejes y confidence de ventanas de seis; Δq en borde de dos)
| Caso | score V3 | máscara L/R pre→post | eje L/R pre→post | Δq L / Δq R | conf L/R post | movimiento post | V3 seleccionó | Pose |
|---|---:|---:|---:|---:|---:|---|---:|---|
{cases}

En `bp_663`, los dos saltos firmados bilaterales (0→4 inferior; 4→5 superior) superan ~97 px pese a que la **distancia sin signo puede ser muy parecida**. Confidence en ambos bordes ~0.994–0.996 (confidence alta no garantiza localización perfecta). [Antes del primer handoff](previews/bp_663/0_4_frame_00010.jpg), [después](previews/bp_663/0_4_frame_00011.jpg); [antes del segundo](previews/bp_663/4_5_frame_00050.jpg), [después](previews/bp_663/4_5_frame_00051.jpg). Estructura inferior está ~120 px por debajo de la barra superior; la barra superior está en tracks 0 y 5 y **también** en parte del track 4: el track 4 mezcla instancias a lo largo del tiempo. RAW solo aporta una máscara `bar` en cada frame de esos bordes; no se inventaron máscaras simultáneas de la barra omitida. Ventanas globales de seis en el segundo borde son engañosas porque el mismo track 4 alterna entre objetos en frames 45–50; por ello se publican ambas escalas temporales.

En `bp_662`, V3 rechazó 0→4 y 4→5 (scores ~0.817 y ~0.806), mientras V2 ancló track 4. Las [previews 0→4](previews/bp_662/0_4_frame_00019.jpg) y [4→5](previews/bp_662/4_5_frame_00066.jpg) evidencian barra superior sostenida y barra inferior; `0/5` se alinean con la superior. Track 4 contiene también muchos frames de la superior: la discrepancia es una **contaminación temporal del track V2 principal**, no prueba de que todos sus frames sean barra inferior. La selección inicial y el handoff son problemas separados. Los dos rechazados carecen de ground truth de handoff en el CSV (`NOT_APPLICABLE`); esta observación de previews no los convierte en negativos confirmados de toda la población.

## 9. Identidad de track y controles exigidos
Conteo descriptivo para bp_662/663 de signo `c_y-promedio(y_muñecas)` sobre **todos** los frames BAR disponibles (no equivale a verdad visual frame a frame):
| video | track V2 | principal V2 | frames | barra arriba | barra abajo |
|---|---:|---|---:|---:|---:|
{identity}

| Control correcto | score V3 | Δq L / R (px) | Pose | revisión original |
|---|---:|---:|---|---|
{controls}

`bp_182`: 3 correctos compatibles, incluidos grandes cambios de longitud. `bp_238` correcto permanece **indeterminado** (2 frames origen y 3 destino en la ventana local; cuatro frames RAW puenteados, sin reconstrucción). `bp_538/540/544`: 3 correctos compatibles aunque `d_mask` cambia ~275–335 px por máscara parcial/completa; eje y `q` permanecen estables. [Previews controles](previews/) con video, máscaras, track, selección V3 y ambas muñecas. `bp_102/103`: previews `identity_frame_*` muestran dos estructuras paralelas y V3 sin cambio de identidad confirmado: controles cualitativos, no handoffs negativos nuevos.

## 10–14. Discriminación, 79 aceptados y 17 no seleccionados
| Referencia original (solo evaluación) | Pose SAME | Pose DIFFERENT | Pose INSUFFICIENT | N |
|---|---:|---:|---:|---:|
| Aceptados visualmente correctos | {c0} | {c1} | {c2} | 77 |
| Aceptados falsos merges | {f0} | {f1} | {f2} | 2 |
| No seleccionados, **sin verdad visual por candidato** | {r0} | {r1} | {r2} | 17 |

Los 17 no seleccionados NO se cuentan como negativos: por propuesta figuran así:
| Propuesta | score V3 | Pose | referencia original |
|---|---:|---|---|
{remaining}

De los 77 correctos, **0 aparecen incompatibles**, 76 compatibles y 1 indeterminado; ello NO implica que se recuperaría el mismo número al alterar un tracker real. Tres de 17 no seleccionados muestran evidencia de objeto diferente (dos en bp_662, uno en bp_238), catorce son indeterminados. No hay TP/FP definibles para los 17 sin revisión específica. Para los 79 aceptados, matriz 77/2 anterior usa exactamente la revisión visual original, sin crear etiquetas nuevas. Scores V3 ~0.81 en bp_663 y ~0.69–0.71 en máscaras parciales correctas: Pose firmado aporta contraste que ni score alto ni sola proximidad garantizan. La aparente separabilidad está **sesgada por exploración del mismo conjunto**.

## 15–17. Limitaciones, trade-offs y propuesta solo conceptual
Una muñeca poco fiable: no sustituir por cero ni imponer bilateralidad; registrar evidencia unilateral y estado indeterminado para la decisión actual. Ambas ausentes: indeterminado. Las 76 secuencias aquí tienen cobertura bilateral de frames excepcionalmente alta; no extrapolar a iluminación/oclusiones nuevas. Pausas: un desplazamiento cercano a cero no demuestra que la barra está equivocada. Tracklets híbridos muestran que persistencia de track ID no garantiza identidad interna. Cuando falta máscara, no estimar su ubicación. Los proxies PCA y distancias a eje son sensibles a oclusión / barras paralelas; el signo vertical puede fallar en cámaras invertidas o perspectiva distinta. El gran cambio de longitud de un mismo objeto necesita máscara parcial explícita: distancia a máscara sola produciría falsos rechazos.

**Concepto eventual Pose-Guided Tracking V3.1 (NO implementado):** un verificador externo entre propuestas V3 y publicación; entradas: Pose RAW por frame + confidence, máscaras BAR V2 RAW + índices V3 y metadata fps/resolución/hash de video. Preflight de secuencia y overlays, sin remuestreo; comparar candidatos de barra inicial por compatibilidad bilateral *y persistencia* antes de anclar (componente de selección independiente), y verificar cada borde sobre frames reales con distancias a máscara, PCA, signo vertical, movimiento medible y cambios intra-track (componente handoff). Ante datos ausentes o ambiguos mantener salida V3 sin promover asociaciones Pose nuevas y generar marca para revisión. Parámetros eventuales: mínimos de confidence/cobertura/ventana, márgenes de salto normalizados, tolerancia a recorte y calibración de orientación por cámara. Ningún número de este experimento debe desplegarse sin contrastación separada y ablation con keypoints de baja confianza, pausas, máscaras parciales, estructuras paralelas y videos nuevos no usados para ajustar la propuesta.

## 18. Integridad y 19. Conclusión
`bp_front_pose_guided_experiment_manifest.json` contiene SHA-256 **antes y después** de {hashed} fuentes: MP4, Pose RAW, segmentación RAW, tracks/features V2, selección/audits V3, modelos BP, CSV de etiquetas y código fuente. Coinciden todos; `03_verify_results.py` falla ante cualquier cambio, ausencia de preview o alteración del número/referencia de handoffs. Solo se ejecutaron los tres scripts offline de este directorio; sin YOLO, training, cambios de V3 ni generación global de features. El manifiesto no es una prueba de procesos externos al experimento.

**Conclusión:** las muñecas son prometedoras como evidencia *complementaria* mediante desplazamiento bilateral vertical firmado, no mediante distancia simple. Los dos falsos merges conocidos quedan identificados y ninguno de los 77 correctos se marca distinto en esta exploración; uno queda indeterminado. `bp_662` revela un track principal V2 parcialmente contaminado y la necesidad de evaluar separadamente selección de identidad. **No aprobar V3.1** aún: Validation76 ya se usó para inspeccionar/escoger esta combinación de características. Próximo experimento: congelar especificación, establecer procedencia verificable de Pose y evaluar sobre videos independientes (sin tocar ahora Validation17_RAW_GAPS). Si se utiliza Pose para seleccionar/tracking, llamar a esa rama *Pose-Guided Segmentation*, no *Segmentation-only*. *Pose + Segmentation* designa un clasificador multimodal posterior; resultados válido/nulo no intervinieron aquí.
'''.format(nvid=len(audit), frames=total_frames, both=both_frames, pct=100*both_frames/total_frames,
           cases='\n'.join(cases), identity='\n'.join(id_lines), controls='\n'.join(controls),
           c0=result['correct_accepted'][0], c1=result['correct_accepted'][1], c2=result['correct_accepted'][2],
           f0=result['false_accepted'][0], f1=result['false_accepted'][1], f2=result['false_accepted'][2],
           r0=result['not_selected_unreviewed'][0], r1=result['not_selected_unreviewed'][1],
           r2=result['not_selected_unreviewed'][2], remaining='\n'.join(remaining), hashed=len(after))
    (HERE/'Reporte_BP_FRONT_Pose_Guided_Tracking_Feasibility.md').write_text(report, encoding='utf-8')
    print('OK:', result, 'hashes:', len(after), 'previews:', manifest['preview_count'])


if __name__ == '__main__':
    verify()
