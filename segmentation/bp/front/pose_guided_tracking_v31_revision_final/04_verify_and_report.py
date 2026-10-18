"""Verifica copia/sources read-only y genera resultados finales exclusivamente en revisión."""
import csv
import hashlib
import json
import platform
import sys
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OLD=ROOT/'segmentation/bp/front/pose_guided_tracking_v31_prototype'
SEG=ROOT/'segmentation/bp/front'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    pre=json.loads((HERE/'revision_pre_snapshot.json').read_text(encoding='utf-8'))
    run=json.loads((HERE/'bp_front_v31_revision_manifest.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'segmentation/bp/front/pose_guided_tracking_v31_spec/BP_FRONT_Pose_Guided_Tracking_V31_Specification.md')==pre['spec_sha256']==run['spec_sha256']
    assert sha(OLD/'bp_front_v31_prototype_manifest.json')==pre['original_prototype_manifest_sha256']
    assert sha(HERE/'experimental_revision_config.json')==run['revision_config_sha256']
    assert all(sha(HERE/name)==digest for name,digest in run['output_sha256'].items())
    # Una renombrada externa posterior al snapshot se admite únicamente si su hash
    # coincide exactamente y queda registrada: igualdad de bytes NO equivale a rutas intactas.
    relocation={'extract_keypoints\\bp\\extract_keypoints_front_bp.py':
                'extract_keypoints/bp/individual/extract_keypoints_front_bp.py'}
    moved={}
    sources_after={}
    for rel,expected in pre['original_source_hashes'].items():
        path=ROOT/rel
        if not path.is_file():
            dest=relocation.get(rel)
            if dest is None or not (ROOT/dest).is_file() or sha(ROOT/dest)!=expected:
                raise FileNotFoundError('protected source missing/changed: '+rel)
            moved[rel]=dest
            path=ROOT/dest
        sources_after[rel]=sha(path)
    videos_after={vid:sha(ROOT/'dataset/bp/front'/(vid+'.mp4')) for vid in pre['inventory_video_hashes']}
    original_after={rel:sha(ROOT/rel) for rel in pre['original_prototype_files']}
    source_changed=[rel for rel,digest in pre['original_source_hashes'].items() if sources_after[rel]!=digest]
    videos_changed=[vid for vid,digest in pre['inventory_video_hashes'].items() if videos_after[vid]!=digest]
    original_changed=[rel for rel,digest in pre['original_prototype_files'].items() if original_after[rel]!=digest]
    assert not source_changed and not videos_changed and not original_changed
    old_rows=read(OLD/'bp_front_v31_selected_trajectories.csv')
    new_rows=read(HERE/'bp_front_v31_revision_selected_trajectories.csv')
    new_identity=read(HERE/'bp_front_v31_revision_frame_identity.csv')
    old_frames=read(OLD/'bp_front_v31_frame_identity.csv')
    comparisons=read(HERE/'bp_front_v31_revision_per_video_comparison.csv')
    handoffs=read(HERE/'bp_front_v31_revision_handoff_comparison.csv')
    diagnostics=read(HERE/'diagnosis_39_correct_boundary_abstentions.csv')
    visual=read(HERE/'bp_front_v31_revision_visual_review.csv')
    dominance=read(HERE/'bp_front_v31_revision_dominance_evidence.csv')
    context=read(HERE/'bp_front_v31_revision_context_evidence.csv')
    assert len(old_rows)==len(new_rows)==sum(int(r['frames']) for r in comparisons)
    assert len(new_identity)==len(old_frames) and len(comparisons)==85 and len(diagnostics)==39
    old={(r['video_id'],int(r['frame_index'])):r for r in old_rows}
    updated={(r['video_id'],int(r['frame_index'])):r for r in new_rows}
    assert set(old)==set(updated)
    valid_edges={(r['video_id'],int(r['transition_frame']),r['source_subtracklet'],r['target_subtracklet']) for r in handoffs if r['pose_decision_revised']=='POSE_COMPATIBLE'}
    retrieved=0;removed=0;new_with_v3_different=[];missing_vel=0
    for key,r in updated.items():
        original=old[key];selected=int(r['selected_global_instance_index']);prev=updated.get((key[0],key[1]-1))
        if original['selected_global_instance_index']!='-1' and selected<0:removed+=1
        if original['selected_global_instance_index']=='-1' and selected>=0:
            retrieved+=1
            if selected!=int(r['v3_baseline_instance_index']):new_with_v3_different.append(key)
        if selected<0:
            assert r['status'] in ('OBSERVED_UNVERIFIED','OBSERVED_INCOMPATIBLE','NO_BAR_DETECTED','SYNC_BLOCKED')
            assert r['position_observed']=='0' and r['velocity_valid']=='0'
            assert all(r[field]=='NaN' for field in ('bar_x_px','bar_y_px','vx_px_s','vy_px_s'))
        else:
            assert r['status']=='OBSERVED_CONFIDENT' and r['position_observed']=='1' and r['bar_identity_valid']=='1'
        if r['velocity_valid']=='1':
            assert prev and prev['selected_global_instance_index']!='-1'
            assert r['subtracklet_id']==prev['subtracklet_id'] or (key[0],key[1],prev['subtracklet_id'],r['subtracklet_id']) in valid_edges
        elif selected>=0:missing_vel+=1
    assert retrieved==203 and removed==0 and not new_with_v3_different
    # Recontar índice seleccionado contra BAR RAW: la revisión no fabrica detecciones.
    for vid in pre['development_ids']:
        with np.load(SEG/'tracking_v2/raw'/(vid+'_segmentation.npz'),allow_pickle=False) as raw:
            for t in range(sum(r['video_id']==vid for r in new_rows)):
                row=updated[(vid,t)];idx=int(row['selected_global_instance_index'])
                if idx<0:continue
                assert 0<=idx<len(raw['instance_frame_index']) and int(raw['instance_frame_index'][idx])==t
                assert raw['instance_class_name'][idx]=='bar'
                assert int(raw['instance_track_id'][idx])==int(row['v2_track_id'])
                assert int(raw['instance_original_track_id'][idx])==int(row['extractor_track_id'])
    for vid in ('bp_662','bp_663','bp_238','bp_182','bp_538','bp_540','bp_544'):
        if vid in ('bp_662','bp_663','bp_238','bp_182','bp_538','bp_540','bp_544'):
            # Los casos de riesgo no pierden ni reasignan frames respecto a V3.1 anterior.
            assert all(updated[(vid,t)]['selected_global_instance_index']==old[(vid,t)]['selected_global_instance_index'] for v,t in old if v==vid)
    assert len(visual)>=12 and all((HERE/r['preview']).is_file() for r in visual)
    with (SEG/'tracking_v3/validation_76/manifests/bp_front_tracking_v3_validation_76_handoffs.csv').open(encoding='utf-8-sig',newline='') as f:
        reference=list(csv.DictReader(f))
    val_ids={r['video_id'] for r in read(OLD/'bp_front_v31_sync_audit.csv') if r['v3_branch']=='validation_76'}
    assert len(val_ids)==76
    hedges={(r['video_id'],r['source_track'],r['target_track'],r['transition_frame']):r for r in handoffs}
    matrix=Counter();continuity_old=Counter();continuity_new=Counter();labels={}
    for row in reference:
        if row['selected_by_v3']!='1':
            assert row['visual_review_status']=='NOT_APPLICABLE'
            continue
        key=(row['video_id'],row['source_track'],row['target_track'],row['transition_frame']);edge=hedges[key]
        matrix[(row['visual_review_status'],edge['pose_decision_revised'])]+=1
        continuity_old[(row['visual_review_status'],edge['original_both_confident'])]+=1
        continuity_new[(row['visual_review_status'],edge['revision_both_confident'])]+=1
        labels[key]=row['visual_review_status']
    assert matrix[('VISUALLY_CONFIRMED','POSE_COMPATIBLE')]==76
    assert matrix[('VISUALLY_CONFIRMED','POSE_INSUFFICIENT_EVIDENCE')]==1
    assert matrix[('VISUALLY_CONFIRMED','POSE_INCOMPATIBLE')]==0
    assert matrix[('VISUAL_PROBLEM','POSE_INCOMPATIBLE')]==2
    assert continuity_old[('VISUALLY_CONFIRMED','True')]==38
    assert continuity_new[('VISUALLY_CONFIRMED','True')]==42
    assert continuity_new[('VISUAL_PROBLEM','True')]==0
    handoff_gained=[key for key,edge in hedges.items() if key in labels and labels[key]=='VISUALLY_CONFIRMED' and edge['original_both_confident']=='False' and edge['revision_both_confident']=='True']
    assert len(handoff_gained)==4
    assert all(hedges[k]['pose_decision_revised']=='POSE_COMPATIBLE' for k in handoff_gained)
    named={r['video_id']:r for r in comparisons}
    for video,expected in (('bp_518',(161,94,160)),('bp_520',(157,69,156)),('bp_663',(64,46,46)),('bp_662',(48,67,67))):
        row=named[video]
        assert tuple(int(row[k]) for k in ('v3_selected','v31_original_confident','revision_confident'))==expected
    val_sums={k:sum(int(r[k]) for r in comparisons if r['video_id'] in val_ids) for k in ('frames','v3_selected','v31_original_confident','revision_confident','revision_unverified','revision_incompatible','no_bar')}
    all_sums={k:sum(int(r[k]) for r in comparisons) for k in ('frames','v3_selected','v31_original_confident','revision_confident')}
    assert val_sums['v3_selected']==7102 and val_sums['v31_original_confident']==6940 and val_sums['revision_confident']==6989
    assert all_sums['revision_confident']-all_sums['v31_original_confident']==203
    assert all(int(r['intratrack_splits'])==int(next(x for x in read(OLD/'bp_front_v31_comparison_v2_v3_v31.csv') if x['video_id']==r['video_id'])['intratrack_splits']) for r in comparisons)
    # Ningún componente de esta revisión ha leído etiquetas de validez BP ni métricas PoseCNN.
    status='REVISION_PROMISING_READY_FOR_INDEPENDENT_VALIDATION'
    result=dict(status=status,development_only=True,confirmed_holdout_status=pre['confirmatory_status'],
                specification_sha256=pre['spec_sha256'],original_files_checked=len(original_after),source_files_checked=len(sources_after),
                mp4_hashes_checked=len(videos_after),changed_originals=original_changed,changed_sources=source_changed,
                changed_videos=videos_changed,protected_source_relocations=moved,protected_source_path_integrity='CHANGED_EXTERNALLY',
                protected_source_byte_integrity='UNCHANGED',validation76=val_sums,all_development=all_sums,
                restored_from_original=retrieved,removed_original_confident=removed,
                restored_different_from_original_v3=len(new_with_v3_different),
                correct_handoff_pose_compatible=matrix[('VISUALLY_CONFIRMED','POSE_COMPATIBLE')],
                correct_handoff_pose_incompatible=matrix[('VISUALLY_CONFIRMED','POSE_INCOMPATIBLE')],
                correct_handoff_pose_insufficient=matrix[('VISUALLY_CONFIRMED','POSE_INSUFFICIENT_EVIDENCE')],
                false_merges_pose_incompatible=matrix[('VISUAL_PROBLEM','POSE_INCOMPATIBLE')],
                correct_both_confident_original=38,correct_both_confident_revision=42,
                correct_still_abstained_boundary=35,false_merge_both_confident_revision=0,
                newly_preserved_correct_edges=[dict(video_id=k[0],source_track=k[1],target_track=k[2],transition_frame=k[3]) for k in handoff_gained],
                visual_review_scope=dict(frames_inspected=len(visual),explicit_statuses=Counter(r['review_status'] for r in visual)),
                new_velocity_nan_with_confident_position=missing_vel,
                frame_identity_truth='NOT_AVAILABLE_FOR_ALL_FRAMES; Pose proxy and sampled visual review are distinct',
                specific_warning='bp_518 and bp_520 are unique Prototype14 cases, NOT members of Validation76; cannot attribute their 153-frame recovery to Validation76',
                limitation='Physical event/session independence of reserved candidates still unverified; no confirmatory assessment performed')
    (HERE/'bp_front_v31_revision_verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,default=lambda x:dict(x)),encoding='utf-8')
    (HERE/'revision_post_snapshot.json').write_text(json.dumps(dict(original_prototype_files=original_after,original_source_hashes=sources_after,
              protected_source_relocations=moved,inventory_video_hashes=videos_after,spec_sha256=pre['spec_sha256']),indent=2),encoding='utf-8')
    focus=['bp_518','bp_520','bp_663','bp_662','bp_182','bp_238','bp_538','bp_540','bp_544']
    table='\n'.join('| %s | %s | %s | %s | %s | %s |'%(v,named[v]['v3_selected'],named[v]['v31_original_confident'],
           named[v]['revision_confident'],named[v]['revision_unverified'],named[v]['intratrack_splits']) for v in focus)
    gains='\n'.join('| %s | %s | %s | %s |'%(
           r['video_id'],r['v31_original_confident'],r['revision_confident'],r['recovered_vs_old'])
           for r in comparisons if int(r['recovered_vs_old'])>0)
    diag=json.loads((HERE/'diagnosis_summary.json').read_text(encoding='utf-8'))
    report='''# BP_FRONT — Última revisión controlada Pose-Guided Tracking V3.1

**Decisión:** `{state}`. Prototipo **offline/sombra**, no tracker aprobado, sin datos para CNN ni prueba confirmatoria. Especificación congelada `{spec}` inalterada. Configuración de revisión única `experimental_revision_config.json`; diff textual `revision_code_diff.patch`. No se cambiaron umbrales V3 ni V3.1 originales. Los nuevos márgenes de dominancia (0.10× separación de muñecas por muñeca en distancia **y** extensión; máscara ganadora ≤0.15× escala; 3 frames consecutivos) son **experimentales** y provienen de diagnóstico de desarrollo, no de calibración independiente.

## 1. Causa exacta diagnosticada ANTES de modificar
`v31_identity.py` clasifica dos máscaras como `SUPPORTED_HELD_BAR` basándose en `q` firmado y eje; en el segundo pase (líneas 67–71 del original), si hay más de una, **revierte ambas a `INDETERMINATE` sin comparar distancias ni extensión real**. En `bp_518` los **67** frames omitidos respecto a V3 se deben precisamente a `multiple supported BAR candidates without calibrated main-bar margin` (65 con dos BAR, 2 con tres); en `bp_520`, los **88** omitidos tienen el mismo motivo (88 con dos BAR). No eran pérdidas causadas por handoffs (0) ni división interna (0). En [518 f0](diagnostic_previews/bp_518/frame_00000.jpg), la candidata de V3 mide ~52/69 px a muñecas frente a ~1051/299 px de la rival; en [520 f0](diagnostic_previews/bp_520/frame_00000.jpg), ~53/71 frente a ~218/975 px. La rival mantiene `q`/eje aparentes porque es **fragmento lateral de rack/barra**, pero su extensión hasta ambas manos es ~200–1000 px. Esto no equivale a afirmar que todo frame omitido esté visualmente anotado como correcto.

`diagnosis_omitted_candidates.csv` detalla **todas** las candidatas de cada frame omitido (confidence BAR/ambas muñecas, máscara, eje, extensión, q, anchura, soporte temporal, razón exacta). `diagnosis_39_correct_boundary_abstentions.csv` documenta individualmente los 39 bordes correctos previos sin continuidad: 38 edges Pose compatibles y uno insuficiente; predominan entrada/salida de competidores (20 con 1→2 BAR, 10 con 2→1), pero también hay 4 con BAR única a cada lado. Los datos no prueban por sí solos identidad física.

## 2. Corrección localizada y decisión anatómica
`v31_revision_identity.py` recupera **únicamente el estado previo al segundo pase** de candidatas anuladas por competencia; permite una ganadora si ya tenía soporte bilateral por `q`, eje y movimiento temporal **y** domina a CADA competidora en distancia a máscara **y** extensión a ambas muñecas por un margen de escala, persiste ≥3 frames consecutivos en el mismo V2/extractor, con ambas muñecas fiables. Ni V2 main, ni V3, ni confidence mayor son entrada del ranking. Si una competidora está cerca de una mano, la máscara es parcial, el intervalo es corto, falta Pose o los márgenes son similares: ambas continúan indeterminadas. No se condena al perdedor como objeto falso solo por perder la comparación.

`v31_revision_handoff.py` permite corroborar **un extremo** de edge V3 ya seleccionado y `POSE_COMPATIBLE` solo si su propia anatomía individual (`q`/eje) es plausible, hay ≥3 observaciones consecutivas compatibles del **mismo subtracklet**, no hay competidor ambiguo en el frame, la relación q concuerda y no hay split híbrido. No convierte automáticamente `POSE_COMPATIBLE` en frame seguro. Rechazos V3 siguen rechazados; V3.1 original (y sus scores recalculados tras split) quedan intactos. `02_run_revision.py` hace replay de métricas BAR/pose del CSV original, cuyas fuentes RAW/MP4 se verificaron por hash. No reejecuta YOLO, no usa reviews visuales como input, y conserva índices BAR originales. La velocidad solo es válida en frames consecutivos del mismo subtracklet o a través de un handoff V3 seleccionado y Pose-compatible; gaps y cambios sin enlace devuelven `NaN` + máscara de disponibilidad=0.

## 3. Evidencia visual acotada
Se generaron 53 previews diagnósticas antes del cambio en inicio, medio, final, rachas de abstención, competencia y aparición/desaparición; se revisaron explícitamente frames individuales de 518/520 y controles/cambios muestreados en `bp_front_v31_revision_visual_review.csv`. [518 f0/f30/f69](diagnostic_previews/bp_518/), [520 f0/f78/f80](diagnostic_previews/bp_520/) y [comparativas revisadas](previews/) muestran barra seleccionada sobre ambas manos y fragmento lateral corto. Se inspeccionaron también `bp_517` f0/f9, `bp_585` f71, `bp_311` f150, `bp_102` f17 y controles. **Solo los frames anotados** tienen revisión visual explícita; ninguno de los otros frames recuperados se convierte automáticamente en ground truth. Máscaras parciales de `bp_544`/`bp_238` siguen ambiguas cuando las geometrías compiten.

## 4. Resultados por video (V3 / V3.1 original / revisión, frames)
| Video | V3 | Original confiables | Revisión confiables | Revisión indeterminados | Splits V2 |
|---|---:|---:|---:|---:|---:|
{focus}

`bp_518`: **94→160/161** (+66), queda uno no confiable. `bp_520`: **69→156/157** (+87), queda uno no confiable. En conjunto +153/155 omitidos; **son casos Prototype14 únicos, NO Validation76**. La cifra de 157 frames V3 atribuida a `bp_518` en el encargo no coincide con sus artefactos: V3 allí selecciona **161**; 157 corresponde a `bp_520`. Su incremento no puede atribuirse a Validation76. Ninguna selección recuperada eligió instancia distinta de V3; esto facilita comparación, **no demuestra que V3 tuviera siempre identidad correcta**.

## 5. Development76, 77 correctos y 2 falsos merges
| Medida | V3 | V3.1 original | Revisión |
|---|---:|---:|---:|
| Frames seleccionados / confiables (76 videos) | {v3} | {o} | {n} |
| Déficit de frames confiables respecto V3 | — | {gapold} | {gapnew} |
| Handoffs visualmente correctos con **ambos extremos confiables** | — | {before}/77 | {after}/77 |
| Handoffs correctos Pose incompatibles | — | 0/77 | 0/77 |
| Falsos merges visualmente confirmados `bp_663` Pose incompatibles | 0/2 detectados por V3 | 2/2 | 2/2 |

El verificador Pose conserva la clasificación 76 compatibles + 1 indeterminado de los 77 correctos; la **continuidad efectivamente publicada** mejora solo **{gain}/77 bordes**, de 38 a {after}, y **{remain}/77 siguen con abstención**. `bp_238` conserva gaps e indeterminación, `bp_182` los 3 edges compatibles; `bp_538/540/544` no sufren rechazos por recorte/longitud. `bp_662` conserva splits @21/@62 y propuestas V3 rechazadas; `bp_663` conserva splits @17/@49, veta 0→4 @11 y 4→5 @51 y no publica la única BAR inferior falsa. No se recuperó ningún frame previamente marcado `OBSERVED_INCOMPATIBLE` en esos videos.

## 6. Resultados agregados y regresiones
| Video con recuperación | Original | Revisado | Recuperados |
|---|---:|---:|---:|
{gains}

La unión 85 development pasa de {allold} a {allnew} frames confiables (+{totalgain}) sobre {allv3} seleccionados V3; Validation76 (76 videos, **sin** 518/520) pasa de {o} a {n} (+{valg}, déficit aún {gapnew} contra V3). No se retiró ningún frame que V3.1 original publicaba; no se publicó una instancia que ese mismo frame careciera de BAR RAW. Los 17 edges V3 no seleccionados siguen sin referencia visual negativa por candidato. No hay etiquetas visuales exhaustivas por frame, por lo que **cobertura recuperada no es precisión de identidad**. Persisten abstenciones de bordes correctos y la limitación previa: ruta por frames/subtracklets, no optimización global completamente independiente de ancla V2.

## 7. Pruebas, integridad y límites
Las 11 pruebas sintéticas originales + 9 pruebas de revisión pasan: ambigüedad real, fragmento lateral distante, rival próxima a una sola mano, parcial, gap y persistencia, contexto correcto/falso/insuficiente y veto ante track híbrido. `bp_front_v31_revision_verification.json` comprueba índices BAR por frame, `NaN`/máscaras de disponibilidad, velocidades, handoffs revisados y datos visuales sólo tras ejecutar el algoritmo. SHA-256 PRE/POST: {orig} ficheros del prototipo original, {source} fuentes protegidas (RAW, modelos, V2/V3, checkpoints/splits, labels y especificación) y {videos} MP4: **cero cambios de bytes**. **Excepción de ruta detectada durante el trabajo:** `extract_keypoints/bp/extract_keypoints_front_bp.py` fue renombrado externamente a `extract_keypoints/bp/individual/extract_keypoints_front_bp.py`; SHA-256 idéntico, ruta NO intacta y manifest de especificación aún apunta a la ruta antigua. No se revirtió trabajo ajeno; verificar la nueva ruta documental antes de reproducir en otro entorno. Versiones y hashes de código/config/resultados en `bp_front_v31_revision_manifest.json`.

**Límites:** margen nuevo ajustado conceptualmente tras inspeccionar 518/520 en desarrollo, sin referencia visual global. No tratarlo como sensibilidad/especificidad confirmatoria. Persisten {remain}/77 handoffs correctos con pérdida de continuidad y algún frame `VISUALLY_INDETERMINATE` podría aparecer en una anotación exhaustiva futura. La procedencia Pose histórica carece de timestamps/hash MP4 original; solo secuencia y muestras visuales respaldan sincronía. El pool de 559 reservados no acredita independencia de evento físico; **no se abrió ni se procesó**.

## 8. Decisión de parada
`{state}` **solo como candidata a preparación de validación independiente, NO tracking final**: recupera contrastes anatómicos artificialmente ambiguos y mejora modestamente bordes sin perder los dos vetos confirmados ni aceptar un nuevo falso merge observado en las muestras revisadas. Congelar hash del código/config presente; antes de ejecutar confirmación, resolver identidad física/sesión y generar anotaciones visuales ciegas por frame/segmento/edge. Si no puede acreditarse la independencia o aparecen identidades erróneas, **detener optimización del tracking** y continuar tesis con señales RAW + disponibilidad explícita. No continuar automáticamente con V3.2/V4 ni entrenar CNN con esta salida.
'''.format(state=status,spec=pre['spec_sha256'],focus=table,v3=val_sums['v3_selected'],o=val_sums['v31_original_confident'],n=val_sums['revision_confident'],
           gapold=val_sums['v3_selected']-val_sums['v31_original_confident'],gapnew=val_sums['v3_selected']-val_sums['revision_confident'],
           before=38,after=42,gain=4,remain=35,
           gains=gains,allold=sum(int(r['v31_original_confident']) for r in comparisons),allnew=sum(int(r['revision_confident']) for r in comparisons),
           totalgain=retrieved,allv3=sum(int(r['v3_selected']) for r in comparisons),valg=val_sums['revision_confident']-val_sums['v31_original_confident'],
           orig=len(original_after),source=len(sources_after),videos=len(videos_after))
    (HERE/'Reporte_BP_FRONT_Pose_Guided_Tracking_V31_Final_Revision.md').write_text(report,encoding='utf-8')
    run['final_status']=status
    run['verification_sha256']=sha(HERE/'bp_front_v31_revision_verification.json')
    run['report_sha256']=sha(HERE/'Reporte_BP_FRONT_Pose_Guided_Tracking_V31_Final_Revision.md')
    code=list(HERE.glob('*.py'))+[HERE/'experimental_revision_config.json']
    run['final_code_sha256']={p.name:sha(p) for p in sorted(code)}
    extras=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='bp_front_v31_revision_manifest.json']
    run['final_artifact_sha256']={str(p.relative_to(HERE)).replace('\\','/'):sha(p) for p in extras}
    (HERE/'bp_front_v31_revision_manifest.json').write_text(json.dumps(run,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PASS',status,'restored',retrieved,'val',val_sums['v31_original_confident'],'->',val_sums['revision_confident'],'boundaries 38->42','protected',len(original_after),len(sources_after),len(videos_after))


if __name__=='__main__':main()
