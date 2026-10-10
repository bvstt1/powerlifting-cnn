"""Verificación offline read-only de fuentes/resultados; escribe auditoría solo aquí."""
import csv
import hashlib
import importlib.metadata
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from v31_common import HERE, ROOT, SEG, SPEC_SHA, sha, json_safe


def read(name):
    with (HERE/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def count(rows,key,value):
    return sum(r[key]==value for r in rows)


def main():
    pre=json.loads((HERE/'bp_front_v31_source_snapshot_before.json').read_text(encoding='utf-8'))
    run=json.loads((HERE/'bp_front_v31_prototype_manifest.json').read_text(encoding='utf-8'))
    spec=sha(ROOT/'segmentation/bp/front/pose_guided_tracking_v31_spec/BP_FRONT_Pose_Guided_Tracking_V31_Specification.md')
    assert spec==SPEC_SHA==pre['spec_sha256']==run['spec_sha256']
    source_after={name:sha(ROOT/name) for name in pre['source_hashes']}
    changed=[name for name,expected in pre['source_hashes'].items() if source_after[name]!=expected]
    video_after={vid:sha(ROOT/'dataset/bp/front'/(vid+'.mp4')) for vid in pre['video_inventory_hashes']}
    video_changed=[vid for vid,expected in pre['video_inventory_hashes'].items() if video_after[vid]!=expected]
    assert not changed and not video_changed,('INPUT CHANGED',changed,video_changed)
    assert all(sha(HERE/name)==value for name,value in run['output_sha256'].items())
    frames=read('bp_front_v31_frame_identity.csv');segments=read('bp_front_v31_track_consistency.csv');tracks=read('bp_front_v31_track_summary.csv');edges=read('bp_front_v31_handoff_decisions.csv')
    trajectories=read('bp_front_v31_selected_trajectories.csv');comps=read('bp_front_v31_comparison_v2_v3_v31.csv');ablations=read('bp_front_v31_ablation_results.csv');sync=read('bp_front_v31_sync_audit.csv')
    assert len(comps)==85 and len(run['errors'])==0 and len(sync)==85 and all(r['status']=='SEQUENCE_AND_VISUAL_SUPPORTED' and r['pose_traceability']=='NOT_CRYPTOGRAPHICALLY_TRACEABLE' for r in sync)
    assert set(run['development_videos']) == set(pre['prototype14_ids'])|set(pre['validation76_ids'])
    assert len(trajectories)==sum(int(r['frames_total']) for r in comps)
    assert len(ablations)==5*len(comps)
    idx={(r['video_id'],int(r['frame_index'])):r for r in trajectories}
    for r in trajectories:
        valid=r['selected_global_instance_index']!='-1'
        assert valid==(r['status']=='OBSERVED_CONFIDENT')
        assert (r['position_observed']=='1')==valid
        assert (r['bar_identity_valid']=='1')==valid
        if not valid:
            assert r['bar_x_px']=='NaN' and r['bar_y_px']=='NaN' and r['velocity_valid']=='0' and r['vx_px_s']=='NaN' and r['vy_px_s']=='NaN'
        if r['velocity_valid']=='1':
            prev=idx.get((r['video_id'],int(r['frame_index'])-1))
            assert valid and prev and prev['selected_global_instance_index']!='-1'
        if r['status']=='NO_BAR_DETECTED':assert r['bar_present_raw']=='0'
        if r['status']=='OBSERVED_INCOMPATIBLE':assert r['bar_present_raw']=='1'
    # Trazabilidad de TODOS los índices confiables a su misma instancia BAR de V2 RAW.
    for vid in run['development_videos']:
        with np.load(SEG/'tracking_v2/raw'/(vid+'_segmentation.npz'),allow_pickle=False) as raw:
            for r in (row for row in trajectories if row['video_id']==vid and row['selected_global_instance_index']!='-1'):
                i=int(r['selected_global_instance_index']);t=int(r['frame_index'])
                assert i<len(raw['instance_frame_index']) and int(raw['instance_frame_index'][i])==t and raw['instance_class_name'][i]=='bar'
                assert int(raw['instance_track_id'][i])==int(r['v2_track_id'])
    assert len(list((HERE/'sync_sheets').glob('sheet_*.jpg')))==11
    review=json.loads((HERE/'bp_front_v31_visual_sync_review.json').read_text(encoding='utf-8'))
    assert review['status']=='SEQUENCE_AND_VISUAL_SUPPORTED_FOR_SAMPLED_FRAMES_ONLY'
    with (SEG/'tracking_v3/validation_76/manifests/bp_front_tracking_v3_validation_76_handoffs.csv').open(encoding='utf-8-sig',newline='') as f:
        originals=list(csv.DictReader(f))
    eidx={(r['video_id'],r['source_track'],r['target_track'],r['transition_frame']):r for r in edges if r['selected_by_v3']=='True'}
    truth=Counter();continuity=Counter();unreviewed=Counter()
    for original in originals:
        key=tuple(original[k] for k in ('video_id','source_track','target_track','transition_frame'))
        edge=eidx.get(key)
        if original['selected_by_v3']=='1':
            assert edge is not None
            truth[(original['visual_review_status'],edge['pose_decision'])]+=1
            prev=idx.get((key[0],int(key[3])-1));next=idx.get((key[0],int(key[3])))
            continuity[(original['visual_review_status'],'BOTH_CONFIDENT' if prev and next and prev['selected_global_instance_index']!='-1' and next['selected_global_instance_index']!='-1' else 'UNVERIFIED_BOUNDARY')]+=1
        else:
            assert original['visual_review_status']=='NOT_APPLICABLE'
            proposals=[r for r in edges if r['video_id']==key[0] and r['source_track']==key[1] and r['target_track']==key[2] and r['transition_frame']==key[3] and r.get('virtual_internal_edge')!='True']
            if proposals:unreviewed[proposals[0]['pose_decision']]+=1
    assert truth[('VISUALLY_CONFIRMED','POSE_COMPATIBLE')]==76
    assert truth[('VISUALLY_CONFIRMED','POSE_INSUFFICIENT_EVIDENCE')]==1
    assert truth[('VISUALLY_CONFIRMED','POSE_INCOMPATIBLE')]==0
    assert truth[('VISUAL_PROBLEM','POSE_INCOMPATIBLE')]==2
    assert sum(unreviewed.values())==17
    critical=[r for r in edges if r['video_id']=='bp_663' and r['selected_by_v3']=='True']
    assert len(critical)==2 and all(r['pose_decision']=='POSE_INCOMPATIBLE' for r in critical)
    cut={vid:sorted(int(frame) for r in tracks if r['video_id']==vid for frame in json.loads(r['proposed_cut_frames'])) for vid in ('bp_662','bp_663')}
    assert cut['bp_662']==[21,62] and cut['bp_663']==[17,49]
    for vid in ('bp_662','bp_663'):
        contaminated={i for segment in segments if segment['video_id']==vid and segment['original_v2_track_id']=='4' and segment['segment_ordinal'] in ('0','2') for i in json.loads(segment['global_instance_indices'])}
        assert not any(row['video_id']==vid and row['selected_global_instance_index']!='-1' and int(row['selected_global_instance_index']) in contaminated for row in trajectories)
    assert sum(r['video_id']=='bp_182' and r['selected_by_v3']=='True' and r['pose_decision']=='POSE_COMPATIBLE' for r in edges)==3
    assert sum(r['video_id'] in ('bp_538','bp_540','bp_544') and r['selected_by_v3']=='True' and r['pose_decision']=='POSE_COMPATIBLE' for r in edges)==3
    assert any(r['video_id']=='bp_238' and r['selected_by_v3']=='True' and r['pose_decision']=='POSE_INSUFFICIENT_EVIDENCE' for r in edges)
    val=[r for r in comps if r['video_id'] in pre['validation76_ids']]
    proto_only=[r for r in comps if r['video_id'] not in pre['validation76_ids']]
    sums={k:sum(int(r[k]) for r in val) for k in ('frames_total','bar_raw_detected_frames','v2_selected_frames','v3_selected_frames','v31_confident_frames','v31_unverified_frames','v31_incompatible_frames','no_bar_frames')}
    assert len(val)==76 and len(proto_only)==9 and sums['v3_selected_frames']==sums['v2_selected_frames']+1641
    ablation_sums={variant:sum(int(r['selected_frames']) for r in ablations if r['video_id'] in pre['validation76_ids'] and r['ablation']==variant) for variant in ('V3','IDENTITY_ONLY','TRACK_ONLY','HANDOFF_ONLY','FULL_V31')}
    proxy_sums={variant:sum(int(r['proxy_selected_incompatible_frames']) for r in ablations if r['video_id'] in pre['validation76_ids'] and r['ablation']==variant) for variant in ablation_sums}
    regressions=sorted(comps,key=lambda r:(int(r['v3_selected_frames'])-int(r['v31_confident_frames'])),reverse=True)
    tested={r['video_id']:r for r in comps}
    original_correct_boundary=sum(n for (status,c),n in continuity.items() if status=='VISUALLY_CONFIRMED' and c=='BOTH_CONFIDENT')
    assert sum(n for (status,c),n in continuity.items() if status=='VISUALLY_CONFIRMED')==77
    result=dict(status='PROTOTYPE_NEEDS_REVISION',specification_sha256=SPEC_SHA,verification='PASS_INTEGRITY_AND_CONTRACTS_WITH_DEVELOPMENT_REGRESSIONS',
                source_hashes_checked=len(source_after),video_hashes_checked=len(video_after),inputs_changed=changed,
                development_videos=len(comps),validation76_videos=len(val),prototype14_unique_videos=len(proto_only),
                handoff_review_matrix={str(k):v for k,v in truth.items()},unselected_without_visual_truth=dict(unreviewed),
                correct_boundaries_both_confident=original_correct_boundary,correct_boundaries_abstained=77-original_correct_boundary,
                false_merges_rejected=2,hybrid_cuts=cut,validation76_frames=sums,validation76_ablations=ablation_sums,
                pose_proxy_selected_incompatible_by_ablation=proxy_sums,
                confidence_vs_v3_loss=sums['v3_selected_frames']-sums['v31_confident_frames'],
                major_regressions=[dict(video_id=r['video_id'],v3_frames=int(r['v3_selected_frames']),v31_frames=int(r['v31_confident_frames']),lost=int(r['v3_selected_frames'])-int(r['v31_confident_frames'])) for r in regressions[:8]],
                frame_truth_limitation='no independent full frame-level identity annotations; geometric Pose proxy is not ground truth',
                holdout_status=pre['confirmed_holdout_status'],
                synced_pose_level='SEQUENCE_AND_VISUAL_SUPPORTED_FROM_3_SAMPLED_FRAMES_PER_VIDEO; NOT_CRYPTOGRAPHICALLY_TRACEABLE',
                notes='correct handoff decision is NOT the same as preserving both boundary frames in the confident trajectory')
    (HERE/'bp_front_v31_verification.json').write_text(json.dumps(json_safe(result),indent=2,ensure_ascii=False),encoding='utf-8')
    (HERE/'bp_front_v31_integrity_after.json').write_text(json.dumps(dict(source_hashes=source_after,video_inventory_hashes=video_after,spec_sha256=spec),indent=2),encoding='utf-8')
    code_paths=[p for p in HERE.glob('*.py')]+[HERE/'experimental_config.json',HERE/'bp_front_v31_visual_sync_review.json']
    run['final_status']='PROTOTYPE_NEEDS_REVISION'
    run['runtime']['scipy_installed_version']=importlib.metadata.version('scipy')
    run['final_code_sha256']={p.name:sha(p) for p in sorted(code_paths)}
    run['verification_sha256']=sha(HERE/'bp_front_v31_verification.json')
    run['integrity_after_sha256']=sha(HERE/'bp_front_v31_integrity_after.json')
    run['visual_sync_review_sha256']=sha(HERE/'bp_front_v31_visual_sync_review.json')
    report='''# BP_FRONT: Tracking Pose-Guided V3.1 — prototipo sombra (desarrollo)

**Estado:** `PROTOTYPE_NEEDS_REVISION`. Implementación offline sin YOLO/entrenamiento ni publicación. Especificación `bp_front_pose_guided_v31_spec_1`, SHA-256 `{spec}`; configuración experimental `experimental_config.json` ({config}). Los valores 30/45 px y ratios derivan de Validation76: **no generalizan por este resultado**. Prototype14 y Validation76 son desarrollo/regresión, jamás confirmación.

## Reserva y procedencia
Inventario read-only: 663 MP4 con SHA-256, {excluded} excluidos por uso previo y {reserved} candidatos `RESERVED_UNVERIFIED_DO_NOT_OPEN`; **0 duplicados exactos por SHA-256**, sin poder descartar near-duplicates ni repeticiones físicas. Esos candidatos **no se procesaron** ni se inspeccionó salida V3.1. La independencia física/sesión no está acreditada (`group_id_status=unverified_individual`), por lo que la confirmación sigue `BLOCKED_PHYSICAL_EVENT_INDEPENDENCE_UNVERIFIED`. No se tocó ningún split del clasificador. Posible intersección con futuros tests CNN requiere cruce de IDs/eventos antes de evaluarlos.

Preflight de los {dev} videos de desarrollo: hash de MP4 contra RAW e inventario, T Pose/Seg/V2/V3/video, FPS=30, 1920×1080, índices 0-based, timestamps nominales, bytes de máscara RAW→V2, referencias V3 y coordenadas Pose. `sync_sheets/` contiene 11 láminas de inicio/medio/final para los 85 videos: revisión visual de puntos en manos y máscaras sobre video; cero discrepancias observadas en **esos frames muestreados**. Pose `.npy` carece de timestamp, hash histórico de MP4 y hash de frame: **no** certificación criptográfica ni revisión exhaustiva frame a frame. Cualquier futuro mismatch es `SYNC_BLOCKED`; V3 queda baseline separado.

## Componentes y pruebas
`v31_identity.py`: todas las máscaras BAR RAW, distancia a máscara/eje/extensión, q firmado, escala/confidence y soporte temporal; `v31_consistency.py`: saltos robustos y subtracklets virtuales V2→extractor→índice RAW; `v31_handoff.py`: clasificación Pose y **recomputación geométrica con las funciones V3 originales** sobre subtracklets de límites nuevos, sin alterar V3. `01_run_shadow.py` compara V2/V3 y propone salida confiable. `test_v31_shadow.py`: 11 pruebas sintéticas (incluyen intercambio de muñecas, ausencia, barra inmóvil/en movimiento, parcial, híbrido que vuelve y falsa BAR única), ejecutadas correctamente. La prueba de orientación es de política de bloqueo, no prueba con una cámara invertida real.

## Handoffs con revisión visual original — solo Validation76
| Referencia visual | Compatible | Incompatible | Insuficiente | N |
|---|---:|---:|---:|---:|
| 77 aceptados correctos | {c} | {ci} | {u} | 77 |
| 2 falsos merges | {fc} | {f} | {fu} | 2 |

`bp_663` 0→4 @11 y 4→5 @51: ambos `POSE_INCOMPATIBLE`, sin publicar BAR inferior como confiable en bordes; track 4 se divide lógicamente en **[11–16] / [17–48] / [49–50]**, cortes 17 y 49. No se inventan máscaras de barra superior cuando RAW solo contiene la estructura inferior. En 4→5 la decisión usa **solo dos frames del segmento corto** con corroboración de split previo: es una relajación *experimental* documentada, no evidencia confirmatoria. Frames confiables `bp_663`: {b663} de 64 frente a 64 V3; {incompat} BAR únicas incompatibles y resto sin verificar.

`bp_662` track 4: splits lógicos @21/@62; V2 principal híbrido; las propuestas 0→4 @19 y 4→5 @66 reciben `POSE_INCOMPATIBLE` en la copia (V3 originalmente las rechazó, Pose no las promociona). {b662}/76 frames confiables; el segmento ambiguo no condena todo track 4. Los conteos previos 41/7 y 32/8 son proxies geométricos, no anotación de verdad visual por frame.

Controles `bp_182`: 3 edges compatibles; `bp_238`: correcto indeterminado (gap real), no frames inventados; `bp_538/540/544`: handoffs compatibles con máscaras parcial/completa y grandes cambios de longitud; `bp_102/103`: overlays y estructuras paralelas. Prototype14 único (9 videos fuera de Validation76) muestra riesgo de abstención elevado en `bp_518` y `bp_520` (ver tabla); gaps en `bp_198/301/467` quedan ausentes. Controles son regresión/desarrollo, no validación independiente.

**Diferencia crucial:** el verificador declara {c}/77 handoffs correctos compatibles y ninguno incompatible, **pero solo {bc}/77 tienen ambos frames del borde publicados como confiables**; {bg}/77 se abstienen en alguno de los lados por ambigüedad por frame, máscaras competidoras, recorte o evidencia insuficiente. Por tanto, no informar «{c} handoffs conservados» como si fueran trayectorias continuas. Son pérdidas reales de recuperación que requieren revisión antes de aprobar cualquier uso downstream.

Las 17 propuestas no seleccionadas por V3 mantienen referencia `NOT_APPLICABLE`: Pose emite {nu} insuficientes y {nd} incompatibles entre ellas; **ninguna** se contabiliza como negativo confirmado sin anotación visual. No se usa su revisión como feature.

## Cobertura y abstenciones (76 videos de Validation76)
| Magnitud | Frames |
|---|---:|
| Frames MP4 | {total} |
| BAR RAW en algún índice | {rawbar} |
| V2 selected | {v2} |
| V3 selected | {v3} (V2 + 1641 frames recuperados) |
| V3.1 `OBSERVED_CONFIDENT` | {v31} |
| V3.1 `OBSERVED_UNVERIFIED` | {unv} |
| V3.1 `OBSERVED_INCOMPATIBLE` | {inc} |
| `NO_BAR_DETECTED` | {nobar} |
| Menos frames confiables que selección V3 | {lost} |

**No hay ground truth visual exhaustiva por frame**: «confiable» es el estado del prototipo, no accuracy demostrada. El **proxy anatómico** selecciona {p3} instancias `INCOMPATIBLE_WITH_HELD_BAR` en V3 y {pf} en V3.1; esto NO equivale a contar frames erróneos con referencia visual. Los CSV distinguen índice legacy y nuevo; cuando nuevo=-1, `bar_x_px`, `bar_y_px`, `vx_px_s`, `vy_px_s` son `NaN` con máscaras disponibilidad=0, nunca velocidad cero. Cobertura menor no es automáticamente mala si elimina falsas identidades, pero pérdidas en casos visualmente correctos exigen análisis.

## Ablaciones de desarrollo (frames seleccionados, NO precisión de identidad)
| Variante | Frames | Alcance |
|---|---:|---|
| V3 original | {av3} | baseline sin Pose |
| V3 + identidad por frame | {aid} | sin split ni veto de handoff |
| V3 + consistencia intratrack | {atr} | descarta track completo con split; control de componente, conservador y *no* solución final |
| V3 + verificación de handoff | {aho} | sin split ni selección nueva; corta destino tras edge incompatible |
| V3.1 completo | {afu} | subtracklets + evaluador + salida confiable |

Las ablaciones no son equivalentes en política de abstención; los números **no** identifican por sí solos causalidad ni precisión. No atribuir beneficios de selección inicial al validador de handoffs. La variante TRACK_ONLY pierde cobertura por descartar todo track híbrido y no reproduce división lógica; está etiquetada limitación.

## Regresiones más relevantes y diagnóstico
| Video | V3 frames | V31 confiables | Pérdida |
|---|---:|---:|---:|
{regressions}

En `bp_520` V2 track virtual 7 cubre 157 frames V3, pero 88 quedan indeterminados por máscaras BAR competidoras anatómicamente plausibles sin margen de selección principal calibrado; no es un error de sincronización. `bp_518` también abstiene muchos frames. El prototipo **no preserva suficiente continuidad de asociaciones correctas**: `PROTOTYPE_NEEDS_REVISION`. No se ajustaron umbrales hasta forzar 77/77 bordes confiables. Se requiere mejor discriminación temporal de múltiples máscaras, menos abstención en mascaras parciales y anotación visual real por frame/track para medir FP/FN de identidad. Los dos falsos merges rechazados **no compensan por sí solos** esas regresiones.

La salida actual es una **propuesta por frame con huecos** y subtracklets trazables, no una optimización global de conectividad independiente de la ancla V2. Cuando una máscara apoyada se readquiere tras un gap o cambia de subtracklet, la identidad de ese frame puede tener soporte anatómico, pero el enlace entre segmentos no queda demostrado por ello; velocidades a través del hueco siguen inválidas. Antes de considerar `PROTOTYPE_READY_FOR_CALIBRATION`, implementar selección/reidentificación global sin ancla circular, más controles para conservar edges correctos y una política explícita de procedencia entre segmentos.

## Integridad y reproducibilidad
Antes de ejecución se hashearon {hashes} fuentes (RAW, V2, V3, Prototype14, Validation76, modelos, keypoints, labels, features, PoseCNN/checkpoints/splits BP y especificación) y 663 MP4 de inventario. Después, **0 diferencias** en fuentes y videos; hashes finales de código/config/resultados en `bp_front_v31_prototype_manifest.json`; versiones/OS/seed/tiempo allí. Tests sintéticos: 11/11. Sin YOLO, entrenamiento, Features V3.1 globales ni ejecución de Validation17_RAW_GAPS. Scripts y resultados solo en este directorio. Las revisiones visuales originales se cargaron **solo en este verificador**, después de calcular y persistir los resultados; jamás en los tres validadores.

**Recomendación:** no publicar V3.1 ni generar entradas CNN. Autorizar solo revisión de la selección entre BAR simultáneas y política de preservación de handoffs correctos, con nuevo análisis de desarrollo y sin mirar los candidatos confirmatorios. Después deberán congelarse valores efectivos nuevos, comprobar independencia física y ejecutar evaluación visual independiente con referencias reales por frame.
'''.format(spec=SPEC_SHA,config=run['config_sha256'],dev=len(comps),excluded=sum('EXCLUDED' in r['reservation_status'] for r in read('bp_front_v31_reservation_inventory.csv')),
            reserved=sum(r['reservation_status'].startswith('RESERVED') for r in read('bp_front_v31_reservation_inventory.csv')),
            c=truth[('VISUALLY_CONFIRMED','POSE_COMPATIBLE')],ci=truth[('VISUALLY_CONFIRMED','POSE_INCOMPATIBLE')],u=truth[('VISUALLY_CONFIRMED','POSE_INSUFFICIENT_EVIDENCE')],
            fc=truth[('VISUAL_PROBLEM','POSE_COMPATIBLE')],f=truth[('VISUAL_PROBLEM','POSE_INCOMPATIBLE')],fu=truth[('VISUAL_PROBLEM','POSE_INSUFFICIENT_EVIDENCE')],
            b663=tested['bp_663']['v31_confident_frames'],b662=tested['bp_662']['v31_confident_frames'],incompat=tested['bp_663']['v31_incompatible_frames'],
            bc=original_correct_boundary,bg=77-original_correct_boundary,nu=unreviewed['POSE_INSUFFICIENT_EVIDENCE'],nd=unreviewed['POSE_INCOMPATIBLE'],
            total=sums['frames_total'],rawbar=sums['bar_raw_detected_frames'],v2=sums['v2_selected_frames'],v3=sums['v3_selected_frames'],v31=sums['v31_confident_frames'],
            unv=sums['v31_unverified_frames'],inc=sums['v31_incompatible_frames'],nobar=sums['no_bar_frames'],lost=result['confidence_vs_v3_loss'],
            p3=proxy_sums['V3'],pf=proxy_sums['FULL_V31'],
            av3=ablation_sums['V3'],aid=ablation_sums['IDENTITY_ONLY'],atr=ablation_sums['TRACK_ONLY'],aho=ablation_sums['HANDOFF_ONLY'],afu=ablation_sums['FULL_V31'],
            regressions='\n'.join('| %s | %s | %s | %s |'%(r['video_id'],r['v3_selected_frames'],r['v31_confident_frames'],int(r['v3_selected_frames'])-int(r['v31_confident_frames'])) for r in regressions[:8]),hashes=len(source_after))
    (HERE/'Reporte_BP_FRONT_Pose_Guided_Tracking_V31_Prototype.md').write_text(report,encoding='utf-8')
    extra=[HERE/name for name in ('Reporte_BP_FRONT_Pose_Guided_Tracking_V31_Prototype.md','bp_front_v31_reservation_inventory.csv',
           'bp_front_v31_source_snapshot_before.json','bp_front_v31_visual_sync_review.json','bp_front_v31_verification.json',
           'bp_front_v31_integrity_after.json','sync_sheets/index.json')]
    extra += list((HERE/'sync_sheets').glob('*.jpg'))+list((HERE/'previews').rglob('*.jpg'))
    run['final_artifact_sha256']={str(path.relative_to(HERE)).replace('\\','/'):sha(path) for path in sorted(extra)}
    (HERE/'bp_front_v31_prototype_manifest.json').write_text(json.dumps(json_safe(run),indent=2,ensure_ascii=False),encoding='utf-8')
    print('PASS',result['status'],'sources',len(source_after),'videos',len(video_after),'correct confident boundaries',original_correct_boundary,'/77','V31',sums['v31_confident_frames'],'V3',sums['v3_selected_frames'])


if __name__=='__main__':main()
