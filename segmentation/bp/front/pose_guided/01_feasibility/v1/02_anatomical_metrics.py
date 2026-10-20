"""Métricas exploratorias offline de muñecas y máscaras originales; no ejecuta YOLO."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
SEG = ROOT / 'segmentation/bp/front'
V3 = SEG / 'tracking_v3/validation_76'
RAW = SEG / 'tracking_v2/raw'
WINDOW = 6  # observaciones reales por lado, máximo; sin interpolación
CONF = 0.5  # umbral ya aplicado por el extractor Pose RAW
PREVIEW_IDS = {'bp_663', 'bp_662', 'bp_182', 'bp_238', 'bp_538', 'bp_540', 'bp_544', 'bp_102', 'bp_103'}


def write_csv(name, rows, fields):
    with (HERE / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def mask_for(raw, index):
    a, b = map(int, raw['mask_byte_offsets'][index:index + 2])
    x, y = map(int, raw['instance_mask_origin_xy'][index])
    h, w = map(int, raw['instance_mask_shape_hw'][index])
    fh, fw = map(int, raw['frame_image_shape_hw'][int(raw['instance_frame_index'][index])])
    mask = np.zeros((fh, fw), np.uint8)
    if h and w:
        mask[y:y+h, x:x+w] = np.unpackbits(raw['mask_bits'][a:b], bitorder='little', count=h*w).reshape(h, w)
    return mask


def observation(raw, pose, index, frame):
    h, w = map(int, raw['frame_image_shape_hw'][frame])
    mask = mask_for(raw, index)
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    xy = np.column_stack((xx, yy)).astype(np.float64)
    center = xy.mean(axis=0)
    cov = np.cov(xy, rowvar=False)
    _, vectors = np.linalg.eigh(cov)
    axis = vectors[:, -1]
    projection = (xy - center) @ axis
    low, high = np.min(projection), np.max(projection)
    # Distancia euclídea desde el píxel redondeado de muñeca al píxel más cercano de máscara.
    distance_map = cv2.distanceTransform(1 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    wrists = []
    for kp in (5, 4):  # left_wrist, right_wrist, índice del modelo v8
        x, y, conf = map(float, pose[frame, kp])
        valid = np.isfinite(x) and np.isfinite(y) and conf >= CONF and 0 <= x <= 1 and 0 <= y <= 1
        point = np.array([x*w, y*h]) if valid else None
        if point is None:
            wrists.append({'conf': conf, 'valid': False, 'distance': np.nan, 'axis': np.nan,
                           'extension': np.nan, 'signed_y': np.nan, 'point': None})
            continue
        proj = (point-center) @ axis
        extension = max(low-proj, 0., proj-high)
        perpendicular = abs(np.linalg.det(np.column_stack((axis, point-center))))
        pix = np.clip(np.rint(point).astype(int), [0, 0], [w-1, h-1])
        wrists.append({'conf': conf, 'valid': True, 'distance': float(distance_map[pix[1], pix[0]]),
                       'axis': float(perpendicular), 'extension': float(extension),
                       'signed_y': float(center[1]-point[1]), 'point': point})
    span = float(np.linalg.norm(wrists[0]['point'] - wrists[1]['point'])) if all(z['valid'] for z in wrists) else np.nan
    return {'frame': frame, 'instance': index, 'track': int(raw['instance_track_id'][index]),
            'centroid': center, 'mask': mask, 'wrists': wrists, 'span': span,
            'area': int(len(xx)), 'width': float(high-low)}


def finite_median(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if len(vals) else np.nan


def summarize(observations):
    good = [o for o in observations if o is not None]
    result = {'n_frames': len(good), 'both_frames': sum(all(z['valid'] for z in o['wrists']) for o in good),
              'one_frames': sum(sum(z['valid'] for z in o['wrists']) == 1 for o in good),
              'none_frames': sum(not any(z['valid'] for z in o['wrists']) for o in good)}
    for i, side in enumerate(('left', 'right')):
        result[side+'_conf'] = finite_median([o['wrists'][i]['conf'] for o in good])
        for name in ('distance', 'axis', 'extension', 'signed_y'):
            result[side+'_'+name] = finite_median([o['wrists'][i][name] for o in good])
    result['wrist_span_px'] = finite_median([o['span'] for o in good])
    result['bilateral_px'] = finite_median([max(z['distance'] for z in o['wrists']) for o in good if all(z['valid'] for z in o['wrists'])])
    result['bilateral_normalized'] = result['bilateral_px'] / result['wrist_span_px'] if np.isfinite(result['wrist_span_px']) and result['wrist_span_px'] > 1 else np.nan
    result['mask_width_px'] = finite_median([o['width'] for o in good])
    # Correlación solo si hay desplazamiento medible; una pausa NO prueba incompatibilidad.
    motion = []
    for a, b in zip(good[:-1], good[1:]):
        if b['frame'] != a['frame']+1:
            continue
        if all(z['valid'] for o in (a, b) for z in o['wrists']):
            motion.append((b['centroid']-a['centroid'],
                           (b['wrists'][0]['point']+b['wrists'][1]['point']-a['wrists'][0]['point']-a['wrists'][1]['point'])/2))
    result['motion_pairs'] = len(motion)
    result['motion_mismatch_px'] = finite_median([np.linalg.norm(bar-wrist) for bar, wrist in motion])
    result['bar_step_px'] = finite_median([np.linalg.norm(bar) for bar, wrist in motion])
    result['wrist_step_px'] = finite_median([np.linalg.norm(wrist) for bar, wrist in motion])
    result['motion_status'] = ('MEASURABLE' if len(motion) >= 3 and
                               max(result['bar_step_px'], result['wrist_step_px']) >= 3 else 'PAUSE_OR_INSUFFICIENT')
    return result


def preview(video, frame, raw, pose, selected, destination):
    cap = cv2.VideoCapture(str(ROOT / 'dataset/bp/front' / (video+'.mp4')))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    cap.release()
    if not ok or image.shape[:2] != tuple(raw['frame_image_shape_hw'][frame]):
        raise ValueError('frame de video no coincide: '+video+' '+str(frame))
    start, end = map(int, raw['frame_instance_offsets'][frame:frame+2])
    bar_indices = [j for j in range(start, end) if str(raw['instance_class_name'][j]) == 'bar']
    overlay = image.copy()
    colors = [(0, 200, 255), (255, 100, 0), (0, 255, 70), (230, 0, 220)]
    for n, index in enumerate(bar_indices):
        mask = mask_for(raw, index).astype(bool)
        color = colors[n % len(colors)]
        overlay[mask] = color
        x1, y1, _, _ = raw['instance_bbox_xyxy_px'][index].astype(int)
        label = 'track=%d idx=%d%s' % (int(raw['instance_track_id'][index]), index,
                                      ' V3' if index == selected else '')
        cv2.putText(image, label, (max(0, x1), max(25, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
    image = cv2.addWeighted(image, .65, overlay, .35, 0)
    h, w = image.shape[:2]
    for kp, name, color in [(5, 'L', (255, 0, 255)), (4, 'R', (0, 0, 255))]:
        x, y, c = pose[frame, kp]
        if np.isfinite(x) and np.isfinite(y):
            point = (int(x*w), int(y*h))
            cv2.circle(image, point, 10, color, -1)
            cv2.putText(image, '%s %.3f' % (name, c), (point[0]+12, point[1]-12), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
    cv2.putText(image, '%s frame=%d fps=%.0f' % (video, frame, 30), (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise IOError(destination)


def main():
    with (HERE/'bp_front_pose_guided_sync_audit.csv').open(encoding='utf-8') as f:
        audit = {row['video_id']: row for row in csv.DictReader(f)}
    with (V3/'manifests/bp_front_tracking_v3_validation_76_handoffs.csv').open(encoding='utf-8-sig') as f:
        handoffs = list(csv.DictReader(f))
    if any(audit[r['video_id']]['sync_status'] != 'SEQUENCE_CONSISTENT_OVERLAY_PENDING' for r in handoffs):
        raise RuntimeError('Sincronización automática no confirmada; abortar métricas')
    all_rows, identities = [], []
    by_video = {}
    for row in handoffs:
        by_video.setdefault(row['video_id'], []).append(row)
    for video in sorted(set(audit)):
        pose = np.load(ROOT/'keypoints/bp/front'/(video+'.npy'), allow_pickle=False)
        with np.load(RAW/(video+'_segmentation.npz'), allow_pickle=False) as raw, np.load(
                V3/'selections'/(video+'_bar_tracking_v3_selection.npz'), allow_pickle=False) as selection:
            # Índices de instancia y track: siempre V2 RAW, los mismos referenciados por V3.
            by_track = {}
            for index in range(len(raw['instance_frame_index'])):
                if raw['instance_class_name'][index] == 'bar':
                    by_track.setdefault(int(raw['instance_track_id'][index]), []).append((int(raw['instance_frame_index'][index]), index))
            cache = {}
            def get(index, frame):
                if index not in cache:
                    cache[index] = observation(raw, pose, index, frame)
                return cache[index]
            for track, pairs in sorted(by_track.items()):
                # Muestreo temporal uniforme (máximo 16), más ventanas de handoff aparte.
                sampled = (pairs if video in {'bp_662', 'bp_663'} else
                           [pairs[i] for i in sorted(set(np.linspace(0, len(pairs)-1, min(16, len(pairs)), dtype=int)))])
                observed = [get(idx, frame) for frame, idx in sampled]
                stats = summarize(observed)
                signed = [np.mean([z['signed_y'] for z in o['wrists']]) for o in observed if o and all(z['valid'] for z in o['wrists'])]
                identities.append({'video_id': video, 'track_id': track, 'is_v2_main': track == json.loads(str(raw['metadata_json']))['selected_tracks']['bar'],
                                   'first_frame': min(frame for frame, idx in pairs), 'last_frame': max(frame for frame, idx in pairs),
                                   'track_total_frames': len(pairs), 'sampling': 'all_actual_frames' if video in {'bp_662', 'bp_663'} else 'uniform_max_16_actual_frames',
                                   'bar_above_wrist_frames': sum(value < 0 for value in signed),
                                   'bar_below_wrist_frames': sum(value > 0 for value in signed), **stats})
            for r in by_video.get(video, []):
                source, target = int(r['source_track']), int(r['target_track'])
                t = int(r['transition_frame'])
                before = [(frame, idx) for frame, idx in by_track.get(source, []) if t-WINDOW <= frame < t]
                after = [(frame, idx) for frame, idx in by_track.get(target, []) if t <= frame < t+WINDOW]
                s, d = summarize([get(idx, frame) for frame, idx in before]), summarize([get(idx, frame) for frame, idx in after])
                # Un tracklet V2 puede cambiar de objeto dentro del propio track; el borde se mide
                # también con los dos frames observados más próximos (sin inventar continuidad).
                near_before = before[-2:]
                near_after = after[:2]
                ns = summarize([get(idx, frame) for frame, idx in near_before])
                nd = summarize([get(idx, frame) for frame, idx in near_after])
                # Comparación con otra barra en los mismos frames si está presente.
                alternative = []
                for frame, index in after:
                    others = [(other, oi) for other, pairs in by_track.items() if other != target for f, oi in pairs if f == frame]
                    for other, oi in others:
                        candidate = get(oi, frame)
                        if candidate is not None and all(z['valid'] for z in candidate['wrists']):
                            alternative.append((frame, other, max(z['distance'] for z in candidate['wrists'])))
                alt_med = finite_median([x[2] for x in alternative])
                delta = d['bilateral_px'] - s['bilateral_px']
                vertical = [nd[side+'_signed_y']-ns[side+'_signed_y'] for side in ('left', 'right')]
                # Criterio exploratorio uniforme para todo el conjunto; NO es un umbral V3.
                quality = 'BOTH_RELIABLE' if min(s['both_frames'], d['both_frames']) >= 3 else (
                    'ONE_OR_SPARSE' if s['one_frames']+d['one_frames']+s['both_frames']+d['both_frames'] else 'NONE')
                assessment = 'POSE_INSUFFICIENT_EVIDENCE'
                reason = 'ventana con menos de 3 observaciones bilaterales por lado o sin escala'
                if quality == 'BOTH_RELIABLE' and ns['both_frames'] == 2 and nd['both_frames'] == 2 and np.all(np.isfinite(vertical)) and np.isfinite(s['wrist_span_px']) and np.isfinite(d['wrist_span_px']):
                    ref = min(s['wrist_span_px'], d['wrist_span_px'])
                    if (min(map(abs, vertical)) > max(45., .08*ref) and np.sign(vertical[0]) == np.sign(vertical[1])
                            and min(s['mask_width_px'], d['mask_width_px']) / max(s['mask_width_px'], d['mask_width_px']) > .65):
                        assessment = 'POSE_SUPPORTS_DIFFERENT_OBJECT'
                        reason = 'salto vertical firmado coherente de ambas muñecas (>45 px y >8% escala); anchura de máscara comparable'
                    elif (max(map(abs, vertical)) < max(30., .045*ref)
                            and max(s['left_axis'], s['right_axis'], d['left_axis'], d['right_axis']) < .15*ref):
                        assessment = 'POSE_SUPPORTS_SAME_BAR'
                        reason = 'posición vertical relativa bilateral estable y eje cercano; compatibilidad, no prueba de identidad'
                    else:
                        reason = 'diferencia vertical, máscara parcial o distancia ambigua; no forzar identidad'
                # La alternativa y el movimiento son descriptivos; NO se emplean como etiquetas de referencia.
                out = dict(r)
                out.update(original_visual_review_status=r['visual_review_status'],
                           original_false_merge_detected=r['false_merge_detected'],
                           left_wrist_confidence=d['left_conf'], right_wrist_confidence=d['right_conf'],
                           left_wrist_bar_distance=d['left_distance'], right_wrist_bar_distance=d['right_distance'],
                           left_wrist_axis_distance=d['left_axis'], right_wrist_axis_distance=d['right_axis'],
                           left_wrist_extension=d['left_extension'], right_wrist_extension=d['right_extension'],
                           bilateral_compatibility=d['bilateral_normalized'], temporal_motion_compatibility=d['motion_status'],
                           pose_evidence_quality=quality, pose_guided_assessment=assessment, assessment_reason=reason,
                           source_bilateral_px=s['bilateral_px'], target_bilateral_px=d['bilateral_px'],
                           distance_jump_px=delta, source_left_wrist_bar_distance=s['left_distance'],
                           source_right_wrist_bar_distance=s['right_distance'], source_left_axis_distance=s['left_axis'],
                           source_right_axis_distance=s['right_axis'], source_wrist_span_px=s['wrist_span_px'],
                           source_left_signed_y=s['left_signed_y'], source_right_signed_y=s['right_signed_y'],
                           target_left_signed_y=d['left_signed_y'], target_right_signed_y=d['right_signed_y'],
                           left_signed_y_jump=vertical[0], right_signed_y_jump=vertical[1],
                           near_source_left_signed_y=ns['left_signed_y'], near_source_right_signed_y=ns['right_signed_y'],
                           near_target_left_signed_y=nd['left_signed_y'], near_target_right_signed_y=nd['right_signed_y'],
                           near_source_bilateral_px=ns['bilateral_px'], near_target_bilateral_px=nd['bilateral_px'],
                           near_source_frames=ns['n_frames'], near_target_frames=nd['n_frames'],
                           target_wrist_span_px=d['wrist_span_px'], source_frames=s['n_frames'], target_frames=d['n_frames'],
                           source_both_frames=s['both_frames'], target_both_frames=d['both_frames'],
                           source_one_frames=s['one_frames'], target_one_frames=d['one_frames'],
                           source_motion_status=s['motion_status'], source_motion_mismatch_px=s['motion_mismatch_px'],
                           target_motion_mismatch_px=d['motion_mismatch_px'], source_mask_width_px=s['mask_width_px'],
                           target_mask_width_px=d['mask_width_px'], alternative_bar_bilateral_px=alt_med,
                           alternative_observations=len(alternative), window_frames=WINDOW)
                all_rows.append(out)
                if video in PREVIEW_IDS:
                    frames = sorted({f for f, _ in before[-2:] + after[:2]})
                    for frame in frames:
                        preview(video, frame, raw, pose, int(selection['selected_instance_indices'][frame]),
                                HERE/'previews'/video/('%s_%s_frame_%05d.jpg' % (source, target, frame)))
            if video in {'bp_102', 'bp_103', 'bp_662', 'bp_663'}:
                key_frames = ([0, min(10, len(pose)-1), len(pose)//2, len(pose)-1]
                              if video in {'bp_102', 'bp_103'} else [0, len(pose)//2])
                for frame in key_frames:
                    preview(video, frame, raw, pose, int(selection['selected_instance_indices'][frame]),
                            HERE/'previews'/video/('identity_frame_%05d.jpg' % frame))
        print(video, 'handoffs', len(by_video.get(video, [])))
    assert len(all_rows) == len(handoffs)
    write_csv('bp_front_pose_guided_handoff_metrics.csv', all_rows, list(all_rows[0]))
    write_csv('bp_front_pose_guided_track_identity_metrics.csv', identities, list(identities[0]))


if __name__ == '__main__':
    main()
