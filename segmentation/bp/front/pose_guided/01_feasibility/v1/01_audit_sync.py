"""Auditoría de alineación y huellas; nunca ejecuta modelos ni modifica fuentes."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
SEG = ROOT / 'segmentation/bp/front'
V3 = SEG / 'tracking_v3/validation_76'
HANDOFFS = V3 / 'manifests/bp_front_tracking_v3_validation_76_handoffs.csv'
PRIORITY = {'bp_663', 'bp_662', 'bp_238', 'bp_538', 'bp_540', 'bp_544', 'bp_182', 'bp_102', 'bp_103'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def paths(video):
    return {
        'video': ROOT / 'dataset/bp/front' / (video + '.mp4'),
        'pose_raw': ROOT / 'keypoints/bp/front' / (video + '.npy'),
        'seg_raw': SEG / 'raw' / (video + '_segmentation.npz'),
        'v2_raw': SEG / 'tracking_v2/raw' / (video + '_segmentation.npz'),
        'v2_features': SEG / 'tracking_v2/features' / (video + '_features.npz'),
        'v3_selection': V3 / 'selections' / (video + '_bar_tracking_v3_selection.npz'),
        'v3_audit': V3 / 'audits' / (video + '_bar_tracking_v3_audit.json'),
    }


def audit():
    with HANDOFFS.open(encoding='utf-8-sig', newline='') as f:
        handoffs = list(csv.DictReader(f))
    with (V3 / 'manifests/bp_front_tracking_v3_validation_76_visual_review.csv').open(encoding='utf-8-sig', newline='') as f:
        reviewed = list(csv.DictReader(f))
    ids = sorted({r['video_id'] for r in handoffs} | {r['video_id'] for r in reviewed} | PRIORITY)
    sources = {str(path.relative_to(ROOT)): path for video in ids for path in paths(video).values()}
    for video in ids:
        for path in [SEG / 'features' / (video + '_features.npz'),
                     SEG / 'tracking_v2/audits' / (video + '_tracking_audit.json')]:
            sources[str(path.relative_to(ROOT))] = path
    for path in (ROOT / 'models/bp').glob('*.pt'):
        sources[str(path.relative_to(ROOT))] = path
    for path in (ROOT / 'etiquetado/bp').glob('*.csv'):
        sources[str(path.relative_to(ROOT))] = path
    for path in [HANDOFFS, V3 / 'manifests/bp_front_tracking_v3_validation_76_visual_review.csv',
                 V3 / 'manifests/bp_front_tracking_v3_validation_76_explicit_reviews.json',
                 ROOT / 'extract_keypoints/bp/extract_keypoints_front_bp.py',
                 ROOT / 'extract_segmentation/bp/extract_segmentation_front_bp.py',
                 ROOT / 'extract_segmentation/bp/bar_tracking_v3_utils.py',
                 ROOT / 'extract_segmentation/bp/rebuild_bar_tracking_v3_from_raw.py',
                 ROOT / 'models/bp/bp_front_skeleton_v8.pt', ROOT / 'models/bp/bp_front_seg_v2.pt',
                 ROOT / 'etiquetado/bp/bp_front.csv']:
        if path.exists():
            sources[str(path.relative_to(ROOT))] = path
    rows = []
    for video in ids:
        p = paths(video)
        row = {'video_id': video, 'video_path': str(p['video']), 'pose_path': str(p['pose_raw']),
               'seg_raw_path': str(p['seg_raw']), 'v2_raw_path': str(p['v2_raw']),
               'v2_features_path': str(p['v2_features']), 'v3_selection_path': str(p['v3_selection']),
               'v3_audit_path': str(p['v3_audit']), 'pose_frame_base': 0, 'coordinate_transform': 'x*width,y*height'}
        try:
            if not all(path.is_file() for path in p.values()):
                raise ValueError('archivo ausente: ' + ','.join(k for k, v in p.items() if not v.is_file()))
            cap = cv2.VideoCapture(str(p['video']))
            if not cap.isOpened():
                raise ValueError('video no se abre')
            n, fps, w, h = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS)),
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()
            pose = np.load(p['pose_raw'], allow_pickle=False)
            with np.load(p['seg_raw'], allow_pickle=False) as raw, np.load(p['v2_raw'], allow_pickle=False) as v2, np.load(p['v3_selection'], allow_pickle=False) as v3:
                metadata = json.loads(str(raw['metadata_json']))
                t = len(raw['frame_index'])
                row.update(video_frames=n, fps=fps, width=w, height=h, pose_shape=str(pose.shape),
                           raw_frames=t, raw_instances=len(raw['instance_frame_index']),
                           bar_instances=int(np.count_nonzero(v2['instance_class_name'] == 'bar')),
                           pose_both_valid_frames=int(np.count_nonzero(np.all(np.isfinite(pose[:, [4, 5], :2]), axis=(1, 2)))),
                           pose_right_valid_frames=int(np.count_nonzero(np.all(np.isfinite(pose[:, 4, :2]), axis=1))),
                           pose_left_valid_frames=int(np.count_nonzero(np.all(np.isfinite(pose[:, 5, :2]), axis=1))),
                           pose_right_conf_median=float(np.nanmedian(pose[:, 4, 2])),
                           pose_left_conf_median=float(np.nanmedian(pose[:, 5, 2])),
                           v2_track_ids='|'.join(map(str, sorted(set(v2['instance_track_id'][v2['instance_class_name'] == 'bar'].tolist())))),
                           seg_frame_base=int(raw['frame_index'][0]), v3_frame_base=int(v3['frame_index'][0]),
                           raw_video_sha256=metadata.get('video_sha256', ''),
                           seg_fps=float(metadata['fps']))
                checks = [pose.shape == (n, 6, 3), t == n, len(v2['frame_index']) == n,
                          len(v3['frame_index']) == n, np.array_equal(raw['frame_index'], np.arange(n)),
                          np.array_equal(v2['frame_index'], raw['frame_index']),
                          np.array_equal(v3['frame_index'], raw['frame_index']),
                          np.array_equal(raw['instance_frame_index'], v2['instance_frame_index']),
                          np.array_equal(raw['instance_mask_area_px'], v2['instance_mask_area_px']),
                          np.array_equal(raw['mask_byte_offsets'], v2['mask_byte_offsets']),
                          np.array_equal(raw['mask_bits'], v2['mask_bits']),
                          np.array_equal(raw['instance_bbox_xyxy_px'], v2['instance_bbox_xyxy_px']),
                          np.array_equal(raw['instance_class_name'], v2['instance_class_name']),
                          np.array_equal(raw['frame_image_shape_hw'], v2['frame_image_shape_hw']),
                          np.all(raw['frame_image_shape_hw'] == (h, w)),
                          np.allclose(raw['timestamp_sec'], np.arange(n) / fps, atol=1e-5),
                          abs(fps - float(metadata['fps'])) < 1e-4,
                          np.all((~np.isfinite(pose[:, :, :2])) | ((pose[:, :, :2] >= 0) & (pose[:, :, :2] <= 1))),
                          np.all((pose[:, :, 2] >= 0) & (pose[:, :, 2] <= 1))]
                selected = v3['selected_instance_indices']
                for frame in np.flatnonzero(selected >= 0):
                    instance = int(selected[frame])
                    if (instance >= len(v2['instance_frame_index']) or
                            int(v2['instance_frame_index'][instance]) != frame or
                            int(v2['instance_track_id'][instance]) != int(v3['selected_v2_track_ids'][frame]) or
                            v2['instance_class_name'][instance] != 'bar'):
                        raise ValueError('selección V3 no corresponde a frame/track/clase RAW: '+str(frame))
                if not all(checks):
                    raise ValueError('checks fallidos: ' + ','.join(str(i) for i, ok in enumerate(checks) if not ok))
                # La procedencia del video se contrasta por hash, no solamente por longitud.
                if digest(p['video']) != metadata['video_sha256']:
                    raise ValueError('hash video distinto del video usado en segmentación')
                row['sync_status'] = 'SEQUENCE_CONSISTENT_OVERLAY_PENDING'
                row['sync_reason'] = 'hash video, frames 0..T-1, timestamps, shape, fps y transformación consistentes; Pose carece de timestamps/hash origen'
        except Exception as exc:
            row['sync_status'] = 'BLOCKED'
            row['sync_reason'] = str(exc)
        rows.append(row)
        print(video, row['sync_status'], row.get('pose_both_valid_frames', ''), '/', row.get('video_frames', ''))
    dest = HERE / 'bp_front_pose_guided_sync_audit.csv'
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with dest.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    hashes = {relative: digest(path) for relative, path in sources.items() if path.is_file()}
    manifest = {'experiment': 'bp_front_pose_guided_feasibility_v1', 'videos': ids,
                'status': 'AUDITED_NOT_EVALUATED', 'hashes_before': hashes,
                'hashes_after': {}, 'missing_sources': [k for k, v in sources.items() if not v.is_file()],
                'pose_provenance_limitation': 'npy sin timestamps ni hash del video; verificar superposición temporal antes de evaluar'}
    (HERE / 'bp_front_pose_guided_experiment_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('audit:', len(rows), 'blocked:', sum(r['sync_status'] == 'BLOCKED' for r in rows), 'hashed:', len(hashes))


if __name__ == '__main__':
    audit()
