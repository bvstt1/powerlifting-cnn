"""Inventario sin inferencia y huellas previas; nunca abre videos candidatos."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SEG = ROOT / 'segmentation/bp/front'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ids_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return {row['video_id'] for row in csv.DictReader(stream)}


def main():
    spec = ROOT / 'segmentation/bp/front/pose_guided_tracking_v31_spec'
    actual = sha(spec / 'BP_FRONT_Pose_Guided_Tracking_V31_Specification.md')
    assert actual == '735befd58856d5d4156bf0ba4b169b2151aeec73e169d46553d67860e8c61855'
    manifest = json.loads((spec / 'BP_FRONT_Pose_Guided_Tracking_V31_Spec_Manifest.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT / name) == value for name, value in manifest['sha256'].items())
    reasons = defaultdict(set)
    sources = {
        'Prototype14': SEG / 'tracking_v3/test_14/manifests/bp_front_tracking_v3_prototype_comparison.csv',
        'Validation76': SEG / 'tracking_v3/validation_76/manifests/bp_front_tracking_v3_validation_76_visual_review.csv',
        'PoseFeasibility76': SEG / 'pose_guided_feasibility/v1/bp_front_pose_guided_sync_audit.csv',
        'manual_tracking_review': SEG / 'manifests/bp_front_bar_tracking_visual_review.csv',
        'smoke_test': SEG / 'manifests/bp_front_smoke_test_selection.csv',
        'qa_previews': SEG / 'manifests/bp_front_qa_preview_selection.csv',
    }
    for label, path in sources.items():
        for video in ids_csv(path):
            reasons[video].add(label)
    for parent in [SEG / 'tracking_v2/bar_tracking_audit_previews', SEG / 'previews']:
        if parent.exists():
            for directory in parent.glob('bp_*'):
                if directory.is_dir():
                    reasons[directory.name].add('prior_preview_dir')
    with (SEG / 'manifests/bp_front_segmentation_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        metadata = {row['video_id']: row for row in csv.DictReader(stream)}
    videos = sorted((ROOT / 'dataset/bp/front').glob('bp_*.mp4'))
    if len(videos) != 663:
        raise RuntimeError('Inventario inesperado: %d videos' % len(videos))
    video_hashes = {p.stem: sha(p) for p in videos}  # inventario solamente; no decodificación
    by_hash = defaultdict(list)
    for video, value in video_hashes.items():
        by_hash[value].append(video)
    rows = []
    for video, digest in sorted(video_hashes.items()):
        info = metadata.get(video, {})
        why = sorted(reasons[video])
        duplicate = sorted(v for v in by_hash[digest] if v != video)
        rows.append({
            'video_id': video, 'video_sha256': digest, 'exact_duplicate_ids': '|'.join(duplicate),
            'group_id': info.get('group_id', ''), 'group_id_status': info.get('group_id_status', 'UNVERIFIED'),
            'session_evidence': '', 'physical_event_evidence': '', 'prior_development': '|'.join(why),
            'reservation_status': 'EXCLUDED_FROM_CONFIRMATION' if why or duplicate else 'RESERVED_UNVERIFIED_DO_NOT_OPEN',
            'reason': 'prior development or exact duplicate' if why or duplicate else 'physical event/session/near-duplicate independence unverified',
        })
    with (HERE / 'bp_front_v31_reservation_inventory.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    registry = {str(p.relative_to(ROOT)): p for directory in [
        SEG / 'tracking_v3/test_14', SEG / 'tracking_v3/validation_76', spec,
    ] for p in directory.rglob('*') if p.is_file()}
    dev = sorted({v for v, tags in reasons.items() if 'Prototype14' in tags or 'Validation76' in tags or 'PoseFeasibility76' in tags})
    for video in dev:
        for p in [ROOT / 'dataset/bp/front' / (video + '.mp4'), ROOT / 'keypoints/bp/front' / (video + '.npy'),
                  SEG / 'raw' / (video + '_segmentation.npz'), SEG / 'features' / (video + '_features.npz'),
                  SEG / 'tracking_v2/raw' / (video + '_segmentation.npz'), SEG / 'tracking_v2/features' / (video + '_features.npz'),
                  SEG / 'tracking_v2/audits' / (video + '_tracking_audit.json')]:
            if not p.is_file():
                raise FileNotFoundError(p)
            registry[str(p.relative_to(ROOT))] = p
    for directory in [ROOT / 'models/bp', ROOT / 'etiquetado/bp', ROOT / 'training/models/bp',
                      ROOT / 'training/data/bp']:
        if directory.exists():
            registry.update({str(p.relative_to(ROOT)): p for p in directory.rglob('*') if p.is_file()})
    for p in [ROOT / 'extract_segmentation/bp/bar_tracking_v3_utils.py',
              ROOT / 'extract_segmentation/bp/rebuild_bar_tracking_v3_from_raw.py',
              ROOT / 'extract_segmentation/bp/segmentation_utils.py',
              ROOT / 'extract_keypoints/bp/extract_keypoints_front_bp.py']:
        registry[str(p.relative_to(ROOT))] = p
    hashes = {name: sha(path) for name, path in sorted(registry.items())}
    before = {'spec_sha256': actual, 'source_hashes': hashes, 'video_inventory_hashes': video_hashes,
              'prototype14_ids': sorted(ids_csv(sources['Prototype14'])),
              'validation76_ids': sorted(ids_csv(sources['Validation76'])),
              'confirmed_holdout_status': 'BLOCKED_PHYSICAL_EVENT_INDEPENDENCE_UNVERIFIED',
              'inventory_policy': 'candidate MP4 hashed, never decoded; no candidate V3.1 outputs inspected',
              'note': 'global scout/audit metadata may cover 663; listing is not physical independence proof'}
    (HERE / 'bp_front_v31_source_snapshot_before.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
    print('spec', actual, 'videos', len(rows), 'excluidos', sum(r['reservation_status']=='EXCLUDED_FROM_CONFIRMATION' for r in rows),
          'reservados sin acreditar', sum(r['reservation_status'].startswith('RESERVED') for r in rows), 'sources hashed', len(hashes))


if __name__ == '__main__':
    main()
