"""Congela entradas originales antes de la única revisión; no abre candidatos reservados."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OLD = HERE.parent / '03_prototype_v31'
SPEC = HERE.parent / '02_specification'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def main():
    spec=sha(SPEC/'BP_FRONT_Pose_Guided_Tracking_V31_Specification.md')
    assert spec=='735befd58856d5d4156bf0ba4b169b2151aeec73e169d46553d67860e8c61855'
    origin=json.loads((OLD/'bp_front_v31_source_snapshot_before.json').read_text(encoding='utf-8'))
    final=json.loads((OLD/'bp_front_v31_prototype_manifest.json').read_text(encoding='utf-8'))
    assert final['final_status']=='PROTOTYPE_NEEDS_REVISION'
    for rel, digest in final['final_code_sha256'].items():
        assert sha(OLD/rel)==digest, rel
    for rel, digest in final['final_artifact_sha256'].items():
        assert sha(OLD/rel)==digest,rel
    for rel, digest in final['output_sha256'].items():
        assert sha(OLD/rel)==digest,rel
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in OLD.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    # Mantener inventario RAW/V2/V3/YOLO/splits del experimento anterior como ancla inmutable.
    for rel,digest in origin['source_hashes'].items():
        assert sha(ROOT/rel)==digest,rel
    for vid,digest in origin['video_inventory_hashes'].items():
        assert sha(ROOT/'dataset/bp/front'/(vid+'.mp4'))==digest,vid
    assert all(sha(ROOT/rel)==digest for rel,digest in json.loads((SPEC/'BP_FRONT_Pose_Guided_Tracking_V31_Spec_Manifest.json').read_text(encoding='utf-8'))['sha256'].items())
    data={'spec_sha256':spec,'original_prototype_manifest_sha256':sha(OLD/'bp_front_v31_prototype_manifest.json'),
          'original_prototype_files':hashes,'original_source_hashes':origin['source_hashes'],
          'inventory_video_hashes':origin['video_inventory_hashes'],
          'reserved_ids_never_open':True,'development_ids':sorted(set(origin['prototype14_ids'])|set(origin['validation76_ids'])),
          'confirmatory_status':origin['confirmed_holdout_status']}
    (HERE/'revision_pre_snapshot.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    print('preflight OK original prototype',len(hashes),'sources',len(origin['source_hashes']),'MP4',len(origin['video_inventory_hashes']))


if __name__=='__main__':main()
