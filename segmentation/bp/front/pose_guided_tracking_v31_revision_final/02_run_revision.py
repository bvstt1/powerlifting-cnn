"""Una revisión sombra sobre métricas RAW cacheadas del V3.1 original, solo 85 development."""
import csv
import json
import platform
import sys
import time
from collections import Counter,defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OLD=ROOT/'segmentation/bp/front/pose_guided_tracking_v31_prototype'
sys.path.insert(0,str(OLD))
from v31_common import full_mask,sha,write_csv,json_safe
from v31_revision_identity import PoseGuidedCompetitorResolver
from v31_revision_handoff import PoseGuidedHandoffContext


def read(name):
    with (OLD/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def overlay(video,t,raw,pose,groups,old,rev):
    cap=cv2.VideoCapture(str(ROOT/'dataset/bp/front'/(video+'.mp4')))
    cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,img=cap.read();cap.release()
    if not ok:raise ValueError((video,t))
    img=cv2.resize(img,(960,540));h,w=img.shape[:2]
    palette=[(40,210,255),(255,110,0),(50,255,50),(255,60,255)]
    layer=img.copy()
    for n,row in enumerate(groups):
        mask=cv2.resize(full_mask(raw,int(row['global_instance_index'])).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
        layer[mask]=palette[n%4]
    img=cv2.addWeighted(img,.68,layer,.32,0)
    for kp,col,tag in [(5,(255,0,255),'L'),(4,(0,0,255),'R')]:
        x,y,c=pose[t,kp]
        if np.isfinite(x) and np.isfinite(y):
            p=(int(w*x),int(h*y));cv2.circle(img,p,7,col,-1)
            cv2.putText(img,'%s %.3f'%(tag,c),(p[0]+6,p[1]+15),cv2.FONT_HERSHEY_SIMPLEX,.48,col,2)
    panel=64+27*len(groups)
    img=cv2.copyMakeBorder(img,0,panel,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
    cv2.putText(img,'%s f%03d V3=%s old=%s revised=%s'%(video,t,old['v3_baseline_instance_index'],old['selected_global_instance_index'],rev['selected_global_instance_index']),
                (8,h+21),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
    cv2.putText(img,'%s -> %s  %s'%(old['status'],rev['status'],rev['reason'][:80]),(8,h+41),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    for n,row in enumerate(groups):
        text=('#%s V2%s orig%s %s->%s dL/R=%s/%s extL/R=%s/%s qL/R=%s/%s axis=%s/%s'%(
            row['global_instance_index'],row['v2_track_id'],row['extractor_track_id'],row['original_state'],row['state'],
            row['left_wrist_mask_distance'][:5],row['right_wrist_mask_distance'][:5],row['left_wrist_extension'][:5],row['right_wrist_extension'][:5],
            row['left_signed_y'][:5],row['right_signed_y'][:5],row['left_wrist_axis_distance'][:5],row['right_wrist_axis_distance'][:5]))
        cv2.putText(img,text[:148],(8,h+61+n*27),cv2.FONT_HERSHEY_SIMPLEX,.34,palette[n%4],1)
    dest=HERE/'previews'/video;dest.mkdir(parents=True,exist_ok=True)
    path=dest/('frame_%05d.jpg'%t)
    if not cv2.imwrite(str(path),img):raise IOError(path)


def main():
    start=time.perf_counter()
    pre=json.loads((HERE/'revision_pre_snapshot.json').read_text(encoding='utf-8'))
    source=json.loads((OLD/'bp_front_v31_prototype_manifest.json').read_text(encoding='utf-8'))
    assert sha(OLD/'bp_front_v31_prototype_manifest.json')==pre['original_prototype_manifest_sha256']
    assert sha(ROOT/'segmentation/bp/front/pose_guided_tracking_v31_spec/BP_FRONT_Pose_Guided_Tracking_V31_Specification.md')==pre['spec_sha256']
    for file,digest in source['output_sha256'].items():assert sha(OLD/file)==digest
    baseline=json.loads((OLD/'experimental_config.json').read_text(encoding='utf-8'))
    rev=json.loads((HERE/'experimental_revision_config.json').read_text(encoding='utf-8'))
    assert rev['frozen_spec_sha256']==pre['spec_sha256']
    ids=pre['development_ids']
    old_frames=read('bp_front_v31_frame_identity.csv')
    old_traj=read('bp_front_v31_selected_trajectories.csv')
    old_edges=read('bp_front_v31_handoff_decisions.csv')
    old_track=read('bp_front_v31_track_summary.csv')
    sync=read('bp_front_v31_sync_audit.csv')
    assert len(ids)==85 and all(r['status']=='SEQUENCE_AND_VISUAL_SUPPORTED' for r in sync)
    fps_by_video={r['video_id']:float(r['fps']) for r in sync}
    by_frame=defaultdict(lambda:defaultdict(list));by_track=defaultdict(lambda:defaultdict(list));by_traj=defaultdict(dict);by_edge=defaultdict(list);by_track_summary=defaultdict(dict)
    for r in old_frames:
        r=dict(r,original_state=r['state'],original_reason=r['reason'])
        t=int(r['frame_index']);by_frame[r['video_id']][t].append(r);by_track[r['video_id']][r['v2_track_id']].append(r)
    for r in old_traj:by_traj[r['video_id']][int(r['frame_index'])]=r
    for r in old_edges:by_edge[r['video_id']].append(r)
    for r in old_track:by_track_summary[r['video_id']][r['track_id']]=int(r['switch_count'])
    resolver=PoseGuidedCompetitorResolver(rev);corroborator=PoseGuidedHandoffContext(baseline,rev)
    frames_out=[];traj_out=[];handoff_out=[];per_video=[];dominance_all=[];context_all=[];preview_index=[]
    priority={'bp_518','bp_520','bp_517','bp_311','bp_585','bp_277','bp_478','bp_571',
              'bp_663','bp_662','bp_182','bp_238','bp_538','bp_540','bp_544','bp_102','bp_103','bp_198','bp_301','bp_467','bp_469'}
    diagnostics=defaultdict(set)
    with (HERE/'diagnosis_preview_index.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):diagnostics[r['video_id']].add(int(r['frame_index']))
    for video in ids:
        rows=by_frame[video];tracks=by_track[video];old=by_traj[video];edges=by_edge[video];summary=by_track_summary[video]
        assert len(old)>0 and video in pre['inventory_video_hashes']
        dominated=resolver.resolve(rows)
        restored,context_notes=corroborator.corroborate(video,edges,None,tracks,rows,summary,old)
        dominance_all.extend(dominated.values());context_all.extend(context_notes)
        invalid_edges=[e for e in edges if e['selected_by_v3']=='True' and e['pose_decision']=='POSE_INCOMPATIBLE']
        uncertain_edges=[e for e in edges if e['selected_by_v3']=='True' and e['pose_decision']=='POSE_INSUFFICIENT_EVIDENCE']
        confident=[];changes=Counter();local={}
        for t in sorted(old):
            original=old[t];available=rows.get(t,[]);v3idx=original['v3_baseline_instance_index']
            previous=next((r for r in available if r['global_instance_index']==original['selected_global_instance_index']),None)
            chosen=previous if previous and previous['state']=='SUPPORTED_HELD_BAR' else None
            why=original['reason']
            if chosen is None and original['status']!='SYNC_BLOCKED':
                if t in restored:chosen=restored[t];why='REVISION_COMPATIBLE_EDGE_AND_INDEPENDENT_CONTEXT'
                else:
                    supported=[r for r in available if r['state']=='SUPPORTED_HELD_BAR']
                    if len(supported)==1:chosen=supported[0];why=supported[0]['reason']
            if chosen is not None:
                if any(int(e['transition_frame'])<=t<int(e['transition_frame'])+baseline['context_frames'] and e['target_track']==chosen['v2_track_id'] for e in invalid_edges):
                    chosen=None;why='INCOMPATIBLE_V3_EDGE_BLOCKS_TARGET_CONTEXT'
                elif any(int(e['transition_frame'])<=t<int(e['transition_frame'])+baseline['near_frames'] and e['target_track']==chosen['v2_track_id'] for e in uncertain_edges):
                    chosen=None;why='INSUFFICIENT_V3_EDGE_AT_BORDER'
            if original['status']=='OBSERVED_INCOMPATIBLE' or not available or original['status']=='SYNC_BLOCKED':
                chosen=None;why=original['reason']
            selected=int(chosen['global_instance_index']) if chosen else -1
            if selected>=0:status='OBSERVED_CONFIDENT'
            elif not available:status='NO_BAR_DETECTED'
            elif original['status']=='OBSERVED_INCOMPATIBLE':status='OBSERVED_INCOMPATIBLE'
            else:status='OBSERVED_UNVERIFIED'
            if selected>=0 and original['selected_global_instance_index']=='-1':changes[why]+=1
            point=(float(chosen['centroid_x']),float(chosen['centroid_y'])) if chosen else None
            previous_row=local.get(t-1)
            has_compatible_edge=bool(chosen is not None and previous_row and any(
                edge['selected_by_v3']=='True' and edge['pose_decision']=='POSE_COMPATIBLE' and
                int(edge['transition_frame'])==t and edge['source_subtracklet']==previous_row['subtracklet_id']
                and edge['target_subtracklet']==chosen['subtracklet_id']
                for edge in edges))
            velocity_link=bool(point is not None and previous_row and previous_row['selected_global_instance_index']!=-1 and
                               (previous_row['subtracklet_id']==chosen['subtracklet_id'] or has_compatible_edge))
            if velocity_link:
                vx=(point[0]-previous_row['bar_x_px'])*fps_by_video[video];vy=(point[1]-previous_row['bar_y_px'])*fps_by_video[video];velocity=1
            else:vx=vy=float('nan');velocity=0
            row=dict(video_id=video,frame_index=t,timestamp_sec=original['timestamp_sec'],status=status,
                     selected_global_instance_index=selected,v3_baseline_instance_index=int(v3idx),
                     v31_original_instance_index=int(original['selected_global_instance_index']),
                     v2_baseline_instance_index=int(original['v2_baseline_instance_index']),
                     v2_track_id=chosen['v2_track_id'] if chosen else -1,
                     extractor_track_id=chosen['extractor_track_id'] if chosen else -1,
                     subtracklet_id=chosen['subtracklet_id'] if chosen else '',
                     bar_present_raw=int(bool(available)),bar_identity_valid=int(selected>=0),position_observed=int(selected>=0),
                     bar_x_px=point[0] if point is not None else float('nan'),bar_y_px=point[1] if point is not None else float('nan'),
                     velocity_valid=velocity,vx_px_s=vx,vy_px_s=vy,
                     reason=why,original_status=original['status'],original_reason=original['reason'],
                     pose_quality=chosen['evidence_quality'] if chosen else 'INSUFFICIENT',
                     sync_status=original['sync_status'],spec_sha256=pre['spec_sha256'],
                     source_v2_sha256=original['source_v2_sha256'])
            local[t]=row;traj_out.append(row);confident.append(selected)
        for group in rows.values():
            for r in group:frames_out.append(r)
        for edge in edges:
            if edge['selected_by_v3']!='True':continue
            t=int(edge['transition_frame']);o=(old.get(t-1),old.get(t));now=(local.get(t-1),local.get(t))
            handoff_out.append(dict(video_id=video,source_track=edge['source_track'],target_track=edge['target_track'],
                source_subtracklet=edge['source_subtracklet'],target_subtracklet=edge['target_subtracklet'],
                transition_frame=t,geometric_score=edge['geometric_score'],original_total_edge_score=edge['original_total_edge_score'],
                pose_decision_original=edge['pose_decision'],pose_decision_revised=edge['pose_decision'],
                original_both_confident=bool(all(z and z['selected_global_instance_index']!='-1' for z in o)),
                revision_both_confident=bool(all(z and z['selected_global_instance_index']!=-1 for z in now)),
                before_reason=now[0]['reason'] if now[0] else '',after_reason=now[1]['reason'] if now[1] else '',
                original_review_status='NOT_INPUT_TO_ALGORITHM'))
        per_video.append(dict(video_id=video,frames=len(old),v3_selected=sum(z['v3_baseline_instance_index']!='-1' for z in old.values()),
            v31_original_confident=sum(z['selected_global_instance_index']!='-1' for z in old.values()),
            revision_confident=sum(x>=0 for x in confident),revision_unverified=sum(x['status']=='OBSERVED_UNVERIFIED' for x in local.values()),
            revision_incompatible=sum(x['status']=='OBSERVED_INCOMPATIBLE' for x in local.values()),
            no_bar=sum(x['status']=='NO_BAR_DETECTED' for x in local.values()),
            restored_competitor=sum(k=='REVISION_PERSISTENT_BILATERAL_MASK_AND_EXTENSION_DOMINANCE' for k in [z['reason'] for z in local.values() if z['original_status']!='OBSERVED_CONFIDENT' and z['status']=='OBSERVED_CONFIDENT']),
            restored_context=changes['REVISION_COMPATIBLE_EDGE_AND_INDEPENDENT_CONTEXT'],
            recovered_vs_old=sum(x>=0 for x in confident)-sum(z['selected_global_instance_index']!='-1' for z in old.values()),
            intratrack_splits=sum(summary.values())))
        if video in priority:
            samples=set(diagnostics[video])
            samples|={0,len(old)//2,len(old)-1}
            for e in edges:
                if e['selected_by_v3']=='True':samples|={max(0,int(e['transition_frame'])-1),int(e['transition_frame'])}
            # Muestra en bordes donde se promueve identidad; no inspeccionar candidatos reservados.
            new=[t for t,z in local.items() if z['selected_global_instance_index']!=-1 and old[t]['selected_global_instance_index']=='-1']
            if new:samples|={new[0],new[len(new)//2],new[-1]}
            pose=np.load(ROOT/'keypoints/bp/front'/(video+'.npy'),allow_pickle=False)
            with np.load(ROOT/'segmentation/bp/front/tracking_v2/raw'/(video+'_segmentation.npz'),allow_pickle=False) as raw:
                for t in sorted(samples):
                    overlay(video,t,raw,pose,rows.get(t,[]),old[t],local[t]);preview_index.append(dict(video_id=video,frame_index=t,
                        path='previews/%s/frame_%05d.jpg'%(video,t)))
        print(video,'old',sum(z['selected_global_instance_index']!='-1' for z in old.values()),'rev',sum(x>=0 for x in confident),dict(changes))
    outputs={'bp_front_v31_revision_frame_identity.csv':frames_out,
             'bp_front_v31_revision_selected_trajectories.csv':traj_out,
             'bp_front_v31_revision_handoff_comparison.csv':handoff_out,
             'bp_front_v31_revision_per_video_comparison.csv':per_video,
             'bp_front_v31_revision_dominance_evidence.csv':dominance_all,
             'bp_front_v31_revision_context_evidence.csv':context_all,
             'bp_front_v31_revision_preview_index.csv':preview_index}
    for name,data in outputs.items():write_csv(HERE/name,data,list(dict.fromkeys(k for r in data for k in r)))
    manifest=dict(revision_id=rev['revision_id'],status='SHADOW_DEVELOPMENT_ONLY',spec_sha256=pre['spec_sha256'],
                  revision_config_sha256=sha(HERE/'experimental_revision_config.json'),
                  original_config_sha256=sha(OLD/'experimental_config.json'),
                  original_prototype_manifest_sha256=pre['original_prototype_manifest_sha256'],
                  input_snapshot_sha256=sha(HERE/'revision_pre_snapshot.json'),
                  visual_review_not_algorithm_input='bp_front_v31_revision_visual_review.csv',
                  videos=ids,counts=dict(videos=len(ids),instances=len(frames_out),frames=len(traj_out),
                                         recovered_vs_original=sum(int(x['recovered_vs_old']) for x in per_video)),
                  elapsed_seconds=time.perf_counter()-start,
                  runtime=dict(python=sys.version,platform=platform.platform(),opencv=cv2.__version__,numpy=np.__version__,seed=baseline['seed']),
                  output_sha256={name:sha(HERE/name) for name in outputs})
    (HERE/'bp_front_v31_revision_manifest.json').write_text(json.dumps(json_safe(manifest),indent=2,ensure_ascii=False),encoding='utf-8')
    print('FINISHED',len(ids),'recovered',manifest['counts']['recovered_vs_original'],'seconds',round(manifest['elapsed_seconds'],1))


if __name__=='__main__':main()
