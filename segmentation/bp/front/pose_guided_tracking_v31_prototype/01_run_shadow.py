"""Prototipo sombra BP_FRONT: dev=union(Prototype14,Validation76), nunca reserva."""
import csv
import argparse
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from v31_common import HERE, ROOT, SEG, SPEC_SHA, sha, json_safe, load, full_mask, write_csv
from v31_identity import PoseGuidedBarIdentityValidator
from v31_consistency import PoseGuidedTrackConsistencyValidator
from v31_handoff import PoseGuidedHandoffValidator, recalc

REVIEW = SEG/'tracking_v3/validation_76/manifests/bp_front_tracking_v3_validation_76_handoffs.csv'
PREVIEWS = {'bp_663','bp_662','bp_182','bp_238','bp_538','bp_540','bp_544','bp_102','bp_103','bp_198','bp_301','bp_467','bp_469','bp_520'}


def review_rows():
    with REVIEW.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def preview(video,frame,raw,pose,v2idx,v3idx,newidx,rows,cut_map):
    cap=cv2.VideoCapture(str(ROOT/'dataset/bp/front'/(video+'.mp4')))
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame); ok,img=cap.read();cap.release()
    if not ok or img.shape[:2]!=tuple(raw['frame_image_shape_hw'][frame]):return False
    h,w=img.shape[:2];scale=min(1,960/w); img=cv2.resize(img,(int(w*scale),int(h*scale)))
    begin,end=map(int,raw['frame_instance_offsets'][frame:frame+2]); colors=[(0,215,250),(255,140,0),(40,220,40)]
    for ordinal,idx in enumerate(j for j in range(begin,end) if raw['instance_class_name'][j]=='bar'):
        mask=cv2.resize(full_mask(raw,idx).astype(np.uint8),(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool)
        color=colors[ordinal%3];img[mask]=(img[mask]*.6+np.asarray(color)*.4).astype(np.uint8)
        bbox=raw['instance_bbox_xyxy_px'][idx]; label='%d t%d %s %s'%(idx,int(raw['instance_track_id'][idx]),cut_map.get(idx,''),'V31' if idx==newidx else 'V3' if idx==v3idx else '')
        cv2.putText(img,label,(int(bbox[0]*scale),max(36,int(bbox[1]*scale)-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
    for k,color,label in [(5,(255,0,255),'L'),(4,(0,0,255),'R')]:
        x,y,c=pose[frame,k]
        if np.isfinite(x) and np.isfinite(y):
            xy=(int(x*img.shape[1]),int(y*img.shape[0]));cv2.circle(img,xy,6,color,-1);cv2.putText(img,'%s %.3f'%(label,c),(xy[0]+6,xy[1]+15),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
    cv2.putText(img,'%s f%d V2=%d V3=%d V31=%d'%(video,frame,v2idx,v3idx,newidx),(14,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    dest=HERE/'previews'/video/('frame_%05d.jpg'%frame);dest.parent.mkdir(parents=True,exist_ok=True)
    return cv2.imwrite(str(dest),img)


def ablation(video,variant,frames,selection,v2indices,frame_rows,tracks,handoffs,confident):
    count=0; abstain=0; proxy_bad=0
    suspect={t['track_id'] for t in tracks if t['consistency_state']=='SUSPECT_CONTAMINATED'}
    bad_edges=[x for x in handoffs if x['selected_by_v3'] and x['pose_decision']=='POSE_INCOMPATIBLE']
    blocked=set()
    for edge in bad_edges:
        target=edge['target_track'];t=edge['transition_frame']
        blocked.update(i for i in range(t,frames) if any(r['v2_track_id']==target for r in frame_rows.get(i,[])))
    for t in range(frames):
        idx=int(selection[t]); r=next((r for r in frame_rows.get(t,[]) if r['global_instance_index']==idx),None)
        if variant=='V3':selected=idx
        elif variant=='IDENTITY_ONLY':selected=idx if r and r['state']=='SUPPORTED_HELD_BAR' else -1
        elif variant=='TRACK_ONLY':selected=idx if r and r['v2_track_id'] not in suspect else -1
        elif variant=='HANDOFF_ONLY':selected=idx if t not in blocked else -1
        else:selected=int(confident[t])
        if selected>=0:count+=1
        elif idx>=0:abstain+=1
        if selected>=0:
            rr=next((r for r in frame_rows.get(t,[]) if r['global_instance_index']==selected),None)
            if rr and rr['state']=='INCOMPATIBLE_WITH_HELD_BAR':proxy_bad+=1
    return dict(video_id=video,ablation=variant,frames=frames,selected_frames=count,abstained_from_v3_frames=abstain,
                proxy_selected_incompatible_frames=proxy_bad,reference='no visual frame truth; incompatibility is Pose proxy only')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--videos',nargs='*',help='IDs de desarrollo solamente; sin permiso para candidatos reservados')
    args=parser.parse_args()
    start=time.perf_counter()
    before=json.loads((HERE/'bp_front_v31_source_snapshot_before.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'segmentation/bp/front/pose_guided_tracking_v31_spec/BP_FRONT_Pose_Guided_Tracking_V31_Specification.md')==SPEC_SHA
    config_path=HERE/'experimental_config.json';config=json.loads(config_path.read_text(encoding='utf-8'))
    assert config['specification_sha256']==SPEC_SHA
    np.random.seed(config['seed'])
    videos=sorted(set(before['prototype14_ids'])|set(before['validation76_ids']))
    if args.videos:
        forbidden=set(args.videos)-set(videos)
        if forbidden:raise ValueError('outside development: '+str(sorted(forbidden)))
        videos=sorted(set(args.videos))
    reservations={}
    with (HERE/'bp_front_v31_reservation_inventory.csv').open(encoding='utf-8') as f:
        reservations={r['video_id']:r for r in csv.DictReader(f)}
    assert all('EXCLUDED' in reservations[v]['reservation_status'] for v in videos)
    original_reviews=review_rows();reviews=defaultdict(list)
    for r in original_reviews:reviews[r['video_id']].append(r)
    identity=PoseGuidedBarIdentityValidator(config); consistency=PoseGuidedTrackConsistencyValidator(config); validator=PoseGuidedHandoffValidator(config)
    frame_all=[];track_all=[];segments_all=[];edges_all=[];trajectory_all=[];comparison=[];ablations=[];sync=[];errors=[]
    for vid in videos:
        begun=time.perf_counter();raw=v2=sel=feat=None
        try:
            pose,raw,v2,sel,feat,audit,preflight=load(vid,before['video_inventory_hashes'][vid])
            if preflight['status']!='SEQUENCE_AND_VISUAL_SUPPORTED':
                raise ValueError('SYNC_BLOCKED: '+str([k for k,v in preflight['checks'].items() if not v]))
            raw_selection=sel['selected_instance_indices'];n=len(pose)
            classes=list(feat['class_names'].astype(str));v2indices=feat['selected_instance_indices'][:,classes.index('bar')]
            rows,by_frame,by_track=identity.evaluate(vid,pose,v2)
            tracks,segments,by_segment,cut=consistency.evaluate(vid,by_track)
            for r in rows:r['subtracklet_id']=cut[r['global_instance_index']]
            for s in segments:s['v3_branch']=preflight['v3_branch']
            # No utilizar review/status visual como entrada a la lógica; consulta exclusivamente audits geométricos.
            edge_rows=[];seen=set()
            for candidate in audit['handoff_candidates']:
                if 'transition_frame' not in candidate or 'source_frame' not in candidate or 'target_frame' not in candidate:continue
                source,target=int(candidate['source_track']),int(candidate['target_track']);t=int(candidate['transition_frame'])
                sr=next((r for r in by_track.get(source,[]) if r['global_instance_index']==int(candidate['source_instance_index'])),None)
                dr=next((r for r in by_track.get(target,[]) if r['global_instance_index']==int(candidate['target_instance_index'])),None)
                if sr is None or dr is None:continue
                source_sid=cut[sr['global_instance_index']];target_sid=cut[dr['global_instance_index']]
                source_rows=by_segment[source_sid];target_rows=by_segment[target_sid]
                split_supported=any(s['status']=='SUBTRACKLET_PROPOSED' and s['subtracklet_id'] in (source_sid,target_sid) for s in segments)
                selected=any(e['source_track']==source and e['target_track']==target and e['transition_frame']==t for e in audit['accepted_handoffs'])
                recomputed=len([s for s in segments if s['original_v2_track_id'] in (source,target) and s['status']=='SUBTRACKLET_PROPOSED'])>0
                geomscore=candidate['total_score'];geom_accepted=bool(candidate['accepted']);geom_reason=candidate.get('rejection_reason','')
                if recomputed:
                    rec=recalc(source_rows,target_rows,v2);geomscore=rec.get('total_score');geom_accepted=bool(rec['accepted']);geom_reason=rec.get('rejection_reason','')
                outcome=validator.evaluate(vid,source_rows,target_rows,t,geomscore,selected,
                                            '%s>%s@%d'%(source_sid,target_sid,t),recomputed,split_supported)
                outcome.update(v3_geometry_accepted=bool(candidate['accepted']),v31_recomputed_geometry_accepted=geom_accepted,
                               v31_recomputed_geometry_reason=geom_reason,original_total_edge_score=candidate['total_score'],
                               original_visual_review_status='NOT_INPUT_TO_ALGORITHM')
                edge_rows.append(outcome);seen.add((source_sid,target_sid,t))
            # Candidatos dentro del mismo ID V2 solo nacen de subtracklets: score SIEMPRE recalculado.
            for track in tracks:
                parts=sorted([s for s in segments if s['original_v2_track_id']==track['track_id']],key=lambda s:s['segment_ordinal'])
                for a,b in zip(parts[:-1],parts[1:]):
                    aa,bb=by_segment[a['subtracklet_id']],by_segment[b['subtracklet_id']];t=b['first_frame']
                    rec=recalc(aa,bb,v2)
                    outcome=validator.evaluate(vid,aa,bb,t,rec.get('total_score'),False,
                                                '%s>%s@%d'%(a['subtracklet_id'],b['subtracklet_id'],t),True,True)
                    outcome.update(v3_geometry_accepted=False,v31_recomputed_geometry_accepted=bool(rec['accepted']),
                                   v31_recomputed_geometry_reason=rec.get('rejection_reason',''),original_total_edge_score=None,
                                   original_visual_review_status='NOT_INPUT_TO_ALGORITHM',virtual_internal_edge=True)
                    edge_rows.append(outcome)
            edge_lookup={(e['source_track'],e['target_track'],e['transition_frame']):e for e in edge_rows if e['selected_by_v3']}
            # Selección de semilla desde TODOS los segmentos, no track principal V2.
            seeds=[]
            for segment in segments:
                rs=by_segment[segment['subtracklet_id']];supported=[r for r in rs if r['state']=='SUPPORTED_HELD_BAR']
                if len(supported)>=config['main_bar_support_min_frames']:
                    seeds.append((supported[0]['frame_index'],segment['subtracklet_id'],len(supported)))
            seeds.sort(key=lambda x:(x[0],-x[2],x[1]));seed=seeds[0][1] if seeds and (len(seeds)==1 or seeds[0][0]<seeds[1][0]) else None
            confident=np.full(n,-1,np.int64);statuses=[]
            for t in range(n):
                candidate_rows=by_frame.get(t,[]);v3idx=int(raw_selection[t]);legacy=next((r for r in candidate_rows if r['global_instance_index']==v3idx),None)
                supported=[r for r in candidate_rows if r['state']=='SUPPORTED_HELD_BAR']
                candidate=legacy if legacy and legacy['state']=='SUPPORTED_HELD_BAR' else supported[0] if len(supported)==1 else None
                if not candidate_rows:status='NO_BAR_DETECTED';reason='no BAR RAW'
                elif candidate is None:
                    status='OBSERVED_INCOMPATIBLE' if len(candidate_rows)==1 and candidate_rows[0]['state']=='INCOMPATIBLE_WITH_HELD_BAR' else 'OBSERVED_UNVERIFIED'
                    reason='sole BAR anatomy incompatible' if status=='OBSERVED_INCOMPATIBLE' else 'identity/competition not resolved'
                elif seed is None:
                    status='OBSERVED_UNVERIFIED';reason='MAIN_BAR_UNRESOLVED'
                else:
                    status='OBSERVED_CONFIDENT';reason='bilateral temporal evidence; seed '+seed
                    # Un handoff incompatible nunca respalda una unión; evitar publicar el borde como observación segura.
                    for edge in edge_rows:
                        if edge['selected_by_v3'] and edge['pose_decision']=='POSE_INCOMPATIBLE' and edge['transition_frame']<=t<edge['transition_frame']+config['context_frames'] and edge['target_track']==candidate['v2_track_id']:
                            status='OBSERVED_UNVERIFIED';reason='V3 edge incompatible; possible independent re-acquisition pending';break
                    # Un edge indeterminado con posible competencia debe abstenerse cerca del borde.
                    if status=='OBSERVED_CONFIDENT':
                        for edge in edge_rows:
                            if edge['selected_by_v3'] and edge['pose_decision']=='POSE_INSUFFICIENT_EVIDENCE' and edge['transition_frame']<=t<edge['transition_frame']+config['near_frames'] and edge['target_track']==candidate['v2_track_id']:
                                status='OBSERVED_UNVERIFIED';reason='selected V3 handoff without bilateral proof';break
                idx=int(candidate['global_instance_index']) if status=='OBSERVED_CONFIDENT' else -1
                confident[t]=idx;statuses.append(status)
                prev=trajectory_all[-1] if trajectory_all and trajectory_all[-1]['video_id']==vid else None
                if idx>=0 and prev and prev['frame_index']==t-1 and prev['selected_global_instance_index']>=0:
                    center=(candidate['centroid_x'],candidate['centroid_y']);dx=center[0]-prev['bar_x_px'];dy=center[1]-prev['bar_y_px'];vx=dx*preflight['fps'];vy=dy*preflight['fps'];vel=1
                else:vx=vy=float('nan');vel=0
                trajectory_all.append(dict(video_id=vid,frame_index=t,timestamp_sec=float(v2['timestamp_sec'][t]),status=status,
                    bar_present_raw=int(bool(candidate_rows)),selected_global_instance_index=idx,v3_baseline_instance_index=v3idx,
                    v2_baseline_instance_index=int(v2indices[t]),v2_track_id=candidate['v2_track_id'] if idx>=0 else -1,
                    extractor_track_id=candidate['extractor_track_id'] if idx>=0 else -1,subtracklet_id=candidate['subtracklet_id'] if idx>=0 else '',
                    bar_x_px=candidate['centroid_x'] if idx>=0 else float('nan'),bar_y_px=candidate['centroid_y'] if idx>=0 else float('nan'),
                    position_observed=int(idx>=0),bar_identity_valid=int(idx>=0),velocity_valid=vel,
                    vx_px_s=vx,vy_px_s=vy,pose_quality=candidate['evidence_quality'] if candidate else 'INSUFFICIENT',
                    identity_uncertainty='LOW' if idx>=0 else 'HIGH',reason=reason,sync_status=preflight['status'],
                    provenance_spec_sha256=SPEC_SHA,source_v2_sha256=audit['source_v2_sha256']))
            if vid in PREVIEWS:
                frames={0,n//2,n-1}
                for e in edge_rows:
                    if e['selected_by_v3'] or e.get('virtual_internal_edge') or vid in {'bp_662','bp_663'}:
                        frames.update([max(0,e['transition_frame']-1),min(n-1,e['transition_frame'])])
                for t in sorted(frames):
                    preview(vid,t,v2,pose,int(v2indices[t]),int(raw_selection[t]),int(confident[t]),by_frame.get(t,[]),cut)
            frame_all+=rows;track_all+=tracks;segments_all+=segments;edges_all+=edge_rows
            for variant in ('V3','IDENTITY_ONLY','TRACK_ONLY','HANDOFF_ONLY','FULL_V31'):
                ablations.append(ablation(vid,variant,n,raw_selection,v2indices,by_frame,tracks,edge_rows,confident))
            comparison.append(dict(video_id=vid,v3_branch=preflight['v3_branch'],frames_total=n,bar_raw_detected_frames=sum(bool(by_frame.get(t)) for t in range(n)),
                v2_selected_frames=int(np.count_nonzero(v2indices>=0)),v3_selected_frames=int(np.count_nonzero(raw_selection>=0)),
                v31_confident_frames=int(np.count_nonzero(confident>=0)),v31_unverified_frames=statuses.count('OBSERVED_UNVERIFIED'),
                v31_incompatible_frames=statuses.count('OBSERVED_INCOMPATIBLE'),no_bar_frames=statuses.count('NO_BAR_DETECTED'),
                v3_selected_edges=sum(e['selected_by_v3'] for e in edge_rows),pose_incompatible_selected_edges=sum(e['selected_by_v3'] and e['pose_decision']=='POSE_INCOMPATIBLE' for e in edge_rows),
                pose_insufficient_selected_edges=sum(e['selected_by_v3'] and e['pose_decision']=='POSE_INSUFFICIENT_EVIDENCE' for e in edge_rows),
                intratrack_splits=sum(t['switch_count'] for t in tracks),v2_main_track_id=audit.get('v2_selected_track_id'),
                v31_seed_subtracklet=seed or 'MAIN_BAR_UNRESOLVED',elapsed_seconds=time.perf_counter()-begun))
            sync.append(preflight)
            print(vid,'OK',len(rows),'BAR',len(tracks),'tracks',sum(t['switch_count'] for t in tracks),'splits','V31',int(np.count_nonzero(confident>=0)),'/',n)
        except Exception as exc:
            errors.append(dict(video_id=vid,error=repr(exc)));print(vid,'BLOCKED',repr(exc))
            sync.append(dict(video_id=vid,status='SYNC_BLOCKED' if 'SYNC_BLOCKED' in repr(exc) else 'PROCESSING_BLOCKED',reason=repr(exc)))
        finally:
            for obj in (raw,v2,sel,feat):
                if obj is not None:obj.close()
    outputs=[('bp_front_v31_frame_identity.csv',frame_all),('bp_front_v31_track_consistency.csv',segments_all),
             ('bp_front_v31_track_summary.csv',track_all),('bp_front_v31_handoff_decisions.csv',edges_all),
             ('bp_front_v31_selected_trajectories.csv',trajectory_all),('bp_front_v31_comparison_v2_v3_v31.csv',comparison),
             ('bp_front_v31_ablation_results.csv',ablations),('bp_front_v31_sync_audit.csv',sync)]
    for name,rows in outputs:
        write_csv(HERE/name,rows,list(dict.fromkeys(k for r in rows for k in r)))
    runtime={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'opencv':cv2.__version__,
             'torch':'not imported or used','seed':config['seed'],'elapsed_seconds':time.perf_counter()-start}
    try:
        import importlib.metadata as md
        runtime['torch_installed_version']=md.version('torch')
    except Exception:runtime['torch_installed_version']='not available'
    manifest=dict(status='SHADOW_DEVELOPMENT_ONLY',spec_sha256=SPEC_SHA,config_sha256=sha(config_path),
                  source_snapshot_before='bp_front_v31_source_snapshot_before.json',
                  holdout_status=before['confirmed_holdout_status'],development_videos=videos,
                  warnings=['Pose NPY lacks original video hash/timestamps','No independent frame-level visual ground truth','All numeric Pose thresholds exploratory'],
                  errors=errors,counts={'videos_requested':len(videos),'videos_completed':len(comparison),'frames':len(trajectory_all),'bar_instances':len(frame_all),
                                        'accepted_v3_handoffs_in_output':sum(x['selected_by_v3'] for x in edges_all)},
                  runtime=runtime,output_sha256={name:sha(HERE/name) for name,_ in outputs})
    (HERE/'bp_front_v31_prototype_manifest.json').write_text(json.dumps(json_safe(manifest),indent=2,ensure_ascii=False),encoding='utf-8')
    print('completed',len(comparison),'/',len(videos),'errors',errors,'runtime',round(runtime['elapsed_seconds'],1))


if __name__=='__main__':main()
