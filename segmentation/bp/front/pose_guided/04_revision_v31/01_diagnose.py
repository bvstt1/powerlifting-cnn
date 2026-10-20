"""Diagnóstico previo a cambios: estados/medidas de todas las BAR omitidas y revisión visual."""
import csv
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
OLD=HERE.parent/'03_prototype_v31'
sys.path.insert(0,str(OLD))
from v31_common import full_mask


def rows(name):
    with (OLD/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def write(name,values):
    with (HERE/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)


def preview(video,frames,pose,raw,points,trajectory):
    dest=HERE/'diagnostic_previews'/video;dest.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(ROOT/'dataset/bp/front'/(video+'.mp4')))
    results=[]
    for t in sorted(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,img=cap.read()
        if not ok:raise RuntimeError((video,t))
        img=cv2.resize(img,(960,540));h,w=img.shape[:2]
        groups=points[t];r=trajectory[t];layer=img.copy()
        cols=[(40,205,250),(255,100,25),(70,250,70),(255,30,210)]
        for n,pt in enumerate(groups):
            idx=int(pt['global_instance_index']);color=cols[n%4]
            mask=cv2.resize(full_mask(raw,idx).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
            layer[mask]=color
        img=cv2.addWeighted(img,.67,layer,.33,0)
        for kp,col,label in [(5,(255,0,255),'L'),(4,(0,0,255),'R')]:
            x,y,conf=pose[t,kp]
            if np.isfinite(x) and np.isfinite(y):
                p=(int(w*x),int(h*y));cv2.circle(img,p,7,col,-1);cv2.putText(img,'%s %.3f'%(label,conf),(p[0]+6,p[1]+15),cv2.FONT_HERSHEY_SIMPLEX,.5,col,2)
        img=cv2.copyMakeBorder(img,0,58+28*len(groups),0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
        cv2.putText(img,'%s f%03d V3=%s OLD=%s %s'%(video,t,r['v3_baseline_instance_index'],r['selected_global_instance_index'],r['status']),
                    (9,h+19),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1)
        cv2.putText(img,r['reason'][:104],(9,h+38),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
        for n,pt in enumerate(groups):
            idx=int(pt['global_instance_index']);text=(
                '#%d tr%s orig%s %s conf%s d=%s/%s axis=%s/%s ext=%s/%s q=%s/%s width=%s n=%s motion=%s'%(
                    idx,pt['v2_track_id'],pt['extractor_track_id'],pt['state'],pt['mask_confidence'][:5],
                    pt['left_wrist_mask_distance'][:5],pt['right_wrist_mask_distance'][:5],
                    pt['left_wrist_axis_distance'][:5],pt['right_wrist_axis_distance'][:5],
                    pt['left_wrist_extension'][:5],pt['right_wrist_extension'][:5],
                    pt['left_signed_y'][:5],pt['right_signed_y'][:5],pt['mask_width_px'][:5],
                    pt['context_frames'],pt['relative_motion_median_px'][:5]))
            cv2.putText(img,text[:145],(9,h+56+28*n),cv2.FONT_HERSHEY_SIMPLEX,.33,cols[n%4],1)
        path=dest/('frame_%05d.jpg'%t)
        if not cv2.imwrite(str(path),img):raise IOError(path)
        results.append((t,str(path.relative_to(HERE)).replace('\\','/')))
    cap.release()
    return results


def main():
    frames=rows('bp_front_v31_frame_identity.csv');trajectories=rows('bp_front_v31_selected_trajectories.csv');edges=rows('bp_front_v31_handoff_decisions.csv')
    ident=defaultdict(lambda:defaultdict(list));by_video=defaultdict(dict)
    for r in frames:
        if r['video_id'] in ('bp_518','bp_520'):ident[r['video_id']][int(r['frame_index'])].append(r)
    for r in trajectories:
        by_video[r['video_id']][int(r['frame_index'])]=r
    detail=[];summary=[];preview_index=[]
    for video in ('bp_518','bp_520'):
        data=by_video[video];groups=ident[video]
        omitted=[t for t,r in data.items() if r['v3_baseline_instance_index']!='-1' and r['selected_global_instance_index']=='-1']
        for t in omitted:
            r=data[t];chosen=next((p for p in groups[t] if p['global_instance_index']==r['v3_baseline_instance_index']),None)
            for p in groups[t]:
                detail.append(dict(video_id=video,frame_index=t,legacy_v3_index=r['v3_baseline_instance_index'],
                                   candidate_index=p['global_instance_index'],original_extractor_track_id=p['extractor_track_id'],
                                   v2_track_id=p['v2_track_id'],selected_v3=p['global_instance_index']==r['v3_baseline_instance_index'],
                                   old_identity=p['state'],old_reason=p['reason'],mask_confidence=p['mask_confidence'],
                                   left_wrist_confidence=p['left_wrist_confidence'],right_wrist_confidence=p['right_wrist_confidence'],
                                   left_mask_distance=p['left_wrist_mask_distance'],right_mask_distance=p['right_wrist_mask_distance'],
                                   left_axis_distance=p['left_wrist_axis_distance'],right_axis_distance=p['right_wrist_axis_distance'],
                                   left_extension=p['left_wrist_extension'],right_extension=p['right_wrist_extension'],
                                   left_signed_y=p['left_signed_y'],right_signed_y=p['right_signed_y'],mask_area_px=p['mask_area_px'],
                                   mask_width_px=p['mask_width_px'],bilateral_compatibility=p['bilateral_compatibility'],
                                   context_frames=p['context_frames'],relative_motion_pairs=p['relative_motion_pairs'],
                                   relative_motion_median_px=p['relative_motion_median_px'],old_trajectory_status=r['status'],
                                   old_trajectory_reason=r['reason'],competitor_count=len(groups[t])-1))
        seq=[]
        for t in omitted:
            if not seq or t==seq[-1][-1]+1:seq[-1].append(t) if seq else seq.append([t])
            else:seq.append([t])
        reasons=Counter(next((p['reason'] for p in groups[t] if p['global_instance_index']==data[t]['v3_baseline_instance_index']), 'V3_MISSING') for t in omitted)
        summary.append(dict(video_id=video,omitted_frames=len(omitted),runs=[(s[0],s[-1],len(s)) for s in seq],
                            v3_identity_reasons=dict(reasons),bar_candidates_per_frame=dict(Counter(len(groups[t]) for t in omitted))))
        samples={0,len(data)//2,len(data)-1}
        for run in seq:
            samples.update([run[0],run[len(run)//2],run[-1]])
            samples.update([max(0,run[0]-1),min(len(data)-1,run[-1]+1)])
        for t in range(1,len(data)):
            if len(groups[t])!=len(groups[t-1]) or data[t]['status']!=data[t-1]['status']:
                samples.update([max(0,t-1),t,min(len(data)-1,t+1)])
        # muestreo deliberado de transiciones; incluye los extremos y mitad de cada corrida
        if len(samples)>36:
            required={0,len(data)//2,len(data)-1}
            for run in seq:required.update([run[0],run[len(run)//2],run[-1]])
            remaining=sorted(samples-required)
            step=max(1,len(remaining)//max(1,36-len(required)))
            samples=required|set(remaining[::step][:max(0,36-len(required))])
        pose=np.load(ROOT/'keypoints/bp/front'/(video+'.npy'),allow_pickle=False)
        with np.load(ROOT/'segmentation/bp/front/tracking_v2/raw'/(video+'_segmentation.npz'),allow_pickle=False) as raw:
            for t,path in preview(video,samples,pose,raw,groups,data):preview_index.append(dict(video_id=video,frame_index=t,preview=path))
    write('diagnosis_omitted_candidates.csv',detail)
    write('diagnosis_preview_index.csv',preview_index)
    (HERE/'diagnosis_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    # Las revisiones visuales originales de handoffs sirven para auditar pérdida, nunca para elegir máscara.
    original={}
    with (ROOT/'segmentation/bp/front/tracking_v3/validation_76/manifests/bp_front_tracking_v3_validation_76_handoffs.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['selected_by_v3']=='1':original[(r['video_id'],r['source_track'],r['target_track'],r['transition_frame'])]=r
    by_key={(r['video_id'],r['source_track'],r['target_track'],r['transition_frame']):r for r in edges if r['selected_by_v3']=='True'}
    handoffs=[]
    for k,review in original.items():
        if review['visual_review_status']!='VISUALLY_CONFIRMED':continue
        video,source,target,t=k;frame=int(t);pair=[by_video[video].get(j) for j in (frame-1,frame)]
        if all(r and r['selected_global_instance_index']!='-1' for r in pair):continue
        e=by_key.get(k)
        b=[p for p in frames if p['video_id']==video and int(p['frame_index']) in (frame-1,frame)]
        handoffs.append(dict(video_id=video,source_track=source,target_track=target,transition_frame=frame,
                             original_visual_review_status=review['visual_review_status'],pose_handoff_decision=e['pose_decision'] if e else 'MISSING',
                             v3_geometric_score=review['total_edge_score'],before_status=pair[0]['status'] if pair[0] else 'NO_FRAME',
                             after_status=pair[1]['status'] if pair[1] else 'NO_FRAME',before_reason=pair[0]['reason'] if pair[0] else '',
                             after_reason=pair[1]['reason'] if pair[1] else '',before_bar_count=sum(int(p['frame_index'])==frame-1 for p in b),
                             after_bar_count=sum(int(p['frame_index'])==frame for p in b),
                             source_subtracklet=e['source_subtracklet'] if e else '',target_subtracklet=e['target_subtracklet'] if e else '',
                             source_quality=e['source_quality'] if e else '',target_quality=e['target_quality'] if e else ''))
    assert len(handoffs)==39
    write('diagnosis_39_correct_boundary_abstentions.csv',handoffs)
    print('diagnosis',[(s['video_id'],s['omitted_frames'],s['v3_identity_reasons']) for s in summary],
          'previews',len(preview_index),'correct boundary losses',len(handoffs))


if __name__=='__main__':main()
