"""Corrobora extremos, sin convertir POSE_COMPATIBLE automáticamente en identidad segura."""
from statistics import median

from v31_revision_identity import number


def endpoint_plausible(row,baseline):
    if row is None or row['evidence_quality']!='BOTH_RELIABLE':return False
    scale=number(row,'wrist_span_px')
    if scale is None or scale<=baseline['min_wrist_span_px']:return False
    for side in ('left','right'):
        q=number(row,side+'_signed_y');axis=number(row,side+'_wrist_axis_distance')
        if q is None or axis is None or not -baseline['expected_bar_above_limit_px']<=q<0 or axis>=baseline['max_axis_ratio']*scale:return False
    if row['state']=='INCOMPATIBLE_WITH_HELD_BAR':return False
    return True


def contextual_endpoint(candidate,t,segment,by_track,by_frame,track_summary,base,rev):
    if not endpoint_plausible(candidate,base):return False,'endpoint lacks bilateral signed/axis evidence'
    if track_summary.get(candidate['v2_track_id'],0)>0:return False,'hybrid source or target track'
    # Si existen BAR competidoras todavía plausibles sin resolver, abstenerse.
    contemporaries=by_frame.get(t,[])
    if any(x is not candidate and x['state']!='INCOMPATIBLE_WITH_HELD_BAR' for x in contemporaries):
        return False,'unresolved competing BAR at boundary'
    window=base['context_frames']
    candidates=[r for r in by_track.get(candidate['v2_track_id'],[]) if
                r['subtracklet_id']==segment and 0<=abs(int(r['frame_index'])-t)<=window and r['state']=='SUPPORTED_HELD_BAR' and endpoint_plausible(r,base)]
    # Los otros apoyos deben corresponder al mismo subtracklet e instante local; no
    # interpolar gaps ni usar un track V2 como identidad verdadera.
    samples=sorted({int(r['frame_index']) for r in candidates})
    runs=[]
    for x in samples:
        if not runs or x!=runs[-1][-1]+1:runs.append([x])
        else:runs[-1].append(x)
    good=[run for run in runs if len(run)>=rev['handoff_context_min_observations'] and min(abs(x-t) for x in run)<=base['context_frames']]
    if not good:return False,'fewer than 3 consecutive supported same-segment context frames'
    # La relación q con muñecas no salta aunque el centro cambie por recorte.
    anchor=min(good,key=lambda run:min(abs(x-t) for x in run))
    source=[r for r in candidates if int(r['frame_index']) in anchor]
    span=min(number(candidate,'wrist_span_px'),*(number(r,'wrist_span_px') for r in source))
    threshold=max(base['compatible_jump_px'],base['compatible_jump_ratio']*span)
    if any(abs(number(candidate,side+'_signed_y')-median(number(r,side+'_signed_y') for r in source))>=threshold for side in ('left','right')):
        return False,'endpoint-relative wrist offset disagrees with temporal context'
    return True,'same subtracklet has >=3 consecutive supported observations; individual endpoint q/axis plausible; no unresolved competitor'


class PoseGuidedHandoffContext:
    def __init__(self,base,revision):self.base,self.rev=base,revision

    def corroborate(self,video,edges,frame_rows,by_track,by_frame,track_summary,old_trajectory):
        promotions={};notes=[]
        for edge in edges:
            if edge['selected_by_v3']!='True' or edge['pose_decision']!='POSE_COMPATIBLE' or edge['v3_geometry_accepted']!='True':continue
            t=int(edge['transition_frame'])
            for frame,segment in [(t-1,edge['source_subtracklet']),(t,edge['target_subtracklet'])]:
                old=old_trajectory.get(frame)
                if old is None or old['selected_global_instance_index']!='-1':continue
                idx=old['v3_baseline_instance_index']
                candidate=next((r for r in by_frame.get(frame,[]) if r['global_instance_index']==idx and r['subtracklet_id']==segment),None)
                ok,reason=contextual_endpoint(candidate,frame,segment,by_track,by_frame,track_summary,self.base,self.rev)
                notes.append(dict(video_id=video,transition_frame=t,frame_index=frame,edge_id=edge['edge_id'],
                                  candidate_index=idx,eligible=ok,reason=reason,component='HANDOFF_CONTEXT'))
                if ok:promotions[frame]=candidate
        return promotions,notes
