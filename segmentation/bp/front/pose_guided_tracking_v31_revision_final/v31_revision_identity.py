"""Única corrección localizada: distinguir empate anatómico real de fragmento lateral."""
from collections import defaultdict
import math


def number(row,name):
    try:
        value=float(row[name]);return value if math.isfinite(value) else None
    except (TypeError,ValueError,KeyError):return None


def prior_supported(row):
    # El código original restablece a INDETERMINATE ambas candidatas soportadas.
    return row['state']=='SUPPORTED_HELD_BAR' or (
        row['state']=='INDETERMINATE' and row['reason']=='multiple supported BAR candidates without calibrated main-bar margin')


def bilateral_dominance(winner,other,params):
    span=number(winner,'wrist_span_px')
    other_span=number(other,'wrist_span_px')
    if not span or not other_span or span<=0 or other_span<=0 or not prior_supported(winner):return False,None
    if winner['evidence_quality']!='BOTH_RELIABLE' or other['evidence_quality']!='BOTH_RELIABLE':return False,None
    left,right='left','right'
    gap=params['dominance_margin_ratio']*min(span,other_span)
    distances=[];extensions=[]
    for side in (left,right):
        d=number(winner,side+'_wrist_mask_distance');od=number(other,side+'_wrist_mask_distance')
        ex=number(winner,side+'_wrist_extension');oe=number(other,side+'_wrist_extension')
        if None in (d,od,ex,oe):return False,None
        distances.append(od-d);extensions.append(oe-ex)
        if d>params['winner_max_mask_distance_ratio']*span or od-d<=gap or oe-ex<=gap:return False,None
    # Prior_supported ya exigió q bilateral, eje y consistencia temporal del propio track;
    # no se decide por cercanía sola ni por selección V2/V3 o confidence más alta.
    evidence=dict(left_distance_margin_px=distances[0],right_distance_margin_px=distances[1],
                  left_extension_margin_px=extensions[0],right_extension_margin_px=extensions[1],
                  margin_required_px=gap,span_px=span)
    return True,evidence


class PoseGuidedCompetitorResolver:
    def __init__(self,params):self.p=params

    def resolve(self,by_frame):
        proposed={};reasons={};runs=defaultdict(list)
        for frame,rows in sorted(by_frame.items()):
            if len(rows)<2:continue
            eligible=[r for r in rows if prior_supported(r)]
            if len(eligible)<2:continue
            winners=[]
            for r in eligible:
                observations=[bilateral_dominance(r,other,self.p) for other in rows if other is not r]
                if observations and all(ok for ok,_ in observations):
                    winners.append((r,[ev for ok,ev in observations]))
            if len(winners)==1:
                winner,evidences=winners[0]
                key=(winner['v2_track_id'],winner['extractor_track_id'])
                proposed[frame]=(winner,key,evidences)
                runs[key].append(frame)
        approved=set()
        for key,frames in runs.items():
            sequence=[]
            for frame in sorted(frames):
                if sequence and frame!=sequence[-1]+1:
                    if len(sequence)>=self.p['dominance_min_consecutive_frames']:approved.update(sequence)
                    sequence=[]
                sequence.append(frame)
            if len(sequence)>=self.p['dominance_min_consecutive_frames']:approved.update(sequence)
        for frame,(winner,key,ev) in proposed.items():
            if frame not in approved:continue
            winner['state']='SUPPORTED_HELD_BAR'
            winner['reason']='REVISION_PERSISTENT_BILATERAL_MASK_AND_EXTENSION_DOMINANCE'
            reasons[frame]=dict(video_id=winner['video_id'],frame_index=frame,winner_index=winner['global_instance_index'],
                                v2_track_id=winner['v2_track_id'],original_extractor_track_id=winner['extractor_track_id'],
                                competitor_indices=[r['global_instance_index'] for r in by_frame[frame] if r is not winner],
                                evidence=ev,consecutive_support=self.p['dominance_min_consecutive_frames'],
                                baseline_v3_not_used_in_decision=True)
            for other in by_frame[frame]:
                if other is winner:continue
                # No declarar falsa barra física solamente por perder el ranking. Si no es
                # incompatible por evidencia original, continúa INDETERMINATE.
                if other['state']=='INDETERMINATE' and other['reason']=='multiple supported BAR candidates without calibrated main-bar margin':
                    other['reason']='REVISION_DOMINATED_BY_BILATERAL_MASK_AND_EXTENSION_EVIDENCE'
        return reasons
