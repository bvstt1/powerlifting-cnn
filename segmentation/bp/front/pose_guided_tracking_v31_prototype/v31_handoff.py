"""Handoffs V3 existentes y cortes virtuales; puntuación V3 recomputada si cambian límites."""
import sys
from pathlib import Path
from statistics import median

import numpy as np

from v31_common import ROOT, full_mask, geometry

sys.path.insert(0,str(ROOT/'extract_segmentation/bp'))
from bar_tracking_v3_utils import build_tracklets, evaluate_candidate, DEFAULT_PARAMETERS


def recalc(source_rows,target_rows,raw):
    """Recalcula edge geométrico usando límites lógicos; nunca altera módulo V3."""
    detections=[]
    for virtual_id,portion in enumerate((source_rows,target_rows)):
        for row in portion:
            idx=row['global_instance_index'];g=geometry(raw,idx)
            detections.append(dict(global_index=idx,frame_index=row['frame_index'],class_name='bar',
                v2_track_id=virtual_id,extractor_track_id=row['extractor_track_id'],
                confidence=float(raw['instance_confidence'][idx]),bbox_xyxy=raw['instance_bbox_xyxy_px'][idx].copy(),
                geometry=dict(centroid_x=float(g['center'][0]),centroid_y=float(g['center'][1]),
                              area=g['area'],orientation_angle_deg=g['angle'],major_axis_length=g['major_axis'],minor_axis_length=g['minor_axis']),
                mask=full_mask(raw,idx)))
    frames=len(raw['frame_index']);by_frame=[[] for _ in range(frames)]
    for detection in detections:by_frame[detection['frame_index']].append(detection)
    tracks=build_tracklets(by_frame,frames,raw['frame_image_shape_hw'][0],0,DEFAULT_PARAMETERS)
    raw_frames={int(raw['instance_frame_index'][i]) for i in range(len(raw['instance_frame_index'])) if raw['instance_class_name'][i]=='bar'}
    result=evaluate_candidate(tracks[0],tracks[1],raw_frames,raw['frame_image_shape_hw'][0],DEFAULT_PARAMETERS)
    return result


class PoseGuidedHandoffValidator:
    def __init__(self,config):self.p=config

    def evaluate(self,video,source_rows,target_rows,transition,score=None,selected_v3=False,edge_id='',recomputed=False,split_supported=False):
        near=self.p['near_frames'];context=self.p['context_frames']
        before=[r for r in source_rows if transition-context<=r['frame_index']<transition]
        after=[r for r in target_rows if transition<=r['frame_index']<transition+context]
        pre,post=before[-near:],after[:near]
        out=dict(video_id=video,edge_id=edge_id,source_track=source_rows[0]['v2_track_id'] if source_rows else None,
                 target_track=target_rows[0]['v2_track_id'] if target_rows else None,transition_frame=transition,
                 source_subtracklet=source_rows[0].get('subtracklet_id','') if source_rows else '',
                 target_subtracklet=target_rows[0].get('subtracklet_id','') if target_rows else '',
                 geometric_score=score,selected_by_v3=bool(selected_v3),recomputed_geometry=bool(recomputed),
                 original_v3_decision='SELECTED' if selected_v3 else 'NOT_SELECTED',
                 pose_decision='POSE_INSUFFICIENT_EVIDENCE',reason='insufficient bilateral before/after',
                 source_frames=len(before),target_frames=len(after),source_near_frames=len(pre),target_near_frames=len(post),
                 left_signed_jump_px=None,right_signed_jump_px=None,source_bilateral_px=None,target_bilateral_px=None,
                 source_quality='BOTH_RELIABLE' if sum(r['wrist_span_px'] is not None for r in before)>=self.p['min_bilateral_window'] else 'LOW',
                 target_quality='BOTH_RELIABLE' if sum(r['wrist_span_px'] is not None for r in after)>=self.p['min_bilateral_window'] else 'LOW',
                 split_supported=bool(split_supported))
        if len(pre)!=near or len(post)!=near or not all(r['wrist_span_px'] is not None for r in pre+post):
            return out
        s=min(r['wrist_span_px'] for r in pre+post)
        jumps=[median(r[side+'_signed_y'] for r in post)-median(r[side+'_signed_y'] for r in pre) for side in ('left','right')]
        bw=median(r['mask_width_px'] for r in pre);dw=median(r['mask_width_px'] for r in post)
        ratio=min(bw,dw)/max(bw,dw) if max(bw,dw)>0 else 0
        short_split=False
        if split_supported and (out['source_quality']=='LOW') != (out['target_quality']=='LOW'):
            short=pre if out['source_quality']=='LOW' else post
            long=post if short is pre else pre
            short_split=(len(short)==near and
                all(median(r[k+'_signed_y'] for r in short)>self.p['compatible_jump_px'] for k in ('left','right')) and
                all(median(r[k+'_signed_y'] for r in long)<0 for k in ('left','right')))
        if out['source_quality']!='BOTH_RELIABLE' or out['target_quality']!='BOTH_RELIABLE':
            if not short_split:
                return out
            out['reason']='short subtracklet independently bounded by verified intratrack split; 2 bilateral observations; experimental high-risk veto'
            out['evidence_quality']='SHORT_SPLIT_BILATERAL_EXPERIMENTAL'
        out.update(left_signed_jump_px=jumps[0],right_signed_jump_px=jumps[1],mask_width_ratio=ratio,
                   source_bilateral_px=median(max(r['left_wrist_mask_distance'],r['right_wrist_mask_distance']) for r in pre),
                   target_bilateral_px=median(max(r['left_wrist_mask_distance'],r['right_wrist_mask_distance']) for r in post))
        if min(map(abs,jumps))>max(self.p['different_jump_px'],self.p['different_jump_ratio']*s) and np.sign(jumps[0])==np.sign(jumps[1]) and ratio>self.p['width_ratio_min']:
            out['pose_decision']='POSE_INCOMPATIBLE'
            if not short_split:out['reason']='bilateral signed discontinuity with similar mask widths'
        elif (max(map(abs,jumps))<max(self.p['compatible_jump_px'],self.p['compatible_jump_ratio']*s)
              and max(r[k+'_wrist_axis_distance'] for r in pre+post for k in ('left','right')) < self.p['max_axis_ratio']*s
              and not any(r['state']=='INCOMPATIBLE_WITH_HELD_BAR' for r in pre+post)):
            out['pose_decision']='POSE_COMPATIBLE';out['reason']='stable bilateral signed relation and mask axis'
        else:
            out['reason']='motion/crop/axis disagreement; do not force association'
        return out
