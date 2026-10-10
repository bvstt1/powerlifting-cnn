"""Identidad BAR por instancia; no conoce V2 main ni revisiones visuales."""
from collections import defaultdict

import numpy as np

from v31_common import geometry, wrist_measure


class PoseGuidedBarIdentityValidator:
    def __init__(self, config):
        self.p=config

    def evaluate(self,video,pose,raw):
        rows=[]; by_track=defaultdict(list); by_frame=defaultdict(list)
        for index in range(len(raw['instance_frame_index'])):
            if raw['instance_class_name'][index]!='bar':
                continue
            frame=int(raw['instance_frame_index'][index]); track=int(raw['instance_track_id'][index]); geom=geometry(raw,index)
            wrists,scale=wrist_measure(raw,pose,index,geom,frame,self.p['confidence_min'],self.p['min_wrist_span_px'])
            valid=all(z['valid'] for z in wrists) and scale is not None
            signed=[z['signed_y'] for z in wrists]
            axis=[z['axis_distance'] for z in wrists]
            row=dict(video_id=video,frame_index=frame,global_instance_index=index,v2_track_id=track,
                     extractor_track_id=int(raw['instance_original_track_id'][index]) if 'instance_original_track_id' in raw else track,
                     centroid_x=float(geom['center'][0]),centroid_y=float(geom['center'][1]),mask_area_px=geom['area'],mask_width_px=geom['width'],
                     mask_confidence=float(raw['instance_confidence'][index]),wrist_span_px=scale,
                     left_wrist_confidence=wrists[0]['confidence'],right_wrist_confidence=wrists[1]['confidence'],
                     left_wrist_valid=wrists[0]['valid'],right_wrist_valid=wrists[1]['valid'],
                     left_wrist_mask_distance=wrists[0]['mask_distance'],right_wrist_mask_distance=wrists[1]['mask_distance'],
                     left_wrist_axis_distance=axis[0],right_wrist_axis_distance=axis[1],
                     left_wrist_extension=wrists[0]['extension'],right_wrist_extension=wrists[1]['extension'],
                     left_signed_y=signed[0],right_signed_y=signed[1],
                     bilateral_compatibility=max(z['mask_distance'] for z in wrists)/scale if valid else None,
                     evidence_quality='BOTH_RELIABLE' if valid else 'ONE_RELIABLE' if any(z['valid'] for z in wrists) else 'NO_WRISTS',
                     state='INDETERMINATE',reason='single frame cannot establish physical identity',
                     relative_motion_pairs=0,relative_motion_median_px=None,context_frames=0)
            rows.append(row);by_track[track].append(row);by_frame[frame].append(row)
        for track,track_rows in by_track.items():
            track_rows.sort(key=lambda r:(r['frame_index'],r['global_instance_index']))
            for j,r in enumerate(track_rows):
                neighbors=[track_rows[k] for k in range(max(0,j-2),min(len(track_rows),j+3)) if abs(track_rows[k]['frame_index']-r['frame_index'])<=2 and track_rows[k]['wrist_span_px'] is not None]
                r['context_frames']=len(neighbors)
                motion=[]; diff=[]
                for a,b in zip(neighbors[:-1],neighbors[1:]):
                    if b['frame_index']!=a['frame_index']+1: continue
                    da=np.array([b['centroid_x']-a['centroid_x'],b['centroid_y']-a['centroid_y']])
                    # c_y - mean(w_y) diferencia: desplazamiento relativo vertical aun si BAR se mueve realmente.
                    relative=(b['left_signed_y']+b['right_signed_y']-a['left_signed_y']-a['right_signed_y'])/2
                    diff.append(abs(relative)); motion.append(abs(relative))
                r['relative_motion_pairs']=len(motion);r['relative_motion_median_px']=float(np.median(motion)) if motion else None
                if r['wrist_span_px'] is None or len(neighbors)<self.p['identity_min_context_frames']:
                    r['reason']='incomplete bilateral/temporal evidence'; continue
                ref=r['wrist_span_px']
                values=[(n['left_signed_y']+n['right_signed_y'])/2 for n in neighbors]
                stable=max(values)-min(values)<=self.p['identity_context_range_px']
                relative_stable=not motion or np.median(motion)<=self.p['max_relative_step_px']
                if (all(v>max(self.p['different_jump_px'],self.p['different_jump_ratio']*ref) for v in (r['left_signed_y'],r['right_signed_y']))
                        and sum(n['left_signed_y']>max(self.p['different_jump_px'],self.p['different_jump_ratio']*ref) and n['right_signed_y']>max(self.p['different_jump_px'],self.p['different_jump_ratio']*ref) for n in neighbors)>=self.p['identity_min_context_frames']):
                    r['state']='INCOMPATIBLE_WITH_HELD_BAR';r['reason']='persistent BAR below both wrists with reliable bilateral evidence'
                elif (all(-self.p['expected_bar_above_limit_px'] <= v < 0 for v in (r['left_signed_y'],r['right_signed_y']))
                      and all(v<self.p['max_axis_ratio']*ref for v in (r['left_wrist_axis_distance'],r['right_wrist_axis_distance']))
                      and stable and relative_stable):
                    r['state']='SUPPORTED_HELD_BAR';r['reason']='bilateral signed offset and temporal coherence'
                else:
                    r['reason']='offset/axis/motion ambiguous or partially cropped'
        # Segundo paso: la competencia simultánea puede invalidar una afirmación positiva ambigua.
        for frame, candidates in by_frame.items():
            supported=[r for r in candidates if r['state']=='SUPPORTED_HELD_BAR']
            if len(supported)>1:
                for r in supported:
                    r['state']='INDETERMINATE';r['reason']='multiple supported BAR candidates without calibrated main-bar margin'
        return rows,by_frame,by_track
