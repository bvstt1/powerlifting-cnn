"""Detección de switches intratrack con subtracklets virtuales trazables."""
from statistics import median

import numpy as np


def q(row,side):
    return row[side+'_signed_y']


class PoseGuidedTrackConsistencyValidator:
    def __init__(self,config):
        self.p=config

    def evaluate(self,video,by_track):
        tracks=[]; segments=[]; by_segment={}; cut_index={}
        for track,rows in sorted(by_track.items()):
            rows=sorted(rows,key=lambda r:(r['frame_index'],r['global_instance_index']))
            cuts=[]
            n=self.p['near_frames']
            for j in range(n,len(rows)-n+1):
                before=rows[j-n:j]; after=rows[j:j+n]
                if not all(z['wrist_span_px'] is not None for z in before+after): continue
                if len({z['frame_index'] for z in before+after})<2*n:continue
                # El intervalo debe ser observado en el mismo track. Huecos no son continuidad inferida.
                if any(b['frame_index']!=a['frame_index']+1 for a,b in zip(before+after,(before+after)[1:])):continue
                span=min(z['wrist_span_px'] for z in before+after)
                jumps=[median(q(z,side) for z in after)-median(q(z,side) for z in before) for side in ('left','right')]
                widths=[float(median(z['mask_width_px'] for z in subset)) for subset in (before,after)]
                width_ratio=min(widths)/max(widths) if max(widths)>0 else 0
                threshold=max(self.p['different_jump_px'],self.p['different_jump_ratio']*span)
                if (all(abs(v)>threshold for v in jumps) and np.sign(jumps[0])==np.sign(jumps[1])
                        and width_ratio>self.p['width_ratio_min']):
                    cuts.append(dict(boundary_frame=after[0]['frame_index'],j=j,left_jump_px=jumps[0],right_jump_px=jumps[1],width_ratio=width_ratio,
                                     evidence_frames=[z['frame_index'] for z in before+after],reason='persistent bilateral signed jump within original V2 track'))
            # Evitar múltiples cortes derivados del mismo switch: retener el mayor soporte local.
            chosen=[]
            for candidate in sorted(cuts,key=lambda z:-min(abs(z['left_jump_px']),abs(z['right_jump_px']))):
                if all(abs(candidate['j']-c['j'])>n for c in chosen):chosen.append(candidate)
            chosen.sort(key=lambda x:x['j'])
            bounds=[0]+[c['j'] for c in chosen]+[len(rows)]
            for ordinal,(start,end) in enumerate(zip(bounds[:-1],bounds[1:])):
                portion=rows[start:end]; sid='%d:%d'%(track,ordinal)
                state='SUSPECT_CONTAMINATED' if chosen else ('CONSISTENT' if sum(z['wrist_span_px'] is not None for z in portion)>=3 else 'UNVERIFIABLE')
                info=dict(video_id=video,original_v2_track_id=track,subtracklet_id=sid,segment_ordinal=ordinal,
                          first_frame=portion[0]['frame_index'],last_frame=portion[-1]['frame_index'],frames=len(portion),
                          global_instance_indices=[z['global_instance_index'] for z in portion],
                          original_extractor_track_ids=sorted({z['extractor_track_id'] for z in portion}),
                          consistency_state=state,split_reason=chosen[ordinal-1]['reason'] if ordinal else ('no supported split' if not chosen else 'initial subtracklet'),
                          evidence_frames=chosen[ordinal-1]['evidence_frames'] if ordinal else [],
                          bilateral_reliable_frames=sum(z['wrist_span_px'] is not None for z in portion),
                          supported_frames=sum(z['state']=='SUPPORTED_HELD_BAR' for z in portion),
                          incompatible_frames=sum(z['state']=='INCOMPATIBLE_WITH_HELD_BAR' for z in portion),
                          status='SUBTRACKLET_PROPOSED' if chosen else state)
                segments.append(info);by_segment[sid]=portion
                for z in portion:cut_index[z['global_instance_index']]=sid
            tracks.append(dict(video_id=video,track_id=track,consistency_state='SUSPECT_CONTAMINATED' if chosen else ('CONSISTENT' if len(rows)>=3 else 'UNVERIFIABLE'),
                               switch_count=len(chosen),proposed_cut_frames=[c['boundary_frame'] for c in chosen],
                               switch_evidence=chosen,frames=len(rows),segments=len(bounds)-1))
        return tracks,segments,by_segment,cut_index
