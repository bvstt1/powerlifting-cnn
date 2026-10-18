"""Verificaciones finales independientes sobre artefactos; no ejecuta modelos."""
import numpy as np
import pandas as pd
from .run import ROOT, outdir, readj, savej, verify, sha, paths, rows_csv, metrics, now


def main():
    for view,n,npos in (("left",91,44),("right",93,41)):
        protocol,source=verify(view);out=outdir(view);p=paths(view)
        before=readj(out/"protected_inputs_before.json")["sha256"]
        after=readj(out/"protected_inputs_after.json")["sha256"]
        assert before==after
        audited=rows_csv(p["source"]/"source_audit.csv")
        for row in audited:
            vid=row["video_id"]
            for path,key in ((ROOT/"dataset/bp"/view/f"{vid}.mp4","video_sha256"),
                             (p["keypoints"]/f"{vid}.npy","pose_sha256"),
                             (p["source"].parent/"raw"/f"{vid}_segmentation.npz","raw_sha256")):
                assert before[str(path.relative_to(ROOT))]==row[key],str(path)
        predictions=pd.read_csv(out/"test_predictions.csv")
        scores=readj(out/"test_metrics.json")
        arrays={}
        for m in protocol["models"]:
            rows=predictions[predictions.model==m]
            assert list(rows.video_id)==protocol["test_ids"] and len(rows)==n
            assert rows.label.sum()==npos and np.array_equal(rows.predicted,rows.probability_valid>=.5)
            calculated=metrics(rows.label,rows.probability_valid)
            saved=next(s for s in scores if s["model"]==m)
            assert all(np.isclose(calculated[k],saved[k],rtol=0,atol=1e-12) for k in calculated)
            arrays[m]=rows.probability_valid.to_numpy()
        variant="full" if view=="left" else "sin_relacion"
        assert np.allclose(arrays[f"late_{variant}"],.5*arrays["pose"]+.5*arrays[f"seg_{variant}"],atol=1e-15,rtol=0)
        assert readj(out/"final_status.json")["test_inference_runs"]==1
        savej(out/"final_output_verification.json",dict(status="PASS",at=now(),
            recomputed_metrics_match=True,late_formula_matches=True,original_video_pose_raw_hashes_match=len(audited),
            all_protected_inputs_unchanged=True,protocol_unchanged=True,test_inference_runs=1))
        artifacts={str(file.relative_to(ROOT)):sha(file) for file in out.iterdir() if file.is_file() and file.name!="output_manifest.json"}
        artifacts["training/Reporte_Final_BP_LEFT_RIGHT_Comparativo.md"]=sha(ROOT/"training/Reporte_Final_BP_LEFT_RIGHT_Comparativo.md")
        savej(out/"output_manifest.json",dict(at=now(),sha256=artifacts))
        print(view,"FINAL_OUTPUT_VERIFICATION_PASS",len(artifacts),"artifacts",flush=True)


if __name__=="__main__":main()
