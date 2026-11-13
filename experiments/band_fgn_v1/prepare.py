import json,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from common import ROOT,OUT,config,sha,write

def one(row):
    with np.load(ROOT/f"data/processed/zipformer_forecasts/{row['split']}/{row['utterance_id']}.npz") as x:
        return row['utterance_id'],dict(gold=x['gold'].astype(int).tolist(),gold_frames=x['gold_frames'].astype(int).tolist())

def main():
    cfg=config();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'gold_metadata.json').exists():return
    rows=json.loads((ROOT/'runs/fgn_state_v1/cache/index.json').read_text())
    rows=[r for r in rows if r['split'] in ['train-clean-100','dev-clean','dev-other']]
    t=time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:meta=dict(ex.map(one,rows))
    write(OUT/'gold_metadata.json',meta)
    write(OUT/'initial_plan.json',dict(config=cfg,created_unix=time.time(),metadata_utterances=len(meta),seconds=time.time()-t,
      ar_initial_sha256=sha(ROOT/'runs/zipformer_ours/band2/best.pt'),state_initial_sha256=sha(ROOT/'runs/fgn_state_v1/state_s37/best.pt'),
      hypotheses=['band2 AR plus forecast continuous audio beats equal-depth AR continuation',
      'functional noise improves held-out NLL/gain over same-size deterministic hybrid',
      'depth expansion has less held-out degradation with functional noise'],
      warning='One seed and eight continuation epochs. If overfitting is not observed, do not claim that FGN prevented it. Noise is in the future-audio branch, not a generic dropout replacement in AR.'))
    print(json.dumps(dict(metadata=len(meta),seconds=time.time()-t)),flush=True)
if __name__=='__main__':main()
