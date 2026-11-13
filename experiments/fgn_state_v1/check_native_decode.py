"""Oracle audit only: decoding real future states should recover cached tokens.

Future states are passed to the decoder in this explicit upper-bound check,
never to a forecaster. This audits timing/context and FP16 cache quantization.
"""
import json,random
import numpy as np
import torch
from data import ROOT,OUT,selections
from model import FrozenRNNT
from runtime import decode_states


@torch.inference_mode()
def main():
    torch.set_num_threads(1);cfg=json.loads((ROOT/'experiments/fgn_state_v1/config.json').read_text())
    _,rows=selections(cfg);bridge=FrozenRNNT(OUT/'rnnt_bridge.npz');rng=random.Random(415)
    mismatches=[];total=0;tokens=0
    for row in rows:
        with np.load(ROOT/f"data/processed/zipformer_forecasts/{row['split']}/{row['utterance_id']}.npz") as v:
            x=torch.from_numpy(v['enc'].astype(np.float32)).view(-1,8,512);ids=v['ids'].tolist();cuts=v['cuts'].tolist()
        if len(x)<6:continue
        cut=rng.randint(0,len(x)-5);n=cuts[cut];context=ids[n-2:n] if n>=2 else ([0,ids[0]] if n else [-1,0])
        decoded=decode_states(x[cut+1:cut+5][None],bridge,context)[0];expected=ids[n:cuts[cut+4]]
        total+=1;tokens+=len(expected)
        if decoded!=expected:mismatches.append(dict(id=row['utterance_id'],expected=expected,decoded=decoded))
    result=dict(utterances=total,future_native_tokens=tokens,exact_sequences=total-len(mismatches),mismatches=mismatches,
                interpretation='Oracle audit only, not a prediction score. Cache encoder states are FP16; native tokens were decoded from original FP32 states. Rare borderline argmax changes are possible.')
    assert len(mismatches)<=max(2,total*.01),result
    (OUT/'native_decode_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
