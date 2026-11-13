"""Causal streaming-state equivalence, bounded tensors and inference cost."""
import json,time
import numpy as np
import torch
from data import ROOT,OUT
from model import StateFGN,FrozenRNNT
from runtime import Runtime


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    ck=torch.load(OUT/'state_s37/best.pt',map_location='cpu',weights_only=False)
    model=StateFGN(ck['cfg'],'state_s37');model.load_state_dict(ck['model']);model.eval()
    bridge=FrozenRNNT(OUT/'rnnt_bridge.npz').eval()
    stats=json.loads((OUT/'cache/stats.json').read_text());mean=torch.tensor(stats['mean']);std=torch.tensor(stats['std'])
    row=json.loads((ROOT/ck['cfg']['gate_protocol']/'model_selection.jsonl').read_text().splitlines()[0])
    with np.load(ROOT/f"data/processed/zipformer_forecasts/{row['split']}/{row['utterance_id']}.npz") as f:
        states=torch.from_numpy(f['enc'].astype(np.float32)).view(-1,8,512)
    normalized=(states-mean)/std;offline=model.observed_memory(normalized[None])
    rt=Runtime([model],bridge,mean,std,8,2,['']*500)
    maxdiff=0.;timings=[];sizes=[];g=torch.Generator().manual_seed(413)
    for i,x in enumerate(states):
        r=rt.advance(x,[-1,0],0.,.32,0,0,g)
        maxdiff=max(maxdiff,float((rt.memories[0]-offline[:,i]).abs().max()))
        timings.append(r['seconds']);sizes.append(r['confirmed_state_bytes'])
    assert maxdiff<2e-5,maxdiff
    assert len(set(sizes))==1
    confirmed_before=[x.clone() for x in [rt.previous,rt.current,*rt.memories]]
    rt.sample_states(g,steps=8,members=16)
    assert all(torch.equal(a,b) for a,b in zip(confirmed_before,[rt.previous,rt.current,*rt.memories]))
    # About ten minutes of actual observations; no stored history in the runtime.
    for i in range(1875):rt.observe(states[i%len(states)])
    assert rt.previous.numel()*4+rt.current.numel()*4+sum(x.numel()*4 for x in rt.memories)==sizes[-1]
    result=dict(status='pass',checkpoint_epoch=ck['epoch']+1,
                streaming_vs_offline_memory_max_difference=maxdiff,forecast_branches_do_not_change_confirmed_state=True,
                observed_chunks_for_bounded_state_check=1875,confirmed_tensor_bytes=sizes[-1],
                single_model_8_member_2_step_mean_ms=1000*np.mean(timings[2:]),
                single_model_8_member_2_step_p95_ms=1000*np.percentile(timings[2:],95),
                timing_note='Warm CPU single-thread benchmark during MPS training; not a live full-ASR end-to-end latency test.')
    (OUT/'runtime_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
