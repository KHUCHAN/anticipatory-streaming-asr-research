import json,random,time
from pathlib import Path
import torch
from common import ROOT,OUT,HERE,config,sha,write,rows_for,datasets
from model import Hybrid

def main():
    cfg=config();training={};torch.set_num_threads(1)
    for v in cfg['variants']:
        logs=[json.loads(s) for s in (OUT/v/'log.jsonl').read_text().splitlines()]
        ev=[r for r in logs if r.get('event')=='eval']
        assert [r['epoch'] for r in ev]==list(range(cfg['epochs']+1))
        assert any(r.get('event')=='done' for r in logs)
        ck=torch.load(OUT/v/'best.pt',map_location='cpu',weights_only=False)
        assert ck['epoch']+1 in range(cfg['epochs']+1)
        assert abs(ck['dev']['nll']-min(r['dev']['nll'] for r in ev))<1e-12
        training[v]=dict(completed_epochs=cfg['epochs'],selected_epoch=ck['epoch']+1,parameters=next(r for r in logs if r.get('event')=='start')['parameters'],sha256=sha(OUT/v/'best.pt'))
    for size in ['small','large']:
        a=[json.loads(s) for s in (OUT/(size+'_det')/'log.jsonl').read_text().splitlines()]
        b=[json.loads(s) for s in (OUT/(size+'_fgn')/'log.jsonl').read_text().splitlines()]
        assert next(r for r in a if r['event']=='start')['initial_sha256']==next(r for r in b if r['event']=='start')['initial_sha256']
    learned=json.loads((OUT/'gates.json').read_text());dev=json.loads((OUT/'events_dev_provenance.json').read_text());test=json.loads((OUT/'events_test_provenance.json').read_text())
    assert dev==test==learned['collection_provenance']
    assert all(sha(OUT/v/'best.pt')==s for v,s in test['model_sha256'].items())
    assert all(sha(HERE/n)==s for n,s in test['source_sha256'].items())
    assert all(sha(ROOT/n)==s for n,s in test['dependency_sha256'].items())
    assert set(learned['fit_ids']).isdisjoint(learned['calibration_ids'])
    ids={r['utterance_id'] for r in rows_for('test',cfg)};present={p.stem for p in (OUT/'events_test').glob('*.json') if p.name!='complete.json'}
    assert ids==present and len(ids)==5559
    differences=0
    for uid in ids:
        row=json.loads((OUT/'events_test'/(uid+'.json')).read_text());old=json.loads((ROOT/'runs/zipformer_ours/events_test'/(uid+'.json')).read_text())
        differences+=int(any(row[k]!=old[k] for k in ['native','hypothesis','errors','reference','correct_native_positions']))
        for v in cfg['variants']:assert len(row['events'][v])==len(old['events']['band2'])
    assert differences==0
    results=json.loads((OUT/'results.json').read_text());assert len(results['old_control_reproduction'])==12
    assert all(v=='exactly reproduced' for v in results['old_control_reproduction'].values())
    assert json.loads((OUT/'checks.json').read_text())['passed']
    report=ROOT/'docs/band_fgn_hybrid_2026-11-12.md';assert report.exists()
    # The initial large model has identity new blocks. Validate the exact text
    # receptive-field crop again after those blocks have actually been trained.
    ds,_,_=datasets(cfg);batch=ds.collate(ds.batches(cfg)[450][:2],random.Random(19),'cpu')
    m=Hybrid('large_fgn',initialize=False).eval()
    m.load_state_dict(torch.load(OUT/'large_fgn/last.pt',map_location='cpu',weights_only=False)['model'])
    states,lengths,cuts,tokens,token_lengths,targets=batch
    with torch.inference_mode():
        logits,_,pred=m(batch,2)
        t=max(1,int(cuts.max())*8);arrived=states.flatten(1,2)[:,:t]
        allow=(torch.arange(t)[None,:]<cuts[:,None]*8)[:,None]
        future=(pred*m.state_std+m.state_mean).flatten(2,3)
        original=m.speculative_ar(tokens,token_lengths-1,arrived,allow,future)
        crop_difference=float((logits-original).abs().max());assert crop_difference<3e-5
    result=dict(passed=True,training=training,matched_initial_weights_det_vs_noise=True,test_utterances=len(ids),native_output_differences=differences,old_control_points_exactly_reproduced=12,all_source_model_dependency_hashes_match=True,completed_unix=time.time())
    result['trained_large_full_prefix_vs_exact_context_crop_max_difference']=crop_difference
    write(OUT/'verification.json',result);write(HERE/'artifacts/verification.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
