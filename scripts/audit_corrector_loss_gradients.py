#!/usr/bin/env python3
"""Post-hoc loss-gradient audit; no optimization or inference evaluation."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/predict_correct_v1'))
from common import HERE,OUT,config,sha,state_module,write
from data import datasets
from model import Corrector
from objective import objective


def geometry(names,first,second):
    result={}
    for group in ['all','audio','head']:
        indices=[i for i,name in enumerate(names) if group=='all' or name.startswith(group+'.')]
        aa=sum(float(first[i].double().square().sum()) for i in indices)
        bb=sum(float(second[i].double().square().sum()) for i in indices)
        ab=sum(float((first[i].double()*second[i].double()).sum()) for i in indices)
        result[group]=dict(mae_gradient_norm=aa**.5,kl_gradient_norm=bb**.5,
            kl_to_mae_norm_ratio=(bb/max(aa,1e-30))**.5,
            cosine=ab/max((aa*bb)**.5,1e-30))
    return result


def main():
    torch.set_num_threads(1);cfg=config();_,dev,_=datasets(cfg)
    indices=[]
    for split in ['dev-clean','dev-other']:
        own=[i for i,row in enumerate(dev.rows) if row['split']==split]
        indices+=sorted(own,key=lambda i:dev.rows[i]['utterance_id'])[:8]
    stats=json.loads((ROOT/cfg['encoder_cache']/'stats.json').read_text())
    mean=torch.tensor(stats['mean']);std=torch.tensor(stats['std'])
    bridge=state_module.FrozenRNNT(ROOT/cfg['bridge']).eval();records=[]
    for checkpoint in ['best','last']:
        path=OUT/'A'/f'{checkpoint}.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
        model=Corrector(cfg,'A').eval();model.load_state_dict(ck['model'])
        names,parameters=zip(*model.named_parameters())
        for batch_id,lo in enumerate(range(0,len(indices),4)):
            batch=dev.collate(indices[lo:lo+4],0,'cpu')
            # Call the unchanged original loss with isolated weights. The
            # objective still uses exactly the original labels and masks.
            rep_cfg={**cfg,'distillation_weight':0.0}
            kl_cfg={**cfg,'representation_weight':0.0}
            rep,_=objective(model,bridge,batch,rep_cfg,mean,std)
            first=torch.autograd.grad(rep,parameters)
            kl,_=objective(model,bridge,batch,kl_cfg,mean,std)
            second=torch.autograd.grad(kl,parameters)
            assert all(torch.isfinite(g).all() for g in (*first,*second))
            records.append(dict(checkpoint=checkpoint,epoch=ck['epoch'],batch=batch_id,
                weighted_mae=float(rep.detach()),weighted_kl=float(kl.detach()),
                gradients=geometry(names,first,second)))
    summary={}
    for checkpoint in ['best','last']:
        own=[r for r in records if r['checkpoint']==checkpoint]
        ratio=[r['gradients']['all']['kl_to_mae_norm_ratio'] for r in own]
        cosine=[r['gradients']['all']['cosine'] for r in own]
        summary[checkpoint]=dict(epoch=own[0]['epoch'],global_norm_ratio_median=statistics.median(ratio),
            global_norm_ratio_range=[min(ratio),max(ratio)],gradient_cosine_median=statistics.median(cosine),
            mae_gradient_nonzero=all(r['gradients']['all']['mae_gradient_norm']>0 for r in own))
    result=dict(status='post-hoc diagnostic completed',training=False,parameter_updates=0,
        sample='16 fixed model-selection utterances, 8 per split; four batches of four; fixed validation windows',
        checkpoint_sha256={name:sha(OUT/'A'/f'{name}.pt') for name in ['best','last']},
        objective_sha256=sha(HERE/'objective.py'),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        utterance_ids=[dev.rows[i]['utterance_id'] for i in indices],summary=summary,records=records,
        interpretation='Weighted loss gradients, not scalar loss sizes, quantify local objective pressure. These final-checkpoint minibatches do not reconstruct historical gradients, optimizer moments or causal effects on training.',test_used=False)
    write(HERE/'artifacts/loss_gradient_audit.json',result);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
