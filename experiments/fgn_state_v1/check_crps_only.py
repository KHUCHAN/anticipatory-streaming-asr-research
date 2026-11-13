"""Prove marginal-only training is invariant to future token supervision."""
import json
import torch
from data import ROOT,OUT
from model import StateFGN,FrozenRNNT,rollout_loss

torch.set_num_threads(1);torch.manual_seed(37)
cfg=json.loads((ROOT/'experiments/fgn_state_v1/config_crps_only.json').read_text())
model=StateFGN(cfg,'crps_only_s37');bridge=FrozenRNNT(OUT/'rnnt_bridge.npz')
states=torch.randn(2,6,8,512);indices=torch.tensor([[0,2],[1,2]]);lengths=torch.tensor([6,6])
contexts=torch.zeros(2,6,8,2,dtype=torch.long);targets=torch.zeros(2,6,8,dtype=torch.long)
mean=torch.zeros(512);std=torch.ones(512);grads=[];losses=[];nll=[]
for variant in range(2):
    model.zero_grad(set_to_none=True)
    if variant:
        contexts=torch.randint(0,500,contexts.shape);targets=torch.randint(0,500,targets.shape)
    loss,parts=rollout_loss(model,bridge,states,contexts,targets,indices,lengths,mean,std,2,
                            generator=torch.Generator().manual_seed(417))
    assert float(loss.detach())==float(parts['state_crps'])
    loss.backward();grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
    losses.append(float(loss.detach()));nll.append(float(parts['token_nll']))
assert losses[0]==losses[1] and torch.equal(grads[0],grads[1])
assert nll[0]!=nll[1] and grads[0].norm()>0
result=dict(status='pass',token_loss_weight=0.,changed_future_contexts_and_targets=True,
            loss_exactly_invariant=True,gradient_exactly_invariant=True,
            loss=losses[0],monitored_token_nll=nll,gradient_norm=float(grads[0].norm()))
(OUT/'checks_crps_only.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
