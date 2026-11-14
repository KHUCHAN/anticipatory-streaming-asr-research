#!/usr/bin/env python3
"""Evaluation-only full fixed-training-set accuracy, likelihood and margins."""
import math
import time

import torch
from torch.nn import functional as F


@torch.inference_mode()
def evaluate(model, ds, cfg, emit=None):
    was_training = model.training
    cpu_rng, mps_rng = torch.get_rng_state(), torch.mps.get_rng_state()
    model.eval()
    torch.manual_seed(502)
    count = errors = 0
    sums = dict(nll=0., ce=0., crps=0.)
    subsets = {name: dict(examples=0, errors=0, nll_sum=0.) for name in ['zero_arrived_audio', 'arrived_audio']}
    margins = []
    started = time.time()
    try:
        for batch_index, indices in enumerate(ds.batches(cfg), 1):
            batch = ds.collate(indices, None, 'mps')
            logits, crps, _ = model(batch, cfg['forecast_horizon'])
            targets = batch[-1]
            logp = torch.logsumexp(logits.log_softmax(-1), 0) - math.log(len(logits))
            nll = F.nll_loss(logp, targets, reduction='none')
            ce = torch.stack([F.cross_entropy(branch, targets, label_smoothing=cfg['label_smoothing'], reduction='none') for branch in logits]).mean(0)
            incorrect = logp.argmax(-1) != targets
            true_logp = logp.gather(1, targets[:, None])[:, 0]
            others = logp.clone()
            others.scatter_(1, targets[:, None], -float('inf'))
            margins.append((true_logp - others.max(-1).values).cpu())
            n = len(indices)
            count += n
            errors += int(incorrect.sum())
            sums['nll'] += float(nll.sum())
            sums['ce'] += float(ce.sum())
            # Match the earlier evaluation's utterance-weighted batch CRPS.
            sums['crps'] += n * float(crps)
            zero = batch[2] == 0
            for name, mask in [('zero_arrived_audio', zero), ('arrived_audio', ~zero)]:
                subsets[name]['examples'] += int(mask.sum())
                subsets[name]['errors'] += int((incorrect & mask).sum())
                subsets[name]['nll_sum'] += float(nll[mask].sum())
            if batch_index % 100 == 0 and emit:
                emit(dict(event='full_train_progress', batches=batch_index, examples=count, seconds=time.time()-started))
            if torch.mps.driver_allocated_memory()/2**30 > cfg['max_driver_memory_gib']:
                raise RuntimeError('Full-training evaluation exceeded the declared MPS memory guard')
        assert count == len(ds.rows)
        combined = torch.cat(margins)
        result = dict(examples=count, errors=errors, hit1=1-errors/count,
                      nll=sums['nll']/count, ce=sums['ce']/count, crps=sums['crps']/count,
                      objective=(sums['ce']+cfg['crps_weight']*sums['crps'])/count,
                      target_vs_best_other_logp_margin=dict(mean=float(combined.mean()), minimum=float(combined.min()),
                          p05=float(combined.quantile(.05)), median=float(combined.median()), nonpositive_count=int((combined<=0).sum())),
                      zero_hard_error_in_this_evaluation=errors==0,
                      noise_seed=502, stochastic_branches=2, subsets=subsets,
                      seconds=time.time()-started,
                      interpretation='Full fixed-case set, evaluation mode, one deterministic noise-seeded two-branch pass. This does not establish correct predictions for every noise draw or an interpolation threshold.')
        return result
    finally:
        torch.set_rng_state(cpu_rng)
        torch.mps.set_rng_state(mps_rng)
        model.train(was_training)
