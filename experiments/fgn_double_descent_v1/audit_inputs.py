#!/usr/bin/env python3
"""Measure conflicting fixed no-audio cases without touching model training."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    cfg = json.loads((ROOT/'experiments/fgn_capacity_v1/config.json').read_text())
    index = json.loads((ROOT/'runs/fgn_state_v1/cache/index.json').read_text())
    metadata = json.loads((ROOT/'runs/band_fgn_v1/gold_metadata.json').read_text())
    rows = [row for row in index if row['split']==cfg['train_split']]
    counts = defaultdict(Counter)
    for row in rows:
        uid = row['utterance_id']
        seed = int.from_bytes(hashlib.sha256(f"{cfg['fixed_case_seed']}:{uid}".encode()).digest()[:8],'big')
        rng = random.Random(seed)
        gold,frames = metadata[uid]['gold'],metadata[uid]['gold_frames']
        j = rng.randrange(len(gold))
        hi = min(row['nchunks']-1,max(0,frames[j]//8))
        lo = min(hi,max(0,frames[j-1]//8+1)) if j else 0
        cutoff = rng.randint(lo,hi)
        if cutoff==0:
            counts[tuple([1]+gold[:j])][gold[j]] += 1
    total_zero = sum(sum(values.values()) for values in counts.values())
    unavoidable = sum(sum(values.values())-max(values.values()) for values in counts.values())
    entropy = sum(-count*math.log(count/sum(values.values())) for values in counts.values() for count in values.values())/len(rows)
    result = dict(training_cases=len(rows),zero_audio_cases=total_zero,
        distinct_complete_prefix_groups=len(counts),conflicting_groups=sum(len(values)>1 for values in counts.values()),
        conservative_expected_hard_error_count_lower_bound_from_zero_audio_groups=unavoidable,
        conservative_expected_hard_error_rate_lower_bound=unavoidable/len(rows),
        conservative_expected_nll_lower_bound=entropy,
        scope='Exact complete reference prefixes with zero arrived audio and identical zero state/memory. Fresh functional noise independent of targets; bounds apply to expected empirical risk over fresh noise, not every finite fixed-noise evaluation. Grouping by complete prefixes is conservative for a shorter effective context. No per-utterance data or targets are exported.',
        test_used=False)
    (HERE/'artifacts/input_conflicts.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
