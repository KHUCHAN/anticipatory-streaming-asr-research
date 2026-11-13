#!/usr/bin/env python3
"""Describe recorded risk trajectories; do not classify or exclude double descent."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'artifacts/learning_curves.csv'


def slope(rows, key):
    xs = [int(row['epoch']) for row in rows]
    ys = [float(row[key]) for row in rows]
    xm, ym = sum(xs)/len(xs), sum(ys)/len(ys)
    return sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)


def main():
    rows = list(csv.DictReader(SOURCE.open()))
    models = []
    for layers in [4, 8, 16]:
        history = [row for row in rows if int(row['layers']) == layers]
        assert [int(row['epoch']) for row in history] == list(range(28))
        early = [row for row in history if int(row['epoch']) <= 11]
        late = [row for row in history if int(row['epoch']) >= 12]
        metrics = {}
        for key in ['dev_nll', 'dev_objective', 'dev_crps']:
            first_min = min(early, key=lambda row: float(row[key]))
            late_min = min(late, key=lambda row: float(row[key]))
            peak = max(history, key=lambda row: float(row[key]))
            metrics[key] = dict(early_min_epoch=int(first_min['epoch']), early_min=float(first_min[key]),
                                late_min_epoch=int(late_min['epoch']), late_min=float(late_min[key]),
                                peak_epoch=int(peak['epoch']), peak=float(peak[key]),
                                final=float(history[-1][key]),
                                late_linear_slope_per_epoch=slope(late, key),
                                late_min_below_early_min=float(late_min[key]) < float(first_min[key]))
        models.append(dict(generator_layers=layers, metrics=metrics))
    value = dict(status='post-hoc descriptive audit; no additional training',
                 observed_epoch_window=[0,27], inherited_epoch_window=[0,11],
                 source_csv_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                 seed_count=1, generator_depths=[4,8,16], models=models,
                 interpretation='Late dev NLL and optimized-objective slopes are positive; dev CRPS slopes are negative. Component recovery is present, but no formal double-descent acceptance/rejection rule is applied.',
                 caveats=['Slopes are descriptive least-squares summaries, not significance tests.',
                          'A second decline need not beat the first minimum; the below-minimum field is descriptive only.',
                          'Finite training and three warm-started depths cannot exclude later epoch-wise or other model-wise double descent.',
                          'The whole-training-set interpolation threshold was not measured. Train-probe accuracy is not a substitute.',
                          'Noise injected into the generator is not the same intervention as corrupted target labels in double-descent studies.',
                          'Label smoothing does not by itself rule out double descent.'],
                 primary_references=['https://arxiv.org/abs/1912.02292','https://arxiv.org/abs/1812.11118'])
    (HERE / 'artifacts/double_descent_audit.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))


if __name__ == '__main__':
    main()
