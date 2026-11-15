#!/usr/bin/env python3
"""Select on the fixed role first, then compare every arm on identical audio."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from common import ROOT, HERE, config, sha, write

RUN = ROOT / 'runs/kl_weight_a3_v1'


def main():
    cfg = config()
    torch.set_num_threads(1)
    provenance = json.loads((HERE / 'artifacts/provenance.json').read_text())
    for name, digest in provenance['sha256'].items():
        assert sha(ROOT / name) == digest, name
    # Reuse the original, unchanged aggregation and paired-bootstrap functions.
    spec = importlib.util.spec_from_file_location('a2_frozen_metrics', ROOT / 'experiments/predict_correct_a2_v1/evaluate.py')
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    selection = {}
    for name in ['control', 'kl4', 'kl16']:
        checkpoint = ROOT / ('runs/predict_correct_a2_v1/kl/best.pt' if name == 'control'
            else f'runs/kl_weight_a3_v1/{name}/kl/best.pt')
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        selection[name] = dict(epoch=ck['epoch'], dev_clean_wer=ck['record']['selection']['dev-clean']['wer_percent'], checkpoint_sha256=sha(checkpoint))
    chosen = min(selection, key=lambda name: selection[name]['dev_clean_wer'])
    write(RUN / 'candidate_selection.json', dict(selected=chosen, selection=selection, selected_before_final_decoding=True))
    order = ['teacher', 'small', 'control', 'kl4', 'kl16']
    raw = {}
    for repeat in range(2):
        for name in order if repeat == 0 else order[::-1]:
            env = dict(os.environ, A3_ARM=name if name in ['kl4', 'kl16'] else 'control')
            subprocess.run([sys.executable, str(HERE / 'evaluate_arm.py'), '--name', name, '--repeat', str(repeat)], cwd=ROOT, env=env, check=True)
            raw[name, repeat] = json.loads((RUN / 'evaluation' / f'{name}_repeat{repeat}.json').read_text())['records']
    primary = {}
    for name in order:
        signature = lambda rows: [(r['utterance_id'], r['reference'], r['token_ids'], r['hypothesis'], r['errors']) for r in rows]
        assert signature(raw[name, 0]) == signature(raw[name, 1])
        for split in ['dev-clean', 'dev-other']:
            own = [metrics.summarize([r for r in raw[name, repeat] if r['split'] == split]) for repeat in range(2)]
            value = dict(own[0])
            for key in ['rtf', 'total_seconds', 'total_ms_per_chunk', 'rtf_with_audio_io', 'feature_ms_per_chunk', 'other_overhead_ms_per_chunk', 'compute_chunk_p50_ms', 'compute_chunk_p95_ms']:
                value[key] = sum(r[key] for r in own) / len(own)
            value['components_ms_per_chunk'] = {key: sum(r['components_ms_per_chunk'][key] for r in own) / len(own) for key in ['audio', 'head', 'decoder']}
            value['timing_repeat_ms_per_chunk'] = [r['total_ms_per_chunk'] for r in own]
            primary[f'{name}/{split}'] = value
    paired = {}
    for name in ['kl4', 'kl16']:
        for split in ['dev-clean', 'dev-other']:
            for other in ['control', 'small', 'teacher']:
                a = [r for r in raw[name, 0] if r['split'] == split]
                b = [r for r in raw[other, 0] if r['split'] == split]
                paired[f'{split}/{name}-minus-{other}'] = metrics.bootstrap(a, b, cfg)
    control_old = json.loads((ROOT / 'runs/predict_correct_a2_v1/evaluation/kl_repeat0.json').read_text())['records']
    assert [(r['utterance_id'], r['token_ids']) for r in control_old] == [(r['utterance_id'], r['token_ids']) for r in raw['control', 0]]
    qualification = {}
    reference = json.loads((ROOT / 'experiments/predict_correct_a2_v1/artifacts/results.json').read_text())['primary']['teacher/dev-clean']['rtf']
    for name in ['control', 'kl4', 'kl16']:
        score = primary[f'{name}/dev-clean']
        same = score['rtf'] / primary['teacher/dev-clean']['rtf']
        old = score['rtf'] / reference
        qualification[name] = dict(wer_below_10=score['wer_percent'] < 10,
            same_run_cost_fraction=same, A2_teacher_cost_fraction=old,
            passes_both=score['wer_percent'] < 10 and same <= .5 and old <= .5)
    result = dict(status='completed', selected_candidate=chosen, model_selection=selection,
        primary=primary, paired_comparisons=paired, qualification=qualification,
        all_repeated_transcripts_match=True, original_A2_control_tokens_unchanged=True,
        test_used=False, seed=cfg['seed'], final_role_utterances=800,
        limitation='Single seed, reused project development roles. Small has 39-frame native window versus teacher/student 45. No matched teacher/student lifetime-data budget. No FGN or 960h expansion.',
        timing_scope='CPU one thread/batch one, full resident-waveform pipeline including fbank/cache/head/native decoder. Startup, reference scoring and file IO excluded from primary; IO-inclusive RTF separate. Two reversed-order repetitions.')
    write(RUN / 'results.json', result)
    write(HERE / 'artifacts/results.json', result)
    print(json.dumps(dict(event='all_evaluation_complete', selected=chosen)), flush=True)


if __name__ == '__main__':
    main()
