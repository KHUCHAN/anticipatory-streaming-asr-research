#!/usr/bin/env python3
"""Causal display controls for the native Zipformer token stream, not FGN.

Stable-word timing uses the final *model hypothesis*, not reference accuracy.
Native RNN-T timestamps are model emission frames, not independent word ends.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from eval_zipformer_baseline import ROOT, atomic_json, sha256


POLICIES = ['eager_subwords', 'word_boundary', 'confirm_two_updates']


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def replay(events, final_text, final_time, policy):
    tokens, displays, previous_observation = [], [], []
    previous_time = None
    for event in events:
        tokens += event['tokens']
        surface = ''.join(tokens).replace('▁', ' ')
        words = surface.upper().split()
        if policy == 'eager_subwords':
            visible = words
        elif policy == 'word_boundary':
            visible = words if surface.endswith(' ') else words[:-1]
        elif policy == 'confirm_two_updates':
            if previous_time is None or event['available_s'] > previous_time:
                visible = words[:common_prefix(words, previous_observation)]
                previous_observation = words
                previous_time = event['available_s']
            else:
                visible = displays[-1][1] if displays else []
        else:
            raise ValueError(policy)
        displays.append((event['available_s'], visible))
    # Final hypothesis is used only at the real final flush, never retrospectively.
    displays.append((final_time, final_text.split()))
    return displays


def native_word_times(tokens, timestamps):
    words, times, current, last_time = [], [], '', None
    for token, timestamp in zip(tokens, timestamps):
        for character in token.replace('▁', ' '):
            if character.isspace():
                if current:
                    words.append(current.upper()); times.append(last_time); current = ''
            else:
                current += character
                last_time = timestamp
    if current:
        words.append(current.upper()); times.append(last_time)
    return words, times


def score(row, policy, padding):
    final_words = row['hypothesis'].split()
    words, native_times = native_word_times(row['tokens'], row['token_timestamps'])
    assert words == final_words
    displayed = replay(row['events'], row['hypothesis'], row['audio_seconds']+padding, policy)
    first, stable = [None]*len(words), [None]*len(words)
    previous, rewritten = [], 0
    for time, visible in displayed:
        common = common_prefix(previous, visible)
        rewritten += len(previous)-common
        aligned = common_prefix(visible, final_words)
        for i in range(aligned):
            if first[i] is None:
                first[i] = time
        previous = visible
    # A retrospective stable-time *measurement* must handle a displayed word
    # disappearing later. This does not change the causal display policy.
    suffix_minimum = len(words)
    for time, visible in reversed(displayed):
        suffix_minimum = min(suffix_minimum, common_prefix(visible, final_words))
        for i in range(suffix_minimum):
            stable[i] = time
    assert all(v is not None for v in stable)
    delays = [1000*(a-b) for a,b in zip(stable,native_times)]
    return dict(words=len(words), rewritten_word_suffix_count=rewritten,
                stable_delay_from_native_last_token_frame_ms=delays,
                word_stable_time_s=stable,
                first_correct_final_hypothesis_prefix_time_s=first)


def metric_checks():
    # A partial word must not count as the final word, and the final transcript
    # must not be exposed by a policy before the final callback.
    events = [dict(tokens=[' CA'], available_s=.5),
              dict(tokens=['T'], available_s=.82),
              dict(tokens=[' SAT'], available_s=1.14)]
    eager = replay(events, 'CAT SAT', 1.8, 'eager_subwords')
    boundary = replay(events, 'CAT SAT', 1.8, 'word_boundary')
    assert eager[:2] == [(.5,['CA']),(.82,['CAT'])]
    assert boundary[:2] == [(.5,[]),(.82,[])]
    assert boundary[2] == (1.14,['CAT'])
    for policy in POLICIES:
        before = replay(events[:1], 'CAT SAT', 1.8, policy)[0][1]
        assert 'SAT' not in before
    row = dict(hypothesis='CAT SAT',tokens=[' CA','T',' SAT'],
               token_timestamps=[.1,.2,.3],events=events,audio_seconds=1.14)
    assert score(row,'eager_subwords',.66)['word_stable_time_s'] == [.82,1.14]
    assert np.allclose(score(row,'word_boundary',.66)['word_stable_time_s'], [1.14,1.8])
    return dict(partial_word_and_final_callback='pass',known_stable_word_times='pass')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/baseline_zipformer')
    args = parser.parse_args()
    out = ROOT/args.run
    provenance = json.loads((out/'provenance.json').read_text())
    padding = provenance['config']['tail_padding_seconds']
    groups = defaultdict(list)
    for line in (out/'utterances.jsonl').open():
        row = json.loads(line)
        for policy in POLICIES:
            groups[(row['split'], policy)].append(score(row,policy,padding))
    summary = {}
    for (split, policy), scores in groups.items():
        values = [t for s in scores for t in s['stable_delay_from_native_last_token_frame_ms']]
        words = sum(s['words'] for s in scores)
        rewritten = sum(s['rewritten_word_suffix_count'] for s in scores)
        summary[f'{split}/{policy}'] = dict(
            utterances=len(scores), final_hypothesis_words=words,
            rewritten_word_suffix_count=rewritten,
            rewritten_suffixes_per_100_final_words=100*rewritten/max(1,words),
            stable_delay_relative_to_native_last_token_mean_ms=float(np.mean(values)),
            stable_delay_relative_to_native_last_token_p50_ms=float(np.percentile(values,50)),
            stable_delay_relative_to_native_last_token_p95_ms=float(np.percentile(values,95)))
    result = dict(scope='Native decoder display controls only. No future-token forecaster or FGN model was added. Final hypotheses are identical by construction.',
                  timing='Simulated arrival clock, excluding computation and UI rendering. Native token timestamps are not independent phonetic ground truth; do not compare these values to the AVS CTC-aligned latency tables.',
                  rewrite_definition='Sum of words removed from each previous display suffix after its longest common word prefix; includes changes to unfinished partial words, not unique corrected words.',
                  metric_checks=metric_checks(), results=summary,
                  input_sha256=sha256(out/'utterances.jsonl'),script_sha256=sha256(__file__))
    atomic_json(out/'display_controls.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
