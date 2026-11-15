#!/usr/bin/env python3
"""Verify matched training, original inputs, checkpoint selection and raw scores."""
import json
import math

import torch

from common import ROOT, HERE, sha, write, word_error

RUN = ROOT / 'runs/kl_weight_a3_v1'


def main():
    torch.set_num_threads(1)
    provenance = json.loads((HERE / 'artifacts/provenance.json').read_text())
    for name, digest in provenance['sha256'].items():
        assert sha(ROOT / name) == digest, name
    replay = json.loads((HERE / 'artifacts/control_replay.json').read_text())
    assert replay['status'] == 'verified' and replay['updates'] == 100
    phases = {}
    for arm, multiplier in [('kl4', 4.), ('kl16', 16.)]:
        folder = RUN / arm / 'kl'
        rows = [json.loads(s) for s in (folder / 'log.jsonl').read_text().splitlines()]
        rows = [r for r in rows if r['event'] == 'eval']
        assert [r['epoch'] for r in rows] == list(range(1, 25))
        best = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
        last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        assert last['step'] == 446 * 24 and last['epoch'] == 24
        assert best['record'] == min(rows, key=lambda r: r['selection']['dev-clean']['wer_percent'])
        assert best['cfg'] == last['cfg'] and best['cfg']['a3_multiplier'] == multiplier
        for name, digest in best['source_sha256'].items():
            assert sha(ROOT / name) == digest, name
        calibration = json.loads((RUN / arm / 'kl_calibration.json').read_text())
        assert calibration['source_mae_sha256'] == sha(ROOT / best['cfg']['initial_checkpoint'])
        for row in rows:
            assert math.isclose(row['weight'], calibration['weight'] * min(1., row['epoch'] / 2))
        phases[arm] = dict(epochs=24, optimizer_steps=last['step'], selected_epoch=best['epoch'],
            coefficient=calibration['weight'], checkpoint_sha256=sha(folder / 'best.pt'))
        del best, last
    result = json.loads((RUN / 'results.json').read_text())
    selection = json.loads((RUN / 'candidate_selection.json').read_text())
    assert selection['selected_before_final_decoding']
    assert selection['selected'] == min(selection['selection'], key=lambda k: selection['selection'][k]['dev_clean_wer']) == result['selected_candidate']
    roles = json.loads((ROOT / provenance['base_config']['protocol']).read_text())
    expected = set(roles['evaluation_ids'])
    for name in ['teacher', 'small', 'control', 'kl4', 'kl16']:
        signatures = []
        times = {split: [] for split in ['dev-clean', 'dev-other']}
        for repeat in range(2):
            raw = json.loads((RUN / 'evaluation' / f'{name}_repeat{repeat}.json').read_text())['records']
            assert len(raw) == 800 and {r['utterance_id'] for r in raw} == expected
            for r in raw:
                assert word_error(r['reference'], r['hypothesis']) == (r['errors'], r['reference_words'])
                assert math.isclose(r['total_seconds'], r['feature_seconds'] + sum(r['components'].values()) + r['other_overhead_seconds'], abs_tol=1e-8)
            signatures.append([(r['utterance_id'], r['token_ids'], r['errors']) for r in raw])
            for split in times:
                own = [r for r in raw if r['split'] == split]
                aggregate = result['primary'][f'{name}/{split}']
                assert len(own) == 400 and sum(r['errors'] for r in own) == aggregate['errors']
                words = sum(r['reference_words'] for r in own)
                assert math.isclose(100 * sum(r['errors'] for r in own) / words, aggregate['wer_percent'])
                seconds = sum(r['total_seconds'] for r in own)
                audio = sum(r['audio_seconds'] for r in own)
                chunks = sum(r['chunks'] for r in own)
                times[split].append((seconds / audio, 1000 * seconds / chunks))
        assert signatures[0] == signatures[1]
        for split, values in times.items():
            for i, key in enumerate(['rtf', 'total_ms_per_chunk']):
                assert math.isclose(sum(x[i] for x in values) / 2, result['primary'][f'{name}/{split}'][key], abs_tol=1e-10)
    for arm in ['control', 'kl4', 'kl16']:
        metadata = json.loads((HERE / 'artifacts' / f'onnx_{arm}.json').read_text())
        assert not metadata['weights_changed'] and not metadata['quantized']
        assert max(metadata['random_stream_max_differences']) < 2e-4
        for name, digest in metadata['files'].items():
            assert sha(ROOT / metadata['directory'] / name) == digest
        assert json.loads((HERE / 'artifacts' / f'backend_{arm}.json').read_text())['all_match']
    assert result['original_A2_control_tokens_unchanged'] and not result['test_used']
    verified = dict(status='verified', phases=phases, first_100_control_updates_reproduced=True,
        original_inputs_and_sources_unchanged=True, selected_before_final_decoding=True,
        all_800_role_ids_verified=True, all_repeated_transcripts_match=True,
        complete_runtime_sums_verified=True, fp32_exports_verified=True,
        original_A2_control_tokens_unchanged=True, test_used=False)
    write(HERE / 'artifacts/verification.json', verified)
    print(json.dumps(verified, indent=2))


if __name__ == '__main__':
    main()
