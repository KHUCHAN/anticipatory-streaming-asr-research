#!/usr/bin/env python3
"""Verify the before/after small-recognizer correction and untouched teacher."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'runs/small_zipformer_reproduction_v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    original = json.loads((ROOT / 'experiments/predict_correct_a2_v1/artifacts/provenance.json').read_text())
    for name, digest in original['sha256'].items():
        assert sha(ROOT / name) == digest, name
    result = json.loads((HERE / 'artifacts/evaluation.json').read_text())
    assert result['source_sha256'] == sha(HERE / 'reproduce.py')
    wrapper = json.loads((HERE / 'artifacts/decoder_wrapper.json').read_text())
    assert wrapper['original_weights_preserved'] and wrapper['wrapper_vs_explicit_blank_max_difference'] == 0
    assert wrapper['original_negative_vs_last_vocab_id_max_difference'] == 0
    assert sha(ROOT / wrapper['wrapper']) == wrapper['wrapped_sha256']
    native = json.loads((HERE / 'artifacts/native_audit.json').read_text())
    assert native['all_tokens_match'] and native['all_chunks_match'] and native['utterances'] == 800
    diagnostics = {}
    for name in ['teacher', 'small']:
        pairs = [json.loads((OUT / f'{name}_evaluation_repeat{i}.json').read_text())['records'] for i in range(2)]
        signature = lambda rows: [(r['utterance_id'], r['token_ids'], r['hypothesis'], r['errors']) for r in rows]
        assert signature(pairs[0]) == signature(pairs[1]) and len(pairs[0]) == 800
        old = json.loads((ROOT / f'runs/predict_correct_a2_v1/evaluation/{name}_repeat0.json').read_text())['records']
        assert [r['utterance_id'] for r in old] == [r['utterance_id'] for r in pairs[0]]
        if name == 'teacher':
            assert signature(old) == signature(pairs[0])
        diagnostics[name] = dict(old_empty=sum(not r['hypothesis'] for r in old),
            repaired_empty=sum(not r['hypothesis'] for r in pairs[0]),
            changed_utterance_tokens=sum(a['token_ids'] != b['token_ids'] for a, b in zip(old, pairs[0])))
        for split in ['dev-clean', 'dev-other']:
            sums = []
            for rows in pairs:
                own = [r for r in rows if r['split'] == split]
                score = result['scores'][name][split]
                assert len(own) == 400 and sum(r['errors'] for r in own) == score['errors']
                assert math.isclose(100 * score['errors'] / score['reference_words'], score['wer_percent'])
                seconds = sum(r['total_seconds'] for r in own)
                sums.append((seconds / sum(r['audio_seconds'] for r in own), 1000 * seconds / sum(r['chunks'] for r in own)))
            for i, key in enumerate(['rtf', 'total_ms_per_chunk']):
                assert math.isclose(sum(x[i] for x in sums) / 2, result['scores'][name][split][key], abs_tol=1e-10)
    evidence = dict(status='verified', original_A2_inputs_sources_and_weights_preserved=True,
        teacher_tokens_unchanged=True, legacy_negative_id_wraparound_confirmed=True,
        original_weights_preserved_by_wrapper=True, all_repeated_transcripts_match=True,
        independent_native_800_tokens_and_chunks_match=True, diagnostics=diagnostics,
        source_sha256={p.name: sha(p) for p in HERE.glob('*.py')}, test_used=False)
    (HERE / 'artifacts/verification.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
