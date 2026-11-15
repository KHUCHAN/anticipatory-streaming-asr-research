#!/usr/bin/env python3
"""Check the legacy wrapper using independent native feature/stream decoding."""
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'runs/small_zipformer_reproduction_v1'


def main():
    cfg = json.loads((ROOT / 'experiments/predict_correct_a2_v1/config.json').read_text())
    folder = ROOT / cfg['small_model']
    decoder = OUT / 'decoder-blank-padding.onnx'
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(tokens=str(folder / 'tokens.txt'),
        encoder=str(folder / 'encoder-epoch-99-avg-1.onnx'), decoder=str(decoder),
        joiner=str(folder / 'joiner-epoch-99-avg-1.onnx'), num_threads=1,
        sample_rate=16000, feature_dim=80, low_freq=20, high_freq=-400, dither=0,
        snip_edges=False, normalize_samples=True, enable_endpoint_detection=False,
        decoding_method='greedy_search', provider='cpu')
    token_ids = {line.rsplit(' ', 1)[0].replace('▁', ' '): int(line.rsplit(' ', 1)[1])
        for line in (folder / 'tokens.txt').read_text().splitlines()}
    rows = {r['utterance_id']: r for r in json.loads((ROOT / cfg['input_cache'] / 'index.json').read_text())}
    primary = json.loads((OUT / 'small_evaluation_repeat0.json').read_text())['records']
    records = []
    for i, old in enumerate(primary, 1):
        row = rows[old['utterance_id']]
        wave, sr = sf.read(ROOT / row['audio'], dtype='float32')
        wave = np.concatenate([wave, np.zeros(10560, np.float32)])
        stream = recognizer.create_stream()
        chunks = 0
        for lo in range(0, len(wave), 320):
            stream.accept_waveform(sr, wave[lo:lo + 320])
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
                chunks += 1
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
            chunks += 1
        result = recognizer.get_result_all(stream)
        ids = [token_ids[x] for x in result.tokens]
        records.append(dict(utterance_id=row['utterance_id'], tokens_match=ids == old['token_ids'],
            chunks_match=chunks == old['chunks'], text=result.text))
        if i % 100 == 0:
            print(json.dumps(dict(done=i, total=len(primary))), flush=True)
    assert len(records) == 800
    assert all(r['tokens_match'] and r['chunks_match'] for r in records)
    (OUT / 'native_records.json').write_text(json.dumps(records, indent=2) + '\n')
    report = dict(status='verified', utterances=len(records), all_tokens_match=True,
        all_chunks_match=True, native_version=importlib.metadata.version('sherpa-onnx'),
        decoder='Unchanged learned parameters; negative padding mapped to the legacy blank convention',
        test_used=False)
    (HERE / 'artifacts/native_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
