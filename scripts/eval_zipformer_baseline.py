#!/usr/bin/env python3
"""Reproduce the public streaming Zipformer baseline on native FLAC manifests.

Use the isolated .local/venv-baseline environment. The existing AVS vocabulary,
features, weights, and results are never overwritten. Wall time is measured;
audio arrival is simulated without sleeping. JSONL retains causal token deltas.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import platform
import time

import numpy as np
import sherpa_onnx
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
RECOGNIZER = None
CONFIG = None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def make_recognizer(config):
    model = ROOT / config['model_dir']
    suffix = config['model_suffix']
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(model / 'tokens.txt'),
        encoder=str(model / ('encoder-' + suffix)),
        decoder=str(model / ('decoder-' + suffix)),
        joiner=str(model / ('joiner-' + suffix)),
        num_threads=config['num_threads_per_worker'],
        sample_rate=config['sample_rate'], feature_dim=config['feature_dim'],
        low_freq=config['feature_low_freq'], high_freq=config['feature_high_freq'],
        dither=config['dither'], snip_edges=config['snip_edges'],
        normalize_samples=config['normalize_samples'],
        enable_endpoint_detection=config['endpoint_detection'],
        decoding_method=config['decoding_method'], provider=config['provider'],
    )


def initialize(config):
    global RECOGNIZER, CONFIG
    CONFIG = config
    RECOGNIZER = make_recognizer(config)


def word_errors(reference, hypothesis):
    """Minimum word edit distance; counts use deterministic diagonal-first ties."""
    ref, hyp = reference.split(), hypothesis.split()
    # Tuple: total errors, substitutions, deletions, insertions.
    previous = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, word in enumerate(ref, 1):
        current = [(i, 0, i, 0)]
        for j, predicted in enumerate(hyp, 1):
            diagonal = previous[j - 1]
            if word == predicted:
                diagonal = diagonal
            else:
                diagonal = (diagonal[0]+1, diagonal[1]+1, diagonal[2], diagonal[3])
            above, left = previous[j], current[-1]
            deletion = (above[0]+1, above[1], above[2]+1, above[3])
            insertion = (left[0]+1, left[1], left[2], left[3]+1)
            current.append(min([diagonal, deletion, insertion], key=lambda v: v[0]))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    return dict(errors=errors, substitutions=substitutions, deletions=deletions,
                insertions=insertions, reference_words=len(ref))


def transcribe(row, recognizer, config, packet_ms=None):
    started = time.perf_counter()
    waveform, sr = sf.read(ROOT / row['audio'], dtype='float32')
    assert sr == config['sample_rate'] and waveform.ndim == 1, row['utterance_id']
    assert len(waveform) == row['num_samples'], row['utterance_id']
    loaded = time.perf_counter()
    stream = recognizer.create_stream()
    step = int(sr * (packet_ms or config['waveform_packet_ms']) / 1000)
    events, previous_tokens, token_times, token_logp = [], [], [], []
    decode_seconds = 0.0
    chunk_seconds = []

    def drain(available_samples, phase):
        nonlocal previous_tokens, token_times, token_logp, decode_seconds
        while recognizer.is_ready(stream):
            begin = time.perf_counter()
            recognizer.decode_stream(stream)
            elapsed = time.perf_counter() - begin
            chunk_seconds.append(elapsed)
            decode_seconds += elapsed
            result = recognizer.get_result_all(stream)
            tokens = list(result.tokens)
            assert tokens[:len(previous_tokens)] == previous_tokens, 'greedy prefix changed'
            times, logp = list(result.timestamps), list(result.ys_probs)
            assert len(tokens) == len(times) == len(logp)
            k = len(previous_tokens)
            events.append(dict(available_s=available_samples/sr, phase=phase,
                               token_count=len(tokens), tokens=tokens[k:],
                               timestamps=times[k:], logp=logp[k:]))
            previous_tokens, token_times, token_logp = tokens, times, logp

    for start in range(0, len(waveform), step):
        end = min(start + step, len(waveform))
        stream.accept_waveform(sr, waveform[start:end])
        drain(end, 'audio')
    padding = int(round(config['tail_padding_seconds'] * sr))
    # Silence is presented incrementally too: a flushed token cannot be dated
    # at the end of speech when its computation needed future tail padding.
    for start in range(0, padding, step):
        end = min(start + step, padding)
        stream.accept_waveform(sr, np.zeros(end-start, dtype=np.float32))
        drain(len(waveform) + end, 'padding')
    stream.input_finished()
    drain(len(waveform) + padding, 'finished')
    result = recognizer.get_result_all(stream)
    hypothesis = ' '.join(result.text.upper().split())
    assert ' '.join(''.join(previous_tokens).replace('▁', ' ').upper().split()) == hypothesis
    ended = time.perf_counter()
    return dict(utterance_id=row['utterance_id'], split=row['split'],
                speaker_id=row['speaker_id'], reference=row['text'], hypothesis=hypothesis,
                audio_seconds=len(waveform)/sr, **word_errors(row['text'], hypothesis),
                load_seconds=loaded-started, processing_seconds=ended-loaded,
                neural_decode_seconds=decode_seconds, chunk_seconds=chunk_seconds,
                tokens=previous_tokens, token_timestamps=token_times, token_logp=token_logp,
                events=events)


def evaluate_row(row):
    return transcribe(row, RECOGNIZER, CONFIG)


def summarize(rows, wall_seconds):
    errors = sum(r['errors'] for r in rows)
    words = sum(r['reference_words'] for r in rows)
    audio = sum(r['audio_seconds'] for r in rows)
    chunks = [v*1000 for r in rows for v in r['chunk_seconds']]
    return dict(utterances=len(rows), audio_hours=audio/3600, errors=errors,
                reference_words=words, wer_percent=100*errors/max(1, words),
                substitutions=sum(r['substitutions'] for r in rows),
                deletions=sum(r['deletions'] for r in rows),
                insertions=sum(r['insertions'] for r in rows),
                summed_worker_processing_rtf=sum(r['processing_seconds'] for r in rows)/audio,
                summed_neural_decode_rtf=sum(r['neural_decode_seconds'] for r in rows)/audio,
                observed_wall_seconds=wall_seconds, observed_wall_rtf=wall_seconds/audio,
                concurrent_chunk_p50_ms=float(np.percentile(chunks, 50)),
                concurrent_chunk_p95_ms=float(np.percentile(chunks, 95)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/zipformer_baseline.json')
    parser.add_argument('--out', default='runs/baseline_zipformer')
    parser.add_argument('--splits', nargs='+')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    if args.workers:
        config['workers'] = args.workers
    if args.splits:
        config['evaluation_splits'] = args.splits
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    manifests, rows = {}, []
    for split in config['evaluation_splits']:
        manifest = ROOT / 'data/processed/manifests' / (split + '.jsonl')
        subset = [json.loads(line) for line in manifest.read_text().splitlines()]
        if args.limit:
            subset = subset[:args.limit]
        manifests[split] = dict(sha256=sha256(manifest), selected_utterances=len(subset))
        rows += subset
    model = ROOT / config['model_dir']
    model_files = [model/(name+'-'+config['model_suffix']) for name in ['encoder','decoder','joiner']]
    model_files += [model/'tokens.txt', model/'bpe.model', model/'README.md', model/'export-onnx-zipformer-online.sh']
    provenance = dict(config=config, config_sha256=sha256(ROOT/args.config),
                      script_sha256=sha256(__file__), manifests=manifests,
                      model_sha256={str(p.relative_to(ROOT)):sha256(p) for p in model_files},
                      python=platform.python_version(), machine=platform.machine(),
                      dependencies={x:importlib.metadata.version(x) for x in ['sherpa-onnx','sherpa-onnx-core','onnxruntime','numpy','soundfile','sentencepiece']},
                      limit=args.limit, started_unix=time.time(),
                      timing_note='No sleeping; CPU workers run concurrently. Arrival clock is simulated. Timings exclude model initialization; processing includes feature extraction, neural decoding and event recording. Not a microphone/display end-to-end latency benchmark.')
    existing = out / 'provenance.json'
    if existing.exists():
        old = json.loads(existing.read_text())
        for key in ['config','config_sha256','manifests','model_sha256','limit']:
            assert old[key] == provenance[key], f'Resume mismatch: {key}; use a new output directory.'
    else:
        atomic_json(existing, provenance)
    records = []
    records_path = out/'utterances.jsonl'
    if records_path.exists():
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
    done = {r['utterance_id'] for r in records}
    assert len(done) == len(records), 'Duplicate saved IDs'
    requested = {row['utterance_id'] for row in rows}
    assert done.issubset(requested), 'Saved records contain IDs outside the requested manifests'
    completed_path = out/'summary.json'
    if existing.exists() and completed_path.exists():
        completed = json.loads(completed_path.read_text())
        if completed['status']=='complete' and done==requested:
            assert completed['utterances_sha256']==sha256(records_path)
            print('Already complete: preserving existing records, provenance and original timing.',flush=True)
            print(json.dumps(completed,indent=2),flush=True)
            return
    if existing.exists():
        old = json.loads(existing.read_text())
        assert old['script_sha256']==provenance['script_sha256'], 'Resume script changed; use a new output directory.'
    snapshot = out/'eval_zipformer_baseline.executed.py'
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    assert sha256(snapshot)==provenance['script_sha256'], 'Executed source snapshot differs'
    rows = [r for r in rows if r['utterance_id'] not in done]
    started = time.perf_counter()
    with records_path.open('a', buffering=1) as stream, ProcessPoolExecutor(
            max_workers=config['workers'], mp_context=mp.get_context('spawn'),
            initializer=initialize, initargs=(config,)) as executor:
        futures = [executor.submit(evaluate_row, row) for row in rows]
        for future in as_completed(futures):
            result = future.result()
            stream.write(json.dumps(result, separators=(',', ':'))+'\n')
            records.append(result)
            if len(records) % 100 == 0:
                print(json.dumps(dict(completed=len(records),total=len(done)+len(rows),
                                      elapsed_seconds=round(time.perf_counter()-started,1))), flush=True)
    elapsed = time.perf_counter()-started
    summaries = {split:summarize([r for r in records if r['split']==split], elapsed)
                 for split in config['evaluation_splits']}
    # Per-split wall ratios share one concurrent run and are not independent RTFs.
    for split, summary in summaries.items():
        summary.pop('observed_wall_rtf')
        summary.pop('observed_wall_seconds')
        expected = config['published_chunk_wise_greedy_wer'].get(split)
        if expected is not None:
            summary['published_wer_percent'] = expected
            summary['difference_percentage_points'] = summary['wer_percent']-expected
            summary['within_prefixed_margin'] = abs(summary['wer_percent']-expected) <= config['wer_acceptance_absolute_percentage_points']
    result = dict(status='complete', splits=summaries, total=summarize(records,elapsed),
                  resumed_saved_utterances=len(done),
                  wall_time_note='This invocation only; if resumed, excludes earlier invocations.',
                  utterances_sha256=sha256(records_path), finished_unix=time.time())
    atomic_json(out/'summary.json', result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
