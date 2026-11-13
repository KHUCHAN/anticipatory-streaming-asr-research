#!/usr/bin/env python3
"""Stage-0 preprocessing: log-mel cache (float16 memmap), train statistics, tokenizer.

Tokenizer and normalisation statistics are fitted on train-clean-100 only.
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avs.features import logmel, num_frames, N_MEL  # noqa: E402

OUT = ROOT / 'data/processed/feats'
_mm = {}


def work(job):
    import soundfile as sf
    path, split, offset, t, total = job
    wave, sr = sf.read(str(ROOT / path), dtype='float32')
    assert sr == 16000 and wave.ndim == 1
    feat = logmel(wave)
    assert feat.shape[0] == t, (path, feat.shape, t)
    if split not in _mm:
        _mm[split] = np.memmap(OUT / f'{split}.f16', dtype=np.float16, mode='r+', shape=(total, N_MEL))
    _mm[split][offset:offset + t] = feat.astype(np.float16)
    return feat.sum(0, dtype=np.float64), (feat.astype(np.float64) ** 2).sum(0), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train-clean-100', 'dev-clean', 'dev-other', 'test-clean', 'test-other'])
    ap.add_argument('--procs', type=int, default=12)
    ap.add_argument('--vocab', type=int, default=500)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        rows = [json.loads(l) for l in (ROOT / f'data/processed/manifests/{split}.jsonl').open()]
        lens = [num_frames(r['num_samples']) for r in rows]
        offsets = np.concatenate([[0], np.cumsum(lens)]).tolist()
        total = offsets[-1]
        np.memmap(OUT / f'{split}.f16', dtype=np.float16, mode='w+', shape=(total, N_MEL)).flush()
        jobs = [(r['audio'], split, offsets[i], lens[i], total) for i, r in enumerate(rows)]
        s1, s2, n = np.zeros(N_MEL), np.zeros(N_MEL), 0
        with Pool(args.procs) as pool:
            for a, b, t in pool.imap_unordered(work, jobs, chunksize=32):
                s1 += a; s2 += b; n += t
        json.dump({'split': split, 'ids': [r['utterance_id'] for r in rows], 'texts': [r['text'] for r in rows],
                   'speakers': [r['speaker_id'] for r in rows], 'offsets': offsets[:-1], 'lens': lens, 'total_frames': total},
                  (OUT / f'{split}.json').open('w'))
        print(f'FEATS {split}: {len(rows)} utts, {total} frames ({total / 360000:.2f} h)', flush=True)
        if split == 'train-clean-100':
            mean = s1 / n
            std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-8))
            json.dump({'mean': mean.tolist(), 'std': std.tolist(), 'frames': n, 'source': split}, (OUT / 'stats.json').open('w'))
            import sentencepiece as spm
            txt = OUT / 'train_text.txt'
            txt.write_text('\n'.join(r['text'] for r in rows) + '\n')
            spm.SentencePieceTrainer.train(input=str(txt), model_prefix=str(OUT / 'spm'), vocab_size=args.vocab - 2,
                                           model_type='unigram', character_coverage=1.0, bos_id=-1, eos_id=-1, unk_id=0,
                                           input_sentence_size=0, minloglevel=2)
            print(f'TOKENIZER unigram vocab={args.vocab} (incl. blank, sos/eos) from train text only', flush=True)


if __name__ == '__main__':
    main()
