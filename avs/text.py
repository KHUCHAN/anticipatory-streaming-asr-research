"""Tokenizer wrapper and WER. Model ids: 0=blank, 1=sos/eos, 2.. = sentencepiece id + 2."""
import sentencepiece as spm

BLANK, SOS, SHIFT = 0, 1, 2


class Tokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.vocab = self.sp.get_piece_size() + SHIFT

    def encode(self, text):
        return [i + SHIFT for i in self.sp.encode(text)]

    def decode(self, ids):
        return self.sp.decode([i - SHIFT for i in ids if i >= SHIFT])

    def piece(self, i):
        return self.sp.id_to_piece(i - SHIFT) if i >= SHIFT else ('<blk>' if i == 0 else '<s>')


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(refs, hyps):
    errs = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    return errs / max(1, sum(len(r.split()) for r in refs))
