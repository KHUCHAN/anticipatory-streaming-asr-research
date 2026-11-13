"""Causal 80-dim log-mel features (25 ms window, 10 ms hop, no centering).

Frame i covers samples [160*i, 160*i+400): it never looks further than 25 ms
ahead of its own start. Normalisation uses fixed train-set statistics, so no
utterance-level (future-dependent) statistic enters the model input.
"""
import numpy as np

SR, N_FFT, HOP, N_MEL = 16000, 400, 160, 80


def num_frames(num_samples):
    return 0 if num_samples < N_FFT else 1 + (num_samples - N_FFT) // HOP


def _mel_matrix(fmin=20.0, fmax=7600.0):
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    inv = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = inv(np.linspace(mel(fmin), mel(fmax), N_MEL + 2))
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    fb = np.zeros((N_MEL, freqs.size), dtype=np.float32)
    for m in range(N_MEL):
        lo, ce, hi = pts[m], pts[m + 1], pts[m + 2]
        fb[m] = np.clip(np.minimum((freqs - lo) / (ce - lo), (hi - freqs) / (hi - ce)), 0, None)
    return fb


_MEL = _mel_matrix()
_WIN = np.hanning(N_FFT + 1)[:-1].astype(np.float32)


def logmel(wave):
    """wave: float32 [-1,1] mono 16 kHz -> float32 [T, 80]."""
    t = num_frames(wave.shape[0])
    if t == 0:
        return np.zeros((0, N_MEL), np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(wave, N_FFT)[::HOP][:t]
    spec = np.fft.rfft(frames * _WIN, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return np.log(np.maximum(power @ _MEL.T, 1e-10))
