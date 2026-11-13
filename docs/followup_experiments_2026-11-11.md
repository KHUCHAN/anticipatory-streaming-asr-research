# Follow-up four experiments — 2026-11-11

This is a follow-up verification of the 100h·17.28M acoustic backbone and the 4.47M predictor. It is not the result of the new 960h·100M models. The original results and learning weights are stored locally in `runs/followups/`.

| Task | Status |
|---|---|
| Uncache real text | Done |
| 5/10 minute waveform stream | complete |
| Two models × Additional seed 2 | Completed |
| 2-pass Screen stabilization | Completed |

## 1. Text KV/logp cache actually turned off

Compare released/unreleased caches in the same layer-band2 model. Retain recent caches required by live beams and committed nodes. Keep historical token IDs/trie metadata for output reconstruction, distinct from audio cross-attention memory.

test-clean/test-other Chunk 160/320/640ms was applied to the 100 utterance and the same 200 utterance, respectively. It is not interpreted as 600 independent speech samples. We compared the confirmed output, final transcription, and confirmed node log-probability of all chunks on CPU FP32.

| split / chunk | Same warrior | Score maximum difference | Unreleased peak KiB | Released peak KiB |
|---|---:|---:|---:|---:|
| test-clean/c4 | 100/100 | 0 | 1413.3 | 109.5 |
| test-clean/c8 | 100/100 | 0 | 1672.1 | 89.6 |
| test-clean/c16 | 100/100 | 0 | 1901.0 | 109.5 |
| test-other/c4 | 100/100 | 0 | 1303.9 | 99.5 |
| test-other/c8 | 100/100 | 0 | 1512.9 | 99.5 |
| test-other/c16 | 100/100 | 0 | 1682.1 | 109.5 |

Peak is the sum of retained text KV/logp **backing storage** after chunk processing. Shared storage across projection views counts once. It excludes weights, temporary work tensors and process RSS. Distinguish the 31kB-per-hypothesis calculation from actual complete-beam storage.

Implementation: [avs/stream.py](../avs/stream.py), `Predictor(..., release_cache=True)`. CLI: `scripts/run_stage3.py --methods H1a --release-cache --hold 0`. Standard stage3 `peak_pred_state_kb` is the old active-node estimate, not total retained storage. This table uses backing-storage measurements from [check_bounded_cache.py](../scripts/check_bounded_cache.py). Original record: `runs/followups/cache.json` (local-only).

## 2. 5/10 minute continuous stream

Concatenate original test FLAC utterances by ID with 300ms silence. Each 5-minute stream is a prefix of its 10-minute stream, not a real long conversation or independent repeat. Approximate reference times add waveform offsets to original per-utterance CTC alignments.

Also measure WER with state reset per utterance using identical audio, frontend and model. Mel is precomputed, but encoder caches are actually updated. CPU 1-thread timings use a simulated arrival clock and overlap seed training, so they are not an isolated-device benchmark.

| Stream | Real seconds | All AR reset → Continuous WER | Each floor 2 reset → Continuous WER | After release text peak KiB | audio KV Final MiB |
|---|---:|---:|---:|---:|---:|
| test-clean/5min | 306.0 | 16.898 → 20.486 | 17.130 → 17.245 | 69.7 | 59.8 |
| test-clean/10min | 617.1 | 16.717 → 20.365 | 17.021 → 16.900 | 69.7 | 120.5 |
| test-other/5min | 304.7 | 34.011 → 41.283 | 34.866 → 35.722 | 89.6 | 59.5 |
| test-other/10min | 604.5 | 36.209 → 42.792 | 36.947 → 37.230 | 89.6 | 118.1 |

Long-stream peaks are maxima of snapshots every 32 chunks, approximately 10.24 seconds at 320ms, plus first/last snapshots. They are not process peak RSS over every step/workspace. Short cache verification measures retained storage after every chunk.

Check long-stream transcript equality for released/unreleased band2. Smaller text tensor caches do not stop audio K/V, encoder history or token metadata from growing. **Bounded complete system state is not established.**

`runs/followups/long_stream.json` (local-only) contains RTF, first/last-minute step times, encoder caches, metadata-node counts and temporal snapshots. [Runner](../scripts/long_stream_followup.py).

## 3. learning seed repetition

The original training seed is 2. Additional seeds 3 and 4 each train full and layer-band2 models for 15 epochs from the same Stage 1 decoder. Dropout, batch order and audio cutoffs are paired by seed. Each uses its best dev-NLL checkpoint.

Evaluate 500 utterances each from dev-clean/test-clean/test-other, sample seed 11, chunks 160/320/640ms, beam 4, prior weight 0.5, hold 0. One frozen acoustic backbone. Scratch initialization or new acoustic-backbone seeds are outside these four follow-ups.

After queue/session changes, the first full_s3 run stopped at 14 epochs. Preserve interruption records in `runs/followups/interrupted/` and rerun all 15 epochs with identical seed/recipe. Exclude the partial run from repetition counts and performance comparisons. Original partial weights were overwritten during rerun; interruption.json records the actual epoch of the retained snapshot.

| split / chunk | Total AR average WER | Average WER by layer 2 Average WER | Difference average ± seed SD (%p) | Difference range by seed (%p) |
|---|---:|---:|---:|---:|
| dev-clean/c4 | 16.784 | 16.915 | +0.131 ± 0.070 | +0.060 … +0.201 |
| dev-clean/c8 | 14.637 | 14.754 | +0.117 ± 0.110 | +0.030 … +0.241 |
| dev-clean/c16 | 13.716 | 13.830 | +0.114 ± 0.081 | +0.040 … +0.201 |
| test-clean/c4 | 16.472 | 16.485 | +0.013 ± 0.041 | -0.030 … +0.051 |
| test-clean/c8 | 14.534 | 14.537 | +0.003 ± 0.117 | -0.121 … +0.111 |
| test-clean/c16 | 13.134 | 13.262 | +0.128 ± 0.097 | +0.040 … +0.232 |
| test-other/c4 | 39.279 | 39.264 | -0.015 ± 0.118 | -0.144 … +0.088 |
| test-other/c8 | 36.839 | 36.916 | +0.077 ± 0.022 | +0.055 … +0.100 |
| test-other/c16 | 34.863 | 34.852 | -0.011 ± 0.128 | -0.155 … +0.088 |

The seed average difference of 9 conditions is -0.015…+0.131%p. Although the observed value is practically close, it does not mean that it passes the equivalence test. In test-clean 640ms, the limits per layer were slightly high for all three seeds. Since the acoustic backbone and decoder initial values ​​were fixed and only the predictor training RNG was repeated, it does not generalize to new backbone and random initialization settings.

Three-seed means/standard deviations are descriptive. Per-seed paired utterance-bootstrap 95% intervals are saved in `runs/followups/seeds/summary.json` (local-only). An interval containing zero does not establish equivalence. [Runner](../scripts/repeat_predictors.py) · [Aggregation](../scripts/summarize_seed_repeats.py).

## 4. 2-pass Screen merging and display stabilization

Use saved events for 300 utterances/split. Count repeat agreement **only after new audio arrives**; a slow-pass reread or forecast event is not another observation. Protect completed slow prefixes and finalize from the last slow output.

Compare temporal-boundary splicing, agreement over 2/3 observations, hiding the last unfinished word, and token-suffix alignment at splice boundaries. Keep existing gate/threshold 0.87. Select rules on dev-clean, never retune on test.

The dev selection rule is **overlap**. Mean delay increase 80ms or lower · P95 increase 160ms or lower The total number of corrections for the presence or absence of forecast was minimized. confirm2/3 also left separate diagnostic test results for the requested basic rules.

| split | rule | forecast | final WER | stable P50 / P95 ms | rewrite /100 word |
|---|---|---|---:|---:|---:|
| test-clean | raw | none | 12.78 | 127.2 / 327.5 | 21.21 |
| test-clean | raw | Enabled | 12.78 | 86.8 / 296.4 | 34.64 |
| test-clean | overlap | none | 12.78 | 127.2 / 248.0 | 19.63 |
| test-clean | overlap | use | 12.78 | 86.7 / 247.2 | 33.13 |
| test-clean | confirm2 | none | 12.78 | 287.0 / 526.5 | 15.66 |
| test-clean | confirm3 | none | 12.78 | 446.4 / 614.6 | 8.02 |
| test-other | raw | none | 37.33 | 130.1 / 689.0 | 42.81 |
| test-other | raw | Enabled | 37.33 | 126.4 / 689.0 | 51.24 |
| test-other | overlap | none | 37.33 | 129.1 / 648.4 | 44.40 |
| test-other | overlap | use | 37.33 | 126.3 / 616.4 | 52.79 |
| test-other | confirm2 | None | 37.33 | 326.4 / 776.6 | 31.76 |
| test-other | confirm3 | None | 37.33 | 447.6 / 857.4 | 16.70 |

Final WER is identical because the slow pass owns final transcripts. Revision counts accumulate suffix rewrites, not uniquely revised words. Token alignment reduces clean revisions/tail latency but increases rewriting on other. Repeat agreement trades display delay for fewer revisions. This does not justify making two-pass the main configuration.

Screen-event replay excludes actual UI and two-encoder contention. Additional alignment CPU time is recorded separately and not added to the replay clock. `runs/followups/two_pass_stability.json` (local-only) · [Analysis](../scripts/stabilize_two_pass.py).

## Execution and Remaining Range

```bash
.venv/bin/python scripts/check_bounded_cache.py
.venv/bin/python scripts/long_stream_followup.py
.venv/bin/python scripts/stabilize_two_pass.py
.venv/bin/python scripts/repeat_predictors.py
.venv/bin/python scripts/summarize_seed_repeats.py
.venv/bin/python scripts/report_followups.py
```

80ms frame/BPE1024 retraining and FastEmit transducer experiments are not included in the four works approved this time. There remains room for forecasting in a stronger backbone, layer-by-layer 2 predictor without initialization, and voice context limitations. Distinguish between the original sound 960h being ready and 960h model learning being completed.

`runs/followups/provenance.json` (local-only) records completed checkpoints, code/input SHA-256 and selected epochs. Interrupted partial runs are excluded. No Git commit/push occurred at that study's completion.
