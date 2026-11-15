# Legacy 20M Zipformer reproduction correction — 2026-11-15

**Resolved:** an incompatible decoder padding convention caused the large recognition gap. The original A2 small-baseline operating point is superseded; the full teacher and all A2 student results remain unchanged.

## Measured correction

| Path | Dev-clean WER % | Dev-other WER % | Clean complete ms/chunk | Other complete ms/chunk |
|---|---:|---:|---:|---:|
| Small, original incompatible padding | 23.706 | 32.352 | 4.252 | 4.240 |
| Small, repaired legacy padding | 3.560 | 10.972 | 4.390 | 4.387 |
| Full teacher, remeasured | 2.933 | 8.817 | 9.262 | 9.261 |

The corrected 20M produced no empty transcripts on these 800 utterances, versus 41 previously. Token sequences changed on 790/800 small-model utterances; all 800 teacher token sequences stayed identical. Two repetitions reproduce every transcript. Costs are complete resident-waveform processing on Apple M5 Pro 24GiB, CPU batch 1 / one thread, all neural paths FP32 ONNX; startup, file IO and reference scoring excluded. Runtime changed slightly across runs, so use the remeasured same-run teacher as the cost reference.

## Source-level cause

The original icefall ONNX greedy decoder initializes a context of two blanks, `[0,0]`. Our A2 adapter and current native sherpa-onnx greedy decoder instead supply `[-1,0]`, a newer padding convention. This legacy 2023 ONNX decoder contains an embedding Gather that accepts negative indices: `-1` selects vocabulary ID 499. The original decoder output for `[-1,0]` is exactly equal to `[499,0]`, and differs from `[0,0]` by a maximum absolute 5.0068. The resulting incorrect prediction-network prior can keep emitting blank or delete the beginning. Encoder caches, waveform scale and tail padding were not the cause of the large gap.

Changing only the small initial context to `[0,0]` restores the source contract. A separate ONNX input wrapper maps negative padding to blank for this model, preserving every learned initializer. On initial, partial and random valid contexts, it is exactly equal to the original decoder with explicitly blank-filled inputs. Unmodified native sherpa-onnx with that wrapper matches all 800 corrected token sequences and chunk counts. The original model graphs remain untouched in local checkpoint storage.

The earlier agreement with the unmodified native engine missed the problem because both implementations shared the incompatible convention. The supplied demo waveform also happened to decode correctly under that convention, so that single example did not validate the corpus-level startup behavior. This repair is model-specific and must not be applied indiscriminately to the 66M teacher or other exports.

## Evaluation boundary

Use the same 400 dev-clean + 400 dev-other role, waveform/feature extraction, tail, chunk/hop, greedy search and common text normalization as A2. No student retraining, reference-conditioned inference, FGN or test-split tuning was introduced. Small uses a 39-frame input window versus teacher/student 45 (60ms shorter native availability); both hop 32 frames. The source card reports 3.94/9.79% on full test splits, whereas this correction measures a development subset. The large extra ~20pp error is resolved; this is not a claim of exact reproduction of the source test numbers. No new full-test decode was used to choose the fix.

Original A2 inputs/sources/weights, all repeated outputs, runtime sums, exact teacher preservation and independent native matches pass verification. The corrected adapter and wrapper have separate source/graph hashes. Original A2 raw records remain preserved; the old small table describes the incompatible-padding run.

[Executed protocol](../experiments/small_zipformer_reproduction_v1/README.md) · [Corrected aggregate results](../experiments/small_zipformer_reproduction_v1/artifacts/evaluation.json) · [Verification](../experiments/small_zipformer_reproduction_v1/artifacts/verification.json) · [Wrapper evidence](../experiments/small_zipformer_reproduction_v1/artifacts/decoder_wrapper.json)

Sources: [icefall original ONNX initialization](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/onnx_pretrained.py), [current sherpa-onnx initialization](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-transducer-greedy-search-decoder.cc), [source model card](https://huggingface.co/desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small).
