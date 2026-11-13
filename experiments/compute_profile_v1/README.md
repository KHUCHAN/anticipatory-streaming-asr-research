# Native Zipformer compute profile

Completed read-only CPU timing of the existing adapter. This is a bottleneck diagnosis, not an acceleration implementation or a live microphone benchmark.

- Model: frozen public 66.11M Streaming Zipformer; FP32, 320ms chunks.
- Runtime: batch 1, CPUExecutionProvider, one intra-op and inter-op thread.
- Sample: 12 dev-clean plus 12 dev-other utterances, independently sampled within each split with seed 37.
- Warmup: two utterances, excluded from component and wall timing.
- Initial result: 431 chunks, 129.2300625 seconds of audio; encoder 98.3159% of neural execution time.
- Component denominator excludes fbank, input loading, bookkeeping and initialization. It is not the entire application runtime.

## Records

`artifacts/profile_initial.json` preserves the first interactive measurement without rewriting its values. `artifacts/profile_reproduction.json` is produced by the standalone script and adds dependency, source, manifest and model hashes. Repeated timings can differ and must not be interpreted as a method improvement.

## Reproduce

From the repository root, after preparing the public baseline environment, model and manifests:

```bash
.local/venv-baseline/bin/python scripts/profile_zipformer_compute.py \
  --output experiments/compute_profile_v1/artifacts/profile_reproduction.json
```

Use a new output path for subsequent runs; existing measurements are protected from overwrite. This command performs inference only and does not train or modify a model.

The proposed prediction/correction architecture and its unimplemented status are documented in [the acceleration plan](../../docs/predict_correct_acceleration_plan_2026-11-12.md).
