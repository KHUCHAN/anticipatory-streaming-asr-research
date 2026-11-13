# Experiment index

| Directory | Status | Purpose | Public records |
|---|---|---|---|
| [fgn_state_v1](fgn_state_v1/README.md) | Training and full test evaluation complete | Independent recurrent state forecaster; noise, memory, rollout and ensemble controls | Configuration, code, aggregate results, diagnostics and figures in `artifacts/` |
| [band_fgn_v1](band_fgn_v1/README.md) | Six continuation runs and full test evaluation complete | Actual layer-band2 AR plus FGN hybrid; two AR depths | Configuration, code, aggregate results, verification and figures in `artifacts/` |
| [compute_profile_v1](compute_profile_v1/README.md) | Small development profile complete | Native encoder/prediction-network/joiner compute breakdown | Initial and reproduced timing records in `artifacts/` |
| [fgn_capacity_v1](fgn_capacity_v1/README.md) | Completed | Actual generator 6.32M / 10.54M / 18.96M; 33 continuation epochs total, all met the overfit rule at 11 | Verified fixed-case train/dev curves and [capacity report](../docs/fgn_capacity_2026-11-12.md); one seed, no test evaluation |
| [fgn_capacity_long_v1](fgn_capacity_long_v1/README.md) | Completed | Restored epoch-11 models and optimizers; 16 extra epochs each, through 27; overfit stopping disabled | [Extended results](../docs/fgn_capacity_long_2026-11-13.md): similar final fitting across sizes, higher dev NLL, SiLU settings verified; no test use |

The [predict–correct encoder acceleration plan](../docs/predict_correct_acceleration_plan_2026-11-12.md) is a proposal. There is no completed residual-corrector or encoder-skipping experiment to reproduce yet.

Training weights, raw speech, feature caches and per-utterance training/evaluation outputs remain in ignored local directories. The repository includes source code, aggregate numerical evidence and source provenance. A fresh checkout requires preparing the documented public inputs before training or evaluating models.
