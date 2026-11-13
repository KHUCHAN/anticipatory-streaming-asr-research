# Official source of implementation, model, and data

Reviewed: 2026-11-10. Links and source relationships checked; no installation, training or model-performance reproduction in that review.

Statements above describe verification during external-source collection. Later local AVS training and corrections are in [history](research_plan_3stage.md); current assessment is in [conclusions](conclusion_and_next_steps.md). Distinguish actual external-checkpoint execution from local-model training.

| Source | Purpose | Verification status |
|---|---|---|
| [google-deepmind/weathernext](https://github.com/google-deepmind/weathernext) | WN2/FGN and previous GraphCast/GenCast | README provides model code/weight bucket/demo |
| [state-spaces/mamba](https://github.com/state-spaces/mamba) | Mamba | official implementation; Check Mac kernel performance separately |
| [state-spaces/s4](https://github.com/state-spaces/s4) | S4 | State space implementation |
| [xi-j/Mamba-ASR](https://github.com/xi-j/Mamba-ASR) | Speech Slytherin / ConMamba | Implementation directly linked to the original text |
| [k2-fsa/icefall](https://github.com/k2-fsa/icefall) | Zipformer and ASR recipes | Implementation released by the paper authors |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | streaming Model distribution/inference | 20M Check ONNX model guidance |
| [NVIDIA NeMo's current speech repository](https://github.com/NVIDIA-NeMo/Speech) | FastConformer·transducer series | The NVIDIA/NeMo link in the original paper currently goes to this address |
| [openai/whisper](https://github.com/openai/whisper) | External accuracy/decoding reference | Open code/model information |
| [Factor Graph NN](https://github.com/zzhang1987/Factor-Graph-Neural-Network) | Same name FGNN classification | JMLR connection implementation |
| [Finite Gaussian FGN](https://github.com/grezesf/FGN---Research) | Same-name FGN classification | Thesis connection implementation |

SpecInfer's original link `flexflow/FlexFlow` now moves to `flexflow/flexflow-train`. Therefore, it is not assumed that the original serving artifact was preserved in the same state. `GlenHGHUANG/Deep_graph_random_process` linked by the DGP original text was HTTP 404 in this test. The original text of the paper was obtained from PMLR.

Items for which the official implementation and checkpoints for SSR, SpecASR, ParaASR, Speculative Listening, and LC-BiMamba could not be confirmed were left unconfirmed in the paper notes. The existence of the paper and the disclosure of the code and pre-learning weights are in different states.

Raw datasets and model weights were not arbitrarily substituted from other repositories. Use the [catalog](../references/catalog.json) for recoverable PDF links and [registry](../data/datasets.json) for original dataset sources and access status.
