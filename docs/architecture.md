# AVS Architecture

Current proposed extension: [predict–correct encoder acceleration](predict_correct_acceleration_plan_2026-11-12.md). The raw-audio corrector and conditional encoder runtime are not implemented. Architectures below preserve completed historical studies.

Written: 2026-11-10. Code: [`avs/model.py`](../avs/model.py) · Methodology: [three-stage plan](research_plan_3stage.md)

This document describes historical 100h/17.28M AVS, distinct from [current conclusions](conclusion_and_next_steps.md) and the [960h/114M proposal](librispeech960_benchmark_plan.md). Bounded per-layer acoustic K/V does not automatically bound the candidate trie, token history or all predictor state.

The [960h architecture proposal](librispeech960_benchmark_plan.md) describes bounded-state recurrent forecasting and acoustic verification. Current 3-token AR experiments bound text, not all acoustic history read by the forecaster. Acoustic-backbone and predictor capacities are separate choices.

## At a glance

```text
16 kHz speech
  │ 80-bin log-mel (25 ms / 10 ms, no centering, fixed training statistics)
  ▼
Conv2d ×2 (k3, s2, without padding) → 40 ms frame, fixed lookahead 6 mel frame
  ▼
Chunk-causal Conformer ×12            d=256, head 4, FF 1024
  ├ RoPE self-attention: current chunk plus 64 left-context frames
  ├ causal depthwise conv: kernel 15, past direction only
  └ feed-forward
  ▼
encoder frame ──┬─► CTC head (Linear → 500): Acoustic verification/alignment
                 └─► Decoder ×4 (cross-attention): Stage 1 auxiliary loss → Stage 2 predictor
```

Total 17.28M parameters: encoder + CTC + decoder.

## Design decisions and reasons

| Decision | Reason |
|---|---|
| CTC verifier | Frame-synchronous rescoring against observed audio is inexpensive. CTC loss was verified on MPS with torch 2.15. |
| Auxiliary attention decoder | 100 Encoder quality is better than CTC alone on time scale, and becomes the initial value of Stage 2 audio-aware predictor. |
Implementing streaming with | chunk mask | frame i only sees the end of its chunk. The result of forwarding the entire utterance once along with the mask is the same as chunk-unit streaming, so learning, evaluation, and sorting are performed with the same code. |
| Left context of 64 frames | Per-layer attention state is finite independently of utterance length (H4). Effective context is larger across 12 layers. |
| conv subsampling without padding | The number of mel frames required by encoder frame j is closed to `4j+7`. The available time for each frame can be accurately calculated. |
| RoPE (relative position) | There is no absolute position, so only the position offset needs to be passed in the chunk-unit increment calculation. |
| Only uses LayerNorm | There is no normalization using time base statistics so future frames are not mixed with past output. |
| dynamic chunk learning | chunk 4/8/16/32 and offline are evaluated as one model. Comparisons are controlled because models are not trained separately for each delay condition. |
| unigram 500 token | 100 usual choice of time scale. The prediction horizon 1–4 token roughly corresponds to the word 0.5–2. |

## causality

`scripts/check_causality.py` disturbs the mel frame behind the chunk boundary and measures the change in encoder output before the boundary. In the random initialization model, both chunks 4/8/16/32 confirmed that past output difference 0, future output difference > 0.5. The offline mode is intentionally non-causal and is only used to generate sort labels. The learned weight (`runs/stage1/avg.pt`) also passed.

## Predictor (Stage 2)

Decoder's cross-attention receives a memory mask for each token location. Token u only sees encoder frames before the cutoff assigned to it. The always visible learnable slot `mem0` is defined even if the cutoff is 0 because it indicates "no voice yet".

- B3 audio-aware: Same as above decoder.
- B2 text-only: Same structure as `cross=False` minus cross-attention.

## Candidate storage structure (Stage 3)

Hypotheses and speculative candidates are prefix-trie nodes containing parent pointers, per-position layer K/V and probability vectors. Common prefixes share state. A candidate-count budget does not automatically bound all historical paths or caches; measure retained storage. The short-context `P*8` field counts token IDs, not complete measured state.

## Learning Settings (Stage 1)

AdamW (β 0.9/0.98, weight decay 0.01), peak lr 1e-3, warmup 4,000 step after cosine, grad clip 5, dropout 0.1, label smoothing 0.1, SpecAugment (frequency 2×27, time 3×40). Arrangement is by length, bucket, 40,000 mel frame including padding. MPS float16 autocast. 40 epoch.

Initial small autocast benchmark: approximately 700–800 times real-time throughput and 8.5GB. Full training logs: 40 epochs, 36,880 steps, summed epoch time about 4.862 hours, maximum recorded MPS driver allocation 13.69GiB. Initial benchmark memory is not full-run peak. `mem_gb` includes driver/cache allocation, not complete RAM or continuously measured peak. Initial no-autocast benchmark: 571 times real time, 15.4GB; an 80,000-frame batch exceeded memory.
## Turn off actual layered text caching — subsequent implementation

Follow-up `Predictor(decoder, hist=2, release_cache=True)` reads current/prior-two-position K/V per layer and releases `kv`/`logp` outside recent live-beam/committed-node needs after each chunk. Select with `scripts/run_stage3.py --methods H1a --release-cache --hold 0`. Default remains unreleased for historical reproducibility.

Output/storage verification covers non-speculative H1a, beam 4, hold 0. Do not extend the same quality/speed guarantees to all speculative-tree reuse.

`Decoder3.state_usage()` counts the tensor's backing storage without duplication and distinguishes the number of text KV/logp, shared audio KV, and token nodes. The existing `peak_pred_state_kb` is an active-node estimation field, so it is not actually used as the entire memory. The actual measurement, output matching, and long stream results follow [the subsequent four experiments](followup_experiments_2026-11-11.md).

Four-layer self-attention sees at most 9 raw text tokens. Three live cache positions do not mean only three original input tokens matter. Cross-attention retains all arrived audio K/V and token parent/child objects reconstruct transcripts. Total memory/chunk work are not constant; `Node.prefix`, commitment/pruning metadata traversal and growing audio context still cost work.

## Strong-backbone adapter — 2026-11-12

Separate from historical 17.28M AVS, an actual forecasting study on frozen official LibriSpeech 960h Streaming Zipformer 66.11M is complete. Native greedy decoding at 320ms owns the final transcript; our forecaster receives only arrived encoder outputs and the recognized token prefix.

```text
Current output of native encoder (512 dimension)
    → Linear(512,256) + LayerNorm
    → 4 floor / 4head / FF1024 Transformer AR preview
       full: full past self-attention
       band2: Current + past 2 location on each floor, maintaining only actual KV/logp 3 nodes
    → Next token distribution of BPE 500
    → Confidence gate learned and calibrated by a separate dev
    → Provisional mark / Verification/withdrawal with next native release
```

Both forecasters have 4,602,868 parameters and share random initialization; each trains on full train-clean-100 for 15 epochs. Four-layer band2 has at most 9 raw text tokens of receptive field. It reads full arrived encoder history, appending adapter/cross-attention K/V only for new frames. Text cache is 29.9KiB; audio KV and metadata still grow. Reusing past activations does not establish reuse of speculative future computation.

Also compare existing PN/joiner current-frame nonblank forecasts without an extra forecaster. Gate positives require reference BPE/complete-word correctness and native acceptance; inference features contain neither future nor reference information. Final WER equals native by design; evaluate provisional display gain, errors and added cost.

Settings: [zipformer_ours.json](../configs/zipformer_ours.json); model: [zipformer_forecaster.py](../avs/zipformer_forecaster.py); incremental forecasting: [collect_zipformer_forecasts.py](../scripts/collect_zipformer_forecasts.py); results: [report](zipformer_ours_2026-11-12.md). First strong-backbone single-seed pilot; no new 960h acoustic training, FastEmit, noise or rollout fine-tuning in that study.



## Floor-limited AR + FGN joint learning — 2026-11-12

[The new combined experiment ](band_fgn_hybrid_2026-11-12.md) does not replace the existing band2 forecaster, but actually maintains it.

```text
Actually arrived encoder features ───────────────────────→ Confirmed AR text KV
         └→ Remember the last two states + GRU256
              → StateFGN with shared noise conditions → Predict future encoder features
              → Generate the next state by feeding back the predicted state and memory
Observed audio cross KV + predicted future features → Temporary last-token AR branch
                                   → Next BPE distribution → Learning gate → Tentative display
```

Confirmed parent text KV sees actual arrived audio only; predicted audio enters a temporary branch. Each AR layer sees current plus previous 2 tokens: effective context 9 for four layers and 25 for twelve. Both retain 3 text KV positions. Audio cross KV grows, unlike the earlier state-only predictor's 33KiB total retained predictor context.

AR4.60M/13.03M starts from the same existing band2 weight. The additional 8 layer of the large AR is the initial identity. The FGN generator compares the existing state_s37 weights to noise0/ probability samples, both sizes being equal to 6.324M,. The loss is the average token CE+0.1 fair CRPS. The CE is backpropagated to the FGN via the predicted features. Rollout is at stage 1→2→4, token holding sees the first 2 prediction chunks. This is not an experiment to inject noise into the entire normalization of the AR body or to expand the FGN generator itself.

We learned 6 configurations 8epoch each and evaluated the entire test of the actual recognition prefix. This combination did not improve the display performance of the existing band2+gate, and the CPU cost increased. Model, gate, error, and overfitting analysis follow the above result document.
