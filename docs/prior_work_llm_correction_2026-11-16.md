# Prior work: LLM candidate selection, ASR correction and streaming cost

Reviewed 2026-11-16. **Research only:** this update collects and inspects ten additional papers, P51-P60, without running models, invoking an LLM inference API, launching training, migrating temporary experiments or changing the selected project architecture. Original experimental results are unchanged. The current catalog contains 60 papers; the earlier [17-paper future-context review](prior_work_predict_refine_2026-11-16.md) remains a separate topic.

**Using an LLM to choose ASR candidates or correct their text is established prior work.** What matters for the fast-text goal is whether a specific causal implementation improves reference-correct display quality after its extra computation and revisions are counted. Most strong offline correction results do not answer that question. StreamCorrect is a direct recent streaming prior; Delayed Fusion is a useful computation-scheduling prior whose published evaluation still uses a full-context encoder.

[Paper folder](../notes/llm-correction/README.md) · [Source/evidence manifest](../references/reviews/llm_correction_2026-11-16.json) · [Current paper index](../papers/INDEX.md)

## 1. Different meanings of "use the LLM to choose"

| Operation | Allowed output | Main advantage | Main limit / closest papers |
|---|---|---|---|
| Score and rerank a fixed N-best list | One complete existing candidate | Preserves the acoustic search space | Cannot beat the best-reference candidate in that same fixed list; P51/P52 |
| Generate a correction and map it back to the list | One existing candidate | Limits unrestricted rewriting | The mapping and selection can still choose a worse candidate; P52 |
| Choose tokens from aligned disagreements | A composition of several candidates | Focuses the LLM on ambiguous spans | Compositions can create a new sentence; option-order bias matters; P54 |
| Generate an extra candidate, then rerank | One candidate from an expanded list | Can improve candidate coverage | Expensive generation and acoustic rescoring remain; P55 |
| Unrestricted generative correction | A new text sequence | Can recover words absent from the list | Can remove or replace correctly recognized speech; P53/P56 |
| Acoustic/LLM fusion | A new sequence from combined decisions | Keeps linguistic decisions connected to audio | Vocabulary matching, calibration and both models' costs matter; P58/P60 |
| Chunk-level speech-language correction | A corrected current segment with a committed prefix | Directly addresses streaming error propagation | Adds runtime and can affect future recognition; P59 |
| Scheduled first-pass LM scoring | A beam-search path influenced by intermittent LLM scores | Reduces LM calls relative to dense fusion | Not necessarily faster than plain ASR; P57 |

An oracle using reference text is a diagnostic bound, not a deployable selector. A method that beats a fixed-list oracle must have changed the allowed output space, candidate set or evaluation conditions; it has not surpassed a mathematical bound for selecting from the unchanged list.

## 2. Collected studies and their comparison role

All numerical examples below are source-reported. They were not reproduced locally. WER, CER, datasets, training data and timing conditions differ, so rows are not a leaderboard.

| Paper | Method and evidence | Streaming and cost boundary |
|---|---|---|
| [P51 N-best T5](../notes/llm-correction/P51-nbest-t5.md), Interspeech 2023 | T5 with N-best/lattice-constrained decoding. LibriSpeech test clean/other WER 2.88/7.06 to 2.54/6.31% for 10-best constrained correction; lattice 2.53/6.27% | Utterance correction; no measured causal display-delay result found |
| [P52 Can Generative LLMs Perform ASR Error Correction?](../notes/llm-correction/P52-chatgpt-asr-correction.md), 2023 preprint | ChatGPT free correction, direct selection and closest-list mapping. LibriSpeech test-other transducer WER 6.90 to 6.24% with one-shot closest mapping | Historical API snapshot; full-utterance WER, not streaming timing |
| [P53 HyPoradise](../notes/llm-correction/P53-hyporadise.md), NeurIPS 2023 | Public N-best/reference pairs, LoRA and prompted correction. Strong LibriSpeech baseline 1.8/3.7%; H2T-LoRA 1.7/3.8% | Offline benchmark; large gains on another domain do not imply gains on this strong recognizer |
| [P54 ClozeGER / Listen Again](../notes/llm-correction/P54-clozeger.md), Findings ACL 2024 | Multiple-choice gaps, calibration, optional source audio and a second correction stage. CommonVoice 14.4 to 8.5% WER with the complete SpeechGPT branch | Final improvement includes post-processing; not a pure selector or measured causal display system |
| [P55 ProGRes](../notes/llm-correction/P55-progres.md), SLT 2024 | Expand the N-best list with an instruction LLM, then combine ASR and LLM scores. Table 1 GPT-4 condition gives 40.84/9.32% for two CommonVoice ASR systems | Uses a heuristic ASR score for the new candidate; unresolved 9.32 versus 9.82 table/prose discrepancy; no streaming timing found |
| [P56 Task-Activating Prompting](../notes/llm-correction/P56-task-activating.md), ASRU 2023 | Task setup and demonstrations for generative correction/rescoring; Table 3 WSJ 11.87 to 8.72%, ATIS 8.82 to 6.39% WER for frozen InstructGPT with TAP | Second-pass utterance evaluation; prompting gains do not establish cheap partial output |
| [P57 Delayed Fusion](../notes/llm-correction/P57-delayed-fusion.md), ICASSP 2025 | Apply LLM scores after beam pruning and at selected intervals; supports vocabulary mismatch | Algorithm can stream, but published experiments explicitly use full-utterance encoder context. Includes measured decoding RTF |
| [P58 SALSA](../notes/llm-correction/P58-salsa.md), Interspeech 2024 | Couple ASR decoder states into the LLM through adapters; avoid a separate N-best correction pass | Synchronous decoding is not causal audio streaming. A100 RTF: plain Whisper 0.42, SALSA 0.64, Whispering-LLaMA 1.30 |
| [P59 StreamCorrect](../notes/llm-correction/P59-streamcorrect.md), EMNLP Industry 2026 | Correct actual streaming chunks using audio, top-k hypotheses and the confirmed prefix; freeze the recognizer | Direct streaming study. Improves Mandarin/Cantonese CER while adding measured first/last-token latency on an H20 |
| [P60 It's Never Too Late / UADF](../notes/llm-correction/P60-late-acoustic-fusion.md), ICLR 2024 | Calibrated uncertainty controls acoustic/LLM late fusion. ATIS/WSJ GER 1.61/2.83 to 1.24/2.47% WER | Acoustic decoder is adapted to a shared vocabulary; offline results do not demonstrate free streaming correction |

For P51-P56, an end-to-end streaming word-delay/RTF evaluation was not found in the inspected text; this is recorded as missing evidence, not zero overhead. P60 likewise does not establish a measured streaming latency gain. Exact source versions, pages and implementation status are in each note.

## 3. Findings that change interpretation

### A better language model can still make recognition worse

HyPoradise's Appendix Table 6 reports a strong LibriSpeech baseline of 1.8/3.7% WER. LoRA correction changes it to 1.7/3.8%; full fine-tuning gives 1.8/3.9%. P52 also reports Whisper TED WER worsening from 3.89 to 5.84% with unrestricted zero-shot ChatGPT correction. These are counterexamples to a universal "LLM correction helps" claim. They do not predict the outcome for our model. [HyPoradise](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6492267465a7ac507be1f9fd1174e78d-Abstract-Datasets_and_Benchmarks.html), [ChatGPT correction](https://arxiv.org/abs/2307.04172)

Constraining output and preserving acoustic scores are established ways to limit this freedom. They still need reference-based accuracy checks: being a plausible member of the beam is not proof of correctness. ClozeGER and UADF are particularly relevant when language-model preference conflicts with what was actually spoken. [N-best T5](https://www.isca-archive.org/interspeech_2023/ma23e_interspeech.html), [ClozeGER](https://aclanthology.org/2024.findings-acl.37/), [UADF](https://arxiv.org/abs/2402.05457)

### Efficient LLM integration does not mean faster than ASR alone

Delayed Fusion's LibriHeavy CTC-prefix example changes clean/other WER from 3.47/6.41 to 3.05/5.68% using OpenLLaMA-3B, while decoding RTF rises from 0.018 to 0.115. The timing subset is 200 utterances on a Xeon CPU plus V100. Table II varies scoring intervals to expose an accuracy-cost frontier. Fewer LLM calls are useful relative to alternative LM integrations, but this example still adds substantial work to plain ASR. Section IV-A explicitly states that encoding uses full utterance context. [Delayed Fusion, Tables I-II](https://arxiv.org/abs/2501.09258)

SALSA similarly reports RTF 0.64 versus Whispering-LLaMA 1.30, but plain Whisper is 0.42. Its experiment transcribes isolated utterances. The word "synchronous" describes cooperation between token decoders, not proof of bounded-lookahead audio processing. [SALSA, Section 5](https://www.isca-archive.org/interspeech_2024/mittal24_interspeech.html)

### There is already a directly relevant streaming correction paper

StreamCorrect's Table 2, AISHELL-1 with 100ms chunks, reports SimulStreaming CER 27.3% versus 11.7% after correction. First-token latency rises from 158 to 207ms; last-token latency rises from 200 to 284ms. At 500ms, CER is 16.3 to 9.8% and those latencies are 158 to 219ms and 213 to 295ms. The study uses NVIDIA H20, Mandarin/Cantonese and a speech-language corrector with approximately 1.92B total parameters despite its "1B" model name. [StreamCorrect, Table 2 and deployment section](https://aclanthology.org/2026.emnlp-industry.142/)

This is evidence for an accuracy-latency trade-off, not direct evidence of faster first display at fixed chunk size. Its first/last-token endpoints also differ from per-word acoustic-end-to-stable-display p50/p90. Its self-generated errors use paired speech/transcripts and boundary recovery; "without extra chunk timestamp annotation" should not be rewritten as "needs no reference transcripts."

### Several source claims need their original qualifications

| Source issue | How it is recorded |
|---|---|
| ProGRes gives the generated candidate the original best ASR score | Record the heuristic; do not describe it as a newly computed acoustic likelihood for that candidate |
| ProGRes Table 1 says 9.32% for ASR2, prose says 9.82% | Retain both with locators; use the table when describing that row, without silently resolving the discrepancy |
| HyPoradise publisher landing abstract says more than 316k pairs, collected PDF says more than 334k | Keep counts version-specific; downstream results refer to the collected PDF |
| StreamCorrect abstract says up to 39% relative reduction, while selected Table 2 rows imply larger changes | Quote exact table cells and comparison conditions; do not treat the headline as a universal upper bound |
| Generative correction beats an N-best oracle | Distinguish an expanded or composed output space from strict selection within the original list |

These are source-reading caveats, not local reproductions or corrections to the authors' code.

## 4. Teacher emission-regularization provenance: read-only check

The local [baseline configuration](../configs/zipformer_baseline.json) identifies the 66.11M teacher with the public `Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17` model. Its model card links to icefall PR #1058. At the pinned model-repository revision `37cb5606808f3d5e55a3fc73554bdf757d82465a` the distributed launch script requests causal LibriSpeech training, 30 epochs and FP16; it has no FastEmit or delay-penalty option. The rank-0 training log's initial parameter dictionary also contains no recorded coefficient for either. [Model card](https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17), [pinned launch script](https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17/blob/37cb5606808f3d5e55a3fc73554bdf757d82465a/exp/train.sh)

The source revision is fully recorded in the [research manifest](../references/reviews/llm_correction_2026-11-16.json); the shortened identifier in prose is only a label. The launch-script link below the prose is replaced by the complete hash in the final source record. The accompanying PR describes delay-penalty support as a future addition to that recipe. This supports treating the deployed teacher as **not documented to use explicit early-emission regularization**, rather than assuming that any k2 transducer has it. [PR #1058](https://github.com/k2-fsa/icefall/pull/1058)

The log reports training from `0d931583-dirty`. Its exact modified training code was not recovered. Absence from a launch script and logged options therefore does **not** certify the absence of an unlogged custom loss term. The narrow finding is a provenance limitation; no measured regularization effect follows from it. No checkpoint was loaded, converted or evaluated in this check.

## 5. What this establishes for the project

These are interpretations of the literature, not instructions to start experiments.

- LLM candidate selection, free correction, disagreement-focused prompting, acoustic/linguistic uncertainty fusion and streaming correction all have direct precedents. Adding an LLM or an uncertainty score is not sufficient by itself for a novelty claim.
- Strong final WER improvements and fast provisional text are separate claims. The relevant comparison for fast text must restrict all inputs to audio/text actually available at the display time and count candidate extraction, tokenization, scoring/generation, verification and revisions.
- The supplied temporary result of dev-other minus 0.62 percentage points remains user-reported here. Its split, test independence, comparison risk and significance were not rechecked; a source-paper result cannot validate that local measurement.
- A prior review's unfilled comparison cell is not a claim that no prior work exists. This targeted search does not establish that encoder skipping, or any other broad family, is unstudied.

One nuance to the latency discussion: advancing a word relative to a baseline's emission is a valid descriptive difference. For the same reference-correct word with acoustic endpoint `a`, `(t_baseline - a) - (t_new - a) = t_baseline - t_new`. An independent word alignment is needed to report absolute reference-anchored latency and identify the correct corresponding words. Agreement with the baseline's later transcript alone cannot establish those conditions. Report the quantity actually measured rather than rejecting every baseline-relative difference or relabeling it as absolute latency.

## 6. Search and execution boundary

The ten collected PDFs contain 107 pages. Core methods and selected result tables were inspected; selected tables were rendered for visual checking. The [manifest](../references/reviews/llm_correction_2026-11-16.json) records versions, SHA-256 hashes, queries, source discrepancies, repository availability and the teacher-source check. Original PDFs and extracted text remain local-only; public notes are original summaries.

Supplementary primary sources include the [2025 journal article by Ma and colleagues](https://ieeexplore.ieee.org/document/10930744/) and [2026 decoder-only streaming LLM-ASR](https://arxiv.org/abs/2601.22779). Their bibliographic/abstract records were checked; they are not additional full-text-reviewed catalog entries. The latter is an integrated ASR architecture, not simply an external N-best reranker. This review is targeted rather than exhaustive and does not rank results from different datasets or certify novelty.

Only paper/source retrieval, text inspection, document generation and repository checks were performed. No proposed filler comparison, ASR decoding evaluation, forced-alignment collection, model training or architecture change was executed.
