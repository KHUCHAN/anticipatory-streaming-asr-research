#!/usr/bin/env python3
"""Render the four follow-up results, keeping incomplete work explicit."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/followups'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    cache=read(RUN/'cache.json'); long=read(RUN/'long_stream.json')
    seeds=read(RUN/'seeds/summary.json'); screen=read(RUN/'two_pass_stability.json')
    lines=['# Follow-up four experiments — 2026-11-11','',
           'This is a follow-up verification of the 100h·17.28M acoustic backbone and the 4.47M predictor. It is not the result of the new 960h·100M models. The original results and learning weights are stored locally in `runs/followups/`.','',
           '| Task | Status |','|---|---|',
           f"| Turn off real text cache | {'Complete' if cache and len(cache['results'])==6 else 'In progress'} |",
           f"| 5/10 minute waveform stream | {'Complete' if long and len(long['results'])==4 else 'In progress'} |",
           f"| two models × additional seed 2 | {'Complete' if seeds else 'Training/decoding in progress'} |",
           f"| 2-pass Screen stabilization | {'Complete' if screen else 'In progress'} |",'',
           '## 1. Text KV/logp cache actually turned off','',
           'Compare released/unreleased caches in the same layer-band2 model. Retain recent caches required by live beams and committed nodes. Keep historical token IDs/trie metadata for output reconstruction, distinct from audio cross-attention memory.','']
    if cache:
        lines += ['test-clean/test-other Chunk 160/320/640ms was applied to the 100 utterance and the same 200 utterance, respectively. It is not interpreted as 600 independent speech samples. We compared the confirmed output, final transcription, and confirmed node log-probability of all chunks on CPU FP32.','',
                  '| split / chunk | Same warrior | Score maximum difference | Unreleased peak KiB | Released peak KiB |',
                  '|---|---:|---:|---:|---:|']
        for key,row in cache['results'].items():
            lines.append(f"| {key} | {row['identical_transcripts']}/{row['utts']} | {row['max_committed_logp_difference']:.3g} | {row['retained']['peak_text_storage_bytes']/1024:.1f} | {row['released']['peak_text_storage_bytes']/1024:.1f} |")
        lines += ['', 'Peak is the sum of retained text KV/logp **backing storage** after chunk processing. Shared storage across projection views counts once. It excludes weights, temporary work tensors and process RSS. Distinguish the 31kB-per-hypothesis calculation from actual complete-beam storage.',
                  '', 'Implementation: [avs/stream.py](../avs/stream.py), `Predictor(..., release_cache=True)`. CLI: `scripts/run_stage3.py --methods H1a --release-cache --hold 0`. Standard stage3 `peak_pred_state_kb` is the old active-node estimate, not total retained storage. This table uses backing-storage measurements from [check_bounded_cache.py](../scripts/check_bounded_cache.py). Original record: [cache.json](../runs/followups/cache.json).','']
    lines += ['## 2. 5/10 minute continuous stream','',
              'Concatenate original test FLAC utterances by ID with 300ms silence. Each 5-minute stream is a prefix of its 10-minute stream, not a real long conversation or independent repeat. Approximate reference times add waveform offsets to original per-utterance CTC alignments.','',
              'Also measure WER with state reset per utterance using identical audio, frontend and model. Mel is precomputed, but encoder caches are actually updated. CPU 1-thread timings use a simulated arrival clock and overlap seed training, so they are not an isolated-device benchmark.','']
    if long:
        lines += ['| Stream | Real seconds | All AR reset → Continuous WER | Each floor 2 reset → Continuous WER | After release text peak KiB | audio KV Final MiB |',
                  '|---|---:|---:|---:|---:|---:|']
        for key,group in long['results'].items():
            full=group['models']['full']; band=group['models']['band2_released']
            lines.append(f"| {key} | {group['audio_seconds']:.1f} | {full['utterance_reset_wer']:.3f} → {full['wer']:.3f} | {band['utterance_reset_wer']:.3f} → {band['wer']:.3f} | {band['peak_text_storage_bytes']/1024:.1f} | {band['final_audio_storage_bytes']/1024**2:.1f} |")
        lines += ['', 'Long-stream peaks are maxima of snapshots every 32 chunks, approximately 10.24 seconds at 320ms, plus first/last snapshots. They are not process peak RSS over every step/workspace. Short cache verification measures retained storage after every chunk.',
                  '', 'Check long-stream transcript equality for released/unreleased band2. Smaller text tensor caches do not stop audio K/V, encoder history or token metadata from growing. **Bounded complete system state is not established.**',
                  '', '[long_stream.json](../runs/followups/long_stream.json) contains RTF, first/last-minute step times, encoder caches, metadata-node counts and temporal snapshots. [Runner](../scripts/long_stream_followup.py).','']
    lines += ['## 3. learning seed repetition','',
              'The original training seed is 2. Additional seeds 3 and 4 each train full and layer-band2 models for 15 epochs from the same Stage 1 decoder. Dropout, batch order and audio cutoffs are paired by seed. Each uses its best dev-NLL checkpoint.','',
              'Evaluate 500 utterances each from dev-clean/test-clean/test-other, sample seed 11, chunks 160/320/640ms, beam 4, prior weight 0.5, hold 0. One frozen acoustic backbone. Scratch initialization or new acoustic-backbone seeds are outside these four follow-ups.','']
    if (RUN/'interrupted/full_s3_before_queue_cleanup/interruption.json').exists():
        lines += ['After queue/session changes, the first full_s3 run stopped at 14 epochs. Preserve interruption records in `runs/followups/interrupted/` and rerun all 15 epochs with identical seed/recipe. Exclude the partial run from repetition counts and performance comparisons. Original partial weights were overwritten during rerun; interruption.json records the actual epoch of the retained snapshot.','']
    if seeds:
        lines += ['| split / chunk | Total AR average WER | Average WER by layer 2 Average WER | Difference average ± seed SD (%p) | Difference range by seed (%p) |',
                  '|---|---:|---:|---:|---:|']
        for key,row in seeds['results'].items():
            low,high=row['difference_range_pp']
            lines.append(f"| {key} | {row['full_mean_wer']:.3f} | {row['band2_mean_wer']:.3f} | {row['mean_difference_pp']:+.3f} ± {row['sd_difference_pp']:.3f} | {low:+.3f} … {high:+.3f} |")
        differences=[row['mean_difference_pp'] for row in seeds['results'].values()]
        lines += ['', f"The seed average difference of 9 conditions is {min(differences):+.3f}…{max(differences):+.3f}%p. Although the observed value is practically close, it does not mean that it passes the equivalence test. In test-clean 640ms, the limits per layer were slightly high for all three seeds. Since the acoustic backbone and decoder initial values ​​were fixed and only the predictor training RNG was repeated, it does not generalize to new backbone and random initialization settings."]
        lines += ['', 'Three-seed means/standard deviations are descriptive. Per-seed paired utterance-bootstrap 95% intervals are saved in [summary.json](../runs/followups/seeds/summary.json). An interval containing zero does not establish equivalence. [Runner](../scripts/repeat_predictors.py) · [Aggregation](../scripts/summarize_seed_repeats.py).','']
    else:
        lines += ['Aggregate with `scripts/summarize_seed_repeats.py` after training/decoding finish. Do not report in-progress values as final.','']
    lines += ['## 4. 2-pass Screen merging and display stabilization','',
              'Use saved events for 300 utterances/split. Count repeat agreement **only after new audio arrives**; a slow-pass reread or forecast event is not another observation. Protect completed slow prefixes and finalize from the last slow output.','',
              'Compare temporal-boundary splicing, agreement over 2/3 observations, hiding the last unfinished word, and token-suffix alignment at splice boundaries. Keep existing gate/threshold 0.87. Select rules on dev-clean, never retune on test.','']
    if screen:
        lines += [f"The dev selection rule is **{screen['chosen_policy']}**all. Mean delay increase 80ms or lower · P95 increase 160ms or lower The total number of corrections for the presence or absence of forecast was minimized. confirm2/3 also left separate diagnostic test results for the requested basic rules.",'',
                  '| split | rule | forecast | final WER | stable P50 / P95 ms | rewrite /100 word |',
                  '|---|---|---|---:|---:|---:|']
        for split,policies in screen['test'].items():
            for name,forecasts in policies.items():
                for fc,row in forecasts.items():
                    if name in ('confirm2','confirm3') and fc=='True': continue
                    lines.append(f"| {split} | {name} | {'Enabled' if fc=='True' else 'None'} | {row['wer']:.2f} | {row['stable_p50']:.1f} / {row['stable_p95']:.1f} | {row['rewritten_words_per_100']:.2f} |")
        lines += ['', 'Final WER is identical because the slow pass owns final transcripts. Revision counts accumulate suffix rewrites, not uniquely revised words. Token alignment reduces clean revisions/tail latency but increases rewriting on other. Repeat agreement trades display delay for fewer revisions. This does not justify making two-pass the main configuration.',
                  '', 'Screen-event replay excludes actual UI and two-encoder contention. Additional alignment CPU time is recorded separately and not added to the replay clock. [two_pass_stability.json](../runs/followups/two_pass_stability.json) · [Analysis](../scripts/stabilize_two_pass.py).','']
    lines += ['## Execution and Remaining Range','',
              '```bash',
              '.venv/bin/python scripts/check_bounded_cache.py',
              '.venv/bin/python scripts/long_stream_followup.py',
              '.venv/bin/python scripts/stabilize_two_pass.py',
              '.venv/bin/python scripts/repeat_predictors.py',
              '.venv/bin/python scripts/summarize_seed_repeats.py',
              '.venv/bin/python scripts/report_followups.py',
              '```','',
              '80ms frame/BPE1024 retraining and FastEmit transducer experiments are not included in the four works approved this time. There remains room for forecasting in a stronger backbone, layer-by-layer 2 predictor without initialization, and voice context limitations. Distinguish between the original sound 960h being ready and 960h model learning being completed.',
              '', "[provenance.json](../runs/followups/provenance.json) records completed checkpoints, code/input SHA-256 and selected epochs. Interrupted partial runs are excluded. No Git commit/push occurred at that study's completion.",'']
    target=ROOT/'docs/followup_experiments_2026-11-11.md'
    target.write_text('\n'.join(lines))
    print(f'WROTE {target.relative_to(ROOT)}')


if __name__=='__main__': main()
