#!/usr/bin/env python3
"""Write the final three-way correction feasibility report in English."""
import hashlib
import json
from common import ROOT,HERE,OUT,config,write


def main():
    cfg=config();r=json.loads((OUT/'results.json').read_text());v=json.loads((HERE/'artifacts/verification.json').read_text())
    from baseline_states import main as baseline_audit
    if not (HERE/'artifacts/state_baselines.json').exists():baseline_audit()
    baselines=json.loads((HERE/'artifacts/state_baselines.json').read_text())
    from coverage import main as coverage_audit
    if not (HERE/'artifacts/training_coverage.json').exists():coverage_audit()
    coverage=json.loads((HERE/'artifacts/training_coverage.json').read_text())
    runner=[json.loads(s) for s in (OUT/'runner.jsonl').read_text().splitlines()]
    assert runner[-1]['event']=='done' and v['status']=='verified'
    result=dict(status='completed',elapsed_seconds=runner[-1]['elapsed_seconds'],results=r,verification=v,state_baselines=baselines,training_coverage=coverage,
        reporting_source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['report.py','evaluate.py','verify.py','analyze.py','finalize.py']})
    passed=[name for name,point in r['operating_points'].items() if point['meets_wer_tolerance'] and point['meets_compute_saving']]
    useful_prior=[split for split,point in r['decisions'].items() if point['b_wer_ci_excludes_zero_in_beneficial_direction'] and point['b_observed_improvement_meets_0_5pp']]
    quality_conclusion=('No tested cheap path retains the full encoder\'s accuracy while meeting the compute-saving target. '
                        if not passed else 'Some tested cheap-path operating points meet the declared quality and compute targets. ')
    conclusion=(quality_conclusion+f"B meets the declared practical-plus-paired-CI WER improvement criterion over A on {len(useful_prior)}/2 splits. "
                f"{len(passed)}/6 cheap-path operating points meet both the full-encoder WER tolerance and compute-saving targets. "
                "These separate checks distinguish prior usefulness within this small student from accuracy-preserving encoder acceleration.")
    result['conclusion']=conclusion
    audit_path=HERE/'artifacts/loss_gradient_audit.json'
    if audit_path.exists():result['post_hoc_loss_gradient_audit']=json.loads(audit_path.read_text())
    write(HERE/'artifacts/completion.json',result)
    lines=['# Fbank prediction-correction three-way pilot — 2026-11-14','',
        conclusion,'',
        f"Completed A/B/C training and held-out dev decoding in {result['elapsed_seconds']/3600:.2f} hours from runner start, excluding prior fbank extraction and engineering checks. Frozen native decoder; own corrected-state feedback, no teacher-state inference input. No test data used.",'',
        '**Interpretation qualification:** the failed cheap audio path prevents this pilot from settling the general value of FGN prediction in a viable recognizer. Fixed-checkpoint paired comparisons remain descriptive evidence within this weak student. See the [post-hoc review, loss-gradient audit and proposed A-only qualification](predict_correct_review_2026-11-14.md). No new A2 training has run.','',
        '## WER and complete inference cost','',
        '| Split | Path | WER % | Normalized state MAE | Total ms/chunk | RTF | RTF including audio file IO |',
        '|---|---|---:|---:|---:|---:|---:|']
    for split in ['dev-clean','dev-other']:
        for variant in ['full_encoder','A','B','C']:
            x=r['primary'][f'{variant}/{split}']
            lines.append(f"| {split} | {variant} | {x['wer_percent']:.3f} | {x['state_mae']:.4f} | {x['total_ms_per_chunk']:.3f} | {x['rtf']:.4f} | {x['total_with_audio_io_rtf']:.4f} |")
    lines+=['','All paths use the same nominal 320ms chunk schedule, 45-frame feature availability and baseline tail padding; no 160/640ms sweep was run here. A uses fbank + causal conv only. B adds one frozen deterministic FGN forecast, based on corrected history. C adds the previous corrected state directly. Common audio/head initialization; prior projections start at zero. Frontend parameters are A 745,600 / B 7,135,104 (including frozen FGN) / C 811,136. These are not matched total-parameter or iso-compute systems.','',
        'Same-set constant-mean MAE baselines: '+', '.join(f"{split} {value['mean_prediction_mae']:.4f}" for split,value in baselines['scores'].items())+'. MAE reductions are not recovered-information percentages.','',
        '## Paired WER comparisons','',
        '| Split | Difference | WER delta, percentage points | Paired 95% CI |',
        '|---|---|---:|---|']
    for key,x in r['paired_comparisons'].items():
        split,pair=key.split('/');lines.append(f"| {split} | {pair} | {x['wer_difference_pp']:+.3f} | [{x['ci95_pp'][0]:+.3f}, {x['ci95_pp'][1]:+.3f}] |")
    lines+=['','Negative B-minus-A deltas favor prediction. Intervals resample utterances jointly; they do not measure seed variance. Lack of significance does not prove equivalence. A practical B-versus-A improvement threshold of 0.5pp and a full-encoder WER-degradation tolerance of 0.5pp were declared before full training.','',
        '## Interpretation','']
    for split in ['dev-clean','dev-other']:
        b=r['primary'][f'B/{split}'];a=r['primary'][f'A/{split}'];c=r['primary'][f'C/{split}']
        lines.append(f"- {split}: B changes WER by {b['wer_percent']-a['wer_percent']:+.3f}pp versus A and {b['wer_percent']-c['wer_percent']:+.3f}pp versus C. Its full inference cost is {b['rtf']/a['rtf']:.2f}x A and {b['rtf']/c['rtf']:.2f}x C. Use the paired intervals above to assess those quality differences.")
    if all(r['paired_comparisons'][f'{split}/B-minus-C']['ci95_pp'][0]<=0<=r['paired_comparisons'][f'{split}/B-minus-C']['ci95_pp'][1] for split in ['dev-clean','dev-other']):
        lines+=['','B\'s additional forecasting benefit over simply retaining the previous corrected state is not established on either split: both B-minus-C intervals include zero. This does not prove equivalence. It matters because B is the more expensive path.']
    if not passed:
        lines+=['','The cheaper runtime comes with a large recognition failure. It does not establish useful encoder-skipping acceleration. Improve and diagnose the cheap audio reconstruction/distillation path before spending more training on a larger frozen state predictor. This result is limited to the fixed small-conv recipe, its causal sampling/context constraints and the frozen-prior training contract; one pilot cannot settle every prediction-correction architecture.']
    lines+=['',
        '## Operating-point checks','']
    for split,d in r['decisions'].items():
        lines.append(f"- {split}: B improves on A by the declared practical amount: {d['b_observed_improvement_meets_0_5pp']}; beneficial paired CI excludes zero: {d['b_wer_ci_excludes_zero_in_beneficial_direction']}; B/full-encoder speedup {d['b_speedup_vs_full_encoder']:.2f}x; within full-encoder WER tolerance: {d['b_meets_teacher_wer_tolerance']}; B/A compute ratio {d['b_total_time_relative_to_a']:.2f}.")
    lines+=['','## Cost accounting','',r['timing_scope'],'',
        '| Split | Path | Fbank | Audio network | FGN forecast | Correction head | Feedback | Decoder | Other | Total |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in ['dev-clean','dev-other']:
        for variant in ['full_encoder','A','B','C']:
            x=r['primary'][f'{variant}/{split}'];c=x['components_ms_per_chunk']
            values=[x['feature_ms_per_chunk'],c['audio'],c['prediction'],c['correction'],c['feedback'],c['decoder'],x['other_overhead_ms_per_chunk'],x['total_ms_per_chunk']]
            lines.append(f"| {split} | {variant} | "+' | '.join(f'{value:.3f}' for value in values)+' |')
    lines+=['','All component values are milliseconds per chunk, averaged over timing repetitions on Apple M5 Pro with 24GiB unified memory. Roundoff may affect displayed sums. The full encoder uses its official ONNX CPU export; cheap conv/FGN and the shared native decoder use PyTorch CPU. Measured speed ratios describe these implementations on this Mac, not backend-independent FLOP savings.','',
        'Two sequential repetitions use reversed path order after warmup, and repeat transcripts match exactly. Primary total runtime includes fbank extraction, normalization, causal conv/cache updates, FGN transition, correction, corrected-memory update, decoder and array/Python overhead. Input file IO is reported separately. Component and repeat timings are in the aggregate JSON. Chunk P50/P95 are computation times, not microphone/display latency.','',
        '## Checkpoint selection','',
        '| Path | Epochs | Selected epoch | Selection WER % |',
        '|---|---:|---:|---:|']
    for model in v['models']:
        lines.append(f"| {model['variant']} | {model['epochs']} | {model['best_epoch']} | {model['best_selection_wer']:.3f} |")
    lines+=['','The public [curve data](../experiments/predict_correct_v1/artifacts/summary.json) retain every epoch\'s selection WER, representation/distillation losses and wall time. Model selection uses only its declared role; the final evaluation role does not choose checkpoints.','',
        '## Training and limitations','',
        '- All 28,539 train-clean-100 utterances are visited each epoch. Every epoch samples a new six-chunk window per utterance; two corrected-only burn-in chunks and four supervised chunks. Convolution also sees its bounded past-audio receptive field. This is not exhaustive dense-cutoff training.',
        f"- Schedule audit: {coverage['supervised_chunk_visits']:,} supervised chunk visits cover {coverage['unique_supervised_chunks']:,}/{coverage['available_chunks']:,} distinct chunks ({100*coverage['unique_coverage_fraction']:.1f}%). The first two chunks are burn-in only and receive no direct loss in this recipe. The same limitation applies to all three paths.",
        '- Train 16 epochs per variant with AdamW; select checkpoints using free-running complete-utterance WER on 128 model-selection dev utterances. Final evaluation uses a separate 400 dev-clean + 400 dev-other utterances from a speaker-disjoint dev role.',
        '- Targets are cached full-encoder states and frozen-decoder distributions under offline teacher contexts. Those contexts and states supervise losses only. Inference uses generated token contexts and corrected feedback; this is a distillation pilot, not new reference-supervised transducer training.',
        '- B predictor weights are frozen and their hashes preserved. Its speculative memory is discarded; confirmed memory is updated from corrected outputs. Feedback is detached between training chunks. Evaluation resets once per utterance and runs the complete sequence.',
        '- The frozen FGN was originally trained from full-encoder history. It is not retrained jointly on this corrector\'s generated trajectories. This pilot therefore also includes the challenge of feeding corrected rather than original encoder states to that fixed prior.',
        '- The teacher was publicly trained on 960h; the new correction modules use local 100h. B additionally reuses a previously trained FGN checkpoint. Equal new correction epochs do not equal matched lifetime training budgets or total model capacity.',
        '- The cheap path fully replaces encoder outputs here. No full encoder runs behind A/B/C to maintain a hidden oracle cache. Adaptive fallback, replay recovery, long uninterrupted streams, noisy/unexpected-speech interventions and mobile-device measurements remain unimplemented.',
        '- The cheap conv is strictly causal at sampled positions 16, 20, ..., 44 of the initial feature window. It does not reproduce the full encoder\'s within-chunk receptive fields. Equal chunk-boundary audio availability is not matched per-output context.',
        '- Priors condition the audio hidden representation before a dense output head. No explicit forecast-plus-residual output constraint or sparse coordinate recomputation is implemented.',
        '- One seed and one small conv recipe do not settle all possible correctors or FGN architectures. Better than A is not enough to claim useful ASR acceleration if WER remains far worse than the full encoder.',
        '- All eight timed decodes completed before a NumPy-bool aggregate JSON serialization error. A separate serialization adapter recovered aggregation without changing the frozen evaluator, any checkpoint or raw result. No training or inference was rerun; the original failure and recovery remain in the runner log.',
        '- Initial tests verify exact causal convolution streaming, future-input invariance, sequence/runtime feedback agreement, live corrected C states, nonzero prior-adapter gradients and a frozen B predictor. All epochs, selected checkpoints, input hashes and final audio coverage pass verification.','',
        '![Learning curves](../experiments/predict_correct_v1/artifacts/learning_curves.png)','',
        '![WER versus total inference compute](../experiments/predict_correct_v1/artifacts/wer_compute.png)','',
        '[Protocol](../experiments/predict_correct_v1/README.md) · [Aggregate results](../experiments/predict_correct_v1/artifacts/results.json) · [Current goal](predict_correct_acceleration_plan_2026-11-12.md)']
    (ROOT/'docs/predict_correct_pilot_2026-11-14.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='completed',hours=result['elapsed_seconds']/3600)))


if __name__=='__main__':main()
