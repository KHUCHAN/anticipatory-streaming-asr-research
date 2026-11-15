#!/usr/bin/env python3
"""Inspect learning curves and generate the source-backed English A2 report."""
import datetime
import json
import os
from zoneinfo import ZoneInfo

from common import ROOT,HERE,OUT,config,sha,write

os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.local/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    cfg=config();histories={}
    for phase in ['mae','kl']:
        path=OUT/phase/'log.jsonl'
        if path.exists():histories[phase]=[r for r in [json.loads(s) for s in path.read_text().splitlines()] if r['event']=='eval']
    histories={k:v for k,v in histories.items() if v}
    summary=dict(status='training in progress',completed_epochs={k:v[-1]['epoch'] for k,v in histories.items()},epoch_history=histories,test_used=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8));colors={'mae':'tab:blue','kl':'tab:orange'}
    for phase,values in histories.items():
        x=[v['epoch']+(cfg['mae_epochs'] if phase=='kl' else 0) for v in values]
        axes[0,0].plot(x,[v['train_mae'] for v in values],'.--',color=colors[phase],label=phase+' train')
        axes[0,0].plot(x,[v['validation']['mae'] for v in values],'.-',color=colors[phase],label=phase+' selection')
        axes[0,0].plot(x,[v['validation']['constant_mae'] for v in values],':',color='gray')
        for split,style in [('dev-clean','-'),('dev-other','--')]:
            axes[0,1].plot(x,[v['selection'][split]['wer_percent'] for v in values],'.'+style,color=colors[phase],label=phase+' '+split)
        if phase=='kl':
            axes[1,0].plot(x,[v['validation']['kl'] for v in values],'.-',label='Selection raw KL')
            axes[1,1].plot(x,[v['gradient']['weighted_kl_to_mae_ratio'] for v in values],'.-',label='Weighted KL / MAE trunk gradient norm')
    labels=[('Representation loss','Normalized state MAE'),('Free-running checkpoint selection','WER (%)'),('Distillation phase only','KL'),('Gradient balance, fixed training probe','Norm ratio')]
    for axis,(title,ylabel) in zip(axes.flat,labels):
        axis.set(title=title,xlabel='Global epoch',ylabel=ylabel);axis.axvline(cfg['mae_epochs']+.5,color='gray',ls=':',lw=1);axis.grid(alpha=.2)
        handles,legend_labels=axis.get_legend_handles_labels()
        if handles:axis.legend(fontsize=8)
    axes[1,1].set_yscale('log');fig.tight_layout();fig.savefig(HERE/'artifacts/learning_curves.png',dpi=160,bbox_inches='tight');plt.close(fig)
    result_path=OUT/'results.json';verification_path=HERE/'artifacts/verification.json'
    if not result_path.exists() or not verification_path.exists():
        write(HERE/'artifacts/summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='epoch_history'}));return
    result=json.loads(result_path.read_text());verification=json.loads(verification_path.read_text())
    runner=[json.loads(s) for s in (OUT/'runner.jsonl').read_text().splitlines()]
    assert runner[-1]['event']=='done' and verification['status']=='verified'
    from baselines import main as baseline_audit
    if not (HERE/'artifacts/state_baselines.json').exists():baseline_audit()
    baselines=json.loads((HERE/'artifacts/state_baselines.json').read_text())
    summary.update(status='completed',results=result,verification=verification)
    write(HERE/'artifacts/summary.json',summary)
    chosen=result['selected_candidate'];q=result['qualification'][chosen]
    verdict='PASS' if q['passes_both_cost_denominators_and_wer'] else 'FAIL'
    selected_score=result['primary'][f'{chosen}/dev-clean']
    conclusion=(f"A2 qualification {verdict}: selected {chosen} checkpoint has dev-clean WER {selected_score['wer_percent']:.3f}%, "
        f"complete cost {100*q['same_run_cost_fraction']:.2f}% of the same-run teacher and {100*q['previous_cost_fraction']:.2f}% of the previous teacher. "
        "Targets are WER <10% and cost <=50% against both denominators.")
    completion=dict(status='completed',completed_local=datetime.datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),elapsed_seconds=runner[-1]['elapsed_seconds'],
        selected=chosen,conclusion=conclusion,source_sha256={name:sha(HERE/name) for name in ['report.py','verify.py','evaluate.py','onnx_backend.py','optimized_inference.py','baselines.py']},
        results=result,verification=verification,state_baselines=baselines)
    write(HERE/'artifacts/completion.json',completion)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3));colors={'teacher':'black','small':'tab:green','mae':'tab:blue','kl':'tab:orange'}
    for axis,split in zip(axes,['dev-clean','dev-other']):
        for name in ['teacher','small',*result['qualification']]:
            score=result['primary'][f'{name}/{split}'];axis.scatter(score['rtf'],score['wer_percent'],s=55,color=colors[name],label=name)
        axis.set(title=split,xlabel='Complete CPU RTF (lower is better)',ylabel='WER % (lower is better)');axis.grid(alpha=.2);axis.legend();axis.margins(.12)
    fig.suptitle('Native ONNX paths; same evaluation audio; small recognizer has shorter lookahead',fontsize=10)
    fig.tight_layout();fig.savefig(HERE/'artifacts/wer_compute.png',dpi=160,bbox_inches='tight');plt.close(fig)
    lines=['# A2 audio-only qualification and small-Zipformer control — 2026-11-15','',conclusion,'',
        f"Completed {sum(v[-1]['epoch'] for v in histories.values())} phase epochs and 800-utterance held-out-role development evaluation in {completion['elapsed_seconds']/3600:.2f} hours from runner start, excluding prior engineering/preparation. One seed; no test data. Local completion: {completion['completed_local']}.",'',
        '## WER and complete cost','',
        '| Split | Path | WER % | State MAE | Total ms/chunk | RTF | RTF + file IO |',
        '|---|---|---:|---:|---:|---:|---:|']
    for split in ['dev-clean','dev-other']:
        for name in ['teacher','small',*result['qualification']]:
            s=result['primary'][f'{name}/{split}'];mae='N/A' if s['state_mae'] is None else f"{s['state_mae']:.4f}"
            lines.append(f"| {split} | {name} | {s['wer_percent']:.3f} | {mae} | {s['total_ms_per_chunk']:.3f} | {s['rtf']:.5f} | {s['rtf_with_audio_io']:.5f} |")
    lines+=['','Teacher is the existing public 66.11M Zipformer; small is the public English export named 20M. A2 has 9,597,696 newly trained frontend/head parameters and shares the frozen teacher prediction network/joiner. Public recognizers and the local 100h student have different training budgets. Small-model states are not teacher-compatible, so state MAE is not compared for that model.','',
        '## Qualification','',
        '| Phase | Clean WER <10% | Cost / same-run teacher | Cost / previous teacher | All conditions pass |',
        '|---|---|---:|---:|---|']
    for name,qphase in result['qualification'].items():
        lines.append(f"| {name} | {qphase['wer_below_10']} | {100*qphase['same_run_cost_fraction']:.2f}% | {100*qphase['previous_cost_fraction']:.2f}% | {qphase['passes_both_cost_denominators_and_wer']} |")
    lines+=['',f"Candidate selection used only the model-selection role: {json.dumps(result['model_selection'])}. The chosen phase is **{chosen}**. The separate final role did not select checkpoints or the phase.",'',
        'The cost condition uses complete CPU RTF on identical dev-clean audio and must hold against both denominators. The previous teacher pipeline used a PyTorch decoder; this run uses the native ONNX decoder. Passing 10%/50% is an entry gate for later B/C studies, not teacher-level WER parity or proof of a new ASR method. No FGN training or adaptive encoder omission was performed.','',
        '## Paired WER differences','',
        '| Split | Comparison | Delta pp | Paired 95% CI |','|---|---|---:|---|']
    for key,v in result['paired_comparisons'].items():
        split,pair=key.split('/');lines.append(f"| {split} | {pair} | {v['wer_difference_pp']:+.3f} | [{v['ci95_pp'][0]:+.3f}, {v['ci95_pp'][1]:+.3f}] |")
    lines+=['','Intervals resample utterances jointly and are conditional on these selected checkpoints. They do not capture seed/optimizer/checkpoint-selection variability. No equivalence is inferred from an interval containing zero.','',
        '## Cost components','',result['timing_scope'],'',
        '| Split | Path | Fbank | Frontend + head | Decoder | Other | Total |',
        '|---|---|---:|---:|---:|---:|---:|']
    for split in ['dev-clean','dev-other']:
        for name in ['teacher','small',*result['qualification']]:
            s=result['primary'][f'{name}/{split}'];c=s['components_ms_per_chunk']
            values=[s['feature_ms_per_chunk'],c['audio']+c['head'],c['decoder'],s['other_overhead_ms_per_chunk'],s['total_ms_per_chunk']]
            lines.append(f"| {split} | {name} | "+' | '.join(f'{x:.3f}' for x in values)+' |')
    lines+=['','All component values are ms/chunk, measured on Apple M5 Pro 24GiB, CPU batch 1 / one thread. Two sequential repetitions reverse path order after excluded warmup; repeated transcripts match. Neural paths all use FP32 ONNX. A2 output denormalization, all cache handoffs and native token decoding are counted. Reference-state scoring and disk IO are outside the primary timer; IO-inclusive RTF is shown separately. Model loading is a one-time excluded startup cost.','',
        '## Architecture, training and checks','',
        '- Initial causal stride 4, twelve width-256 residual blocks and a joint chunk output head. All eight outputs can use the final arrived feature. Batch/stream outputs, future-chunk isolation and stride phase passed preflight. Tiny fitting used a disposable model, not a warm start.',
        '- MAE: 12 epochs; best selected by fixed-window representation MAE. KL: up to 24 epochs after the representation gate, initialized from MAE-best; one-time train-only trunk-gradient calibration and two-epoch coefficient ramp. Final KL checkpoint selected by free-running selection-role clean WER. MAE-only and last checkpoints remain preserved.',
        '- All 28,539 train-clean-100 utterances contribute each epoch. Eight-chunk beginning/end/random windows directly supervise utterance starts. Causal history is available as input; future teacher states and native contexts supervise losses only. No reference transcript or teacher state is an inference input.',
        '- The shared teacher distribution is native greedy-context distillation, not new reference-supervised transducer training. WER is obtained from independently generated transcripts and is not fixed by construction.',
        '- The initial untrained PyTorch path was approximately 10.8ms/chunk. Equivalent first/step ONNX graphs were prepared during MAE training, before held-out collection, without changed weights or quantization. Initial untrained ONNX timing was approximately 1.3ms/chunk during GPU training, not a qualification result. Final exported checkpoints have their own numerical and real-audio token checks.',
        '- Existing teacher and small adapters match independent sherpa-onnx outputs on eight raw-audio decodes. The small model has a native 39-frame window, teacher/A2 45; all hop 32 and emit eight frames. Its available input window is 60ms shorter. Native endpoints/tail padding can yield slightly different chunk counts. Compare RTF per original audio duration, not only per chunk.',
        '- The same project dev roles were used before this study. Current selection/evaluation roles are speaker-disjoint, but this is not an untouched external test. One seed and one bounded recipe cannot settle all cheap audio encoders or FGN prediction value.',
        '- Training source/input hashes, exact epoch/optimizer-step counts, selected checkpoints, unchanged decoder/teacher files, FP32 export identities, raw coverage, repeated transcripts and timing sums passed verification. No live microphone, user-perceived display latency, adaptive fallback, mobile power or long-stream recovery evaluation is claimed.','',
        f"PyTorch/ONNX real-audio token checks on eight model-selection utterances per checkpoint: {json.dumps({phase:check['all_checked_tokens_match'] for phase,check in result['backend_checks'].items()})}. The per-checkpoint random-stream numeric tolerance is 2e-4; exact token checks and numerical closeness are recorded separately.",'',
        f"Same-set constant-state MAE diagnostics: {json.dumps(baselines['scores'])}. The median estimate uses 65,536 training frames, not exact population medians. Error reductions are not recovered-information percentages.",'',
        '![Learning curves](../experiments/predict_correct_a2_v1/artifacts/learning_curves.png)','',
        '![WER and compute](../experiments/predict_correct_a2_v1/artifacts/wer_compute.png)','',
        '[Protocol](../experiments/predict_correct_a2_v1/README.md) · [Aggregate results](../experiments/predict_correct_a2_v1/artifacts/results.json) · [Prior review](predict_correct_review_2026-11-14.md)']
    (ROOT/'docs/predict_correct_a2_2026-11-15.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='completed',conclusion=conclusion)))


if __name__=='__main__':main()
