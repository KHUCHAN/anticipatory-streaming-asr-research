"""Generate auditable tables and scientific figures from completed results."""
import csv,hashlib,json,platform,shutil,time
import importlib.metadata
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import ROOT,OUT

LABELS={'native':'Native probability','full':'Prior full-context AR','band2':'Prior layer-band2 AR',
        'state':'Recurrent FGN: seed37','state38':'Recurrent FGN: seed38','ensemble':'Recurrent FGN: two models','reset':'No old-state memory',
        'deterministic':'No functional noise','single_step':'One-step training only','crps_only':'CRPS only: no token guidance','persistence':'Persistence: replay arrived chunk'}
ORDER=list(LABELS)


def main():
    cfg=json.loads((ROOT/'experiments/fgn_state_v1/config.json').read_text())
    backbone_cfg=json.loads((ROOT/'configs/zipformer_baseline.json').read_text())
    result=json.loads((OUT/'results.json').read_text());metrics=result['results']
    trajectory=json.loads((OUT/'trajectory_diagnostics.json').read_text())
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    histories={};training=[]
    for variant in cfg['variants']+['crps_only_s37']:
        records=[json.loads(s) for s in (OUT/variant/'log.jsonl').read_text().splitlines()]
        epochs=[r for r in records if r['event']=='eval'];histories[variant]=epochs
        ck=torch.load(OUT/variant/'best.pt',map_location='cpu',weights_only=False)
        training.append(dict(variant=variant,completed_epochs=len(epochs),selected_epoch=ck['epoch']+1,
                             selected_horizon=next(r['horizon'] for r in epochs if r['epoch']==ck['epoch']),
                             selected_validation=ck['validation'],selection_criterion='state CRPS only' if variant.startswith('crps_only') else 'state CRPS +0.1 token NLL',params=sum(v.numel() for v in ck['model'].values()),
                             initial_state_sha256=ck['initial_state_sha256'],checkpoint_sha256=hashlib.sha256((OUT/variant/'best.pt').read_bytes()).hexdigest()))
    assert len({r['initial_state_sha256'] for r in training if r['variant'].endswith('s37')})==1
    with (OUT/'learning_curves.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['variant','epoch','training_horizon','steps','validation_loss_h4','validation_crps_h4','validation_token_nll_h4','epoch_seconds'])
        for variant,epochs in histories.items():
            for r in epochs:w.writerow([variant,r['epoch']+1,r['horizon'],r['step'],r['validation']['state_crps']+.1*r['validation']['token_nll'],r['validation']['state_crps'],r['validation']['token_nll'],r['epoch_seconds']])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for variant,epochs in histories.items():
        x=[r['epoch']+1 for r in epochs]
        for ax,key,title in zip(axes,['loss','state_crps','token_nll'],['Combined validation loss','Continuous-state fair CRPS','Frozen-decoder token NLL']):
            y=[r['validation']['state_crps']+.1*r['validation']['token_nll'] if key=='loss' else r['validation'][key] for r in epochs]
            ax.plot(x,y,label=variant,lw=1.7);ax.set(xlabel='Epoch',title=title);ax.grid(alpha=.18)
    axes[0].legend(fontsize=8);fig.suptitle('New state-model learning curves: fixed H4 validation, identical 256 utterances')
    fig.tight_layout();fig.savefig(figdir/'learning_curves.png');fig.savefig(figdir/'learning_curves.svg');plt.close(fig)
    colors=['#667085','#98a2b3','#344054','#2563eb','#a855f7','#7c3aed','#0891b2','#d97706','#16a34a','#db2777','#be123c']
    fig,axes=plt.subplots(2,2,figsize=(15,8),sharex=True)
    for col,split in enumerate(cfg['evaluation_splits']):
        values=[metrics[f'{split}/{k}/gate'] for k in ORDER];gain=np.array([v['stable_mean_gain_ms'] for v in values])
        ci=np.array([v['stable_mean_gain_ci95_ms'] for v in values])
        axes[0,col].bar(range(len(ORDER)),gain,color=colors)
        axes[0,col].errorbar(range(len(ORDER)),gain,yerr=np.maximum(0,np.stack([gain-ci[:,0],ci[:,1]-gain])),fmt='none',color='black',capsize=3)
        axes[0,col].set(title=split+' — stable display gain',ylabel='Mean gain over eager native display (ms)')
        axes[1,col].bar(range(len(ORDER)),[v['wrong_forecasts_per_100_native_tokens'] for v in values],color=colors)
        axes[1,col].axhline(.5,color='#b42318',ls='--',lw=1,label='Dev calibration target; test not constrained')
        axes[1,col].set(ylabel='False forecasts / 100 native tokens',xticks=range(len(ORDER)),xticklabels=[LABELS[k] for k in ORDER])
        axes[1,col].tick_params(axis='x',rotation=38);axes[1,col].legend(fontsize=8)
        for ax in axes[:,col]:ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.suptitle('Frozen Zipformer, 320ms, full test sets — gain must be read together with test risk')
    fig.tight_layout();fig.savefig(figdir/'display_gain_and_risk.png');fig.savefig(figdir/'display_gain_and_risk.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,split in zip(axes,cfg['evaluation_splits']):
        for kind,color in zip(ORDER,colors):
            points=result['risk_gain_operating_points'][f'{split}/{kind}']
            ax.plot([p['wrong_forecasts_per_100_native_tokens'] for p in points],[p['stable_mean_gain_ms'] for p in points],'.-',color=color,label=LABELS[kind],alpha=.85)
            main=next(p for p in points if p['dev_budget']==.5)
            ax.scatter([main['wrong_forecasts_per_100_native_tokens']],[main['stable_mean_gain_ms']],s=45,color=color)
        ax.set(title=split,xlabel='Actual test false forecasts / 100 native tokens',ylabel='Stable mean display gain (ms)');ax.grid(alpha=.18)
    axes[0].legend(fontsize=7);fig.suptitle('Five operating points chosen only on dev; no test threshold selection')
    fig.tight_layout();fig.savefig(figdir/'risk_gain_curves.png');fig.savefig(figdir/'risk_gain_curves.svg');plt.close(fig)
    table=[]
    for split in cfg['evaluation_splits']:
        for k in ORDER:table.append(dict(split=split,model=k,**metrics[f'{split}/{k}/gate']))
    with (OUT/'metrics.csv').open('w') as f:
        fields=['split','model','utterances','wer_percent','stable_mean_gain_ms','wrong_forecasts_per_100_native_tokens',
                'retracted_per_100_native_tokens','reference_correct_and_kept_precision','extra_ms_per_chunk','extra_forecaster_rtf',
                'confirmed_state_peak_bytes','speculative_state_peak_bytes']
        fields+=['audio_projection_cache_peak_bytes','retained_forecaster_context_peak_bytes']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(table)
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'experiments/fgn_state_v1').glob('*')) if p.is_file()}
    budget=json.loads((OUT/'budget.json').read_text())
    provenance=dict(training=training,source_sha256=source,torch=torch.__version__,platform=platform.platform(),
                    dependencies={n:importlib.metadata.version(n) for n in ['torch','numpy','onnx','onnxruntime','sentencepiece','matplotlib']},
                    training_backend='MPS; one sequential training process',inference_backend='CPU; one Torch thread per worker, six collection workers',
                    collection_provenance=json.loads((OUT/'events_test_provenance.json').read_text()),
                    finished_unix=time.time(),wall_elapsed_seconds=time.time()-budget['started_unix'],wall_budget_seconds=21600,
                    model_definition='6323968 trainable parameters per forecaster; frozen 66.11M backbone is additional.',
                    bridge_max_difference=json.loads((OUT/'checks.json').read_text())['frozen_joiner_max_absolute_difference'])
    (OUT/'final_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    lines=['<!-- generated by experiments/fgn_state_v1/report.py -->','# FGN continuous state pilot — 2026-11-12','',
      'With the acoustic model frozen, this pilot adds **persistent summarized state, continuous-state feedback, multi-step backpropagation, shared within-step noise and independent-model ensembles**. It is not an exact reproduction of weather FGN.','',
      '## Experimental conditions','',
      '- Backbone: Public LibriSpeech960h Streaming Zipformer66.11M, BPE500, chunk320ms, Fixed weights and final transcription.',
      '- Additional predictors: 6.324M/ model, train only train-clean-100, plan 16epoch. Actual completion/selection epochs are recorded in the table below.',
      '- Status: Remembers the last two voice states (each 8×512) and the 256 dimension GRU. Confirmation memory is updated with the actual arrival voice, and prediction branches are copied and used.',
      '- Re-enter the predicted continuous speech states and memories into the next step. The future answer state is used only for loss calculation and learns up to 8 steps without detachment.',
      '- At each future step, a new 32-dimensional noise is extracted and shared to all frame/conditional normalization layers at that step. Not fixing a single noise for the entire future.',
      '- Loss: continuous state fairCRPS +0.1×fixed RNN-T native-token distillationNLL. Does not directly optimize full-text transcription WER.',
      '- CRPS-only additional control is token loss weight 0, and checkpoint is also selected only with H4 CRPS. Check whether the loss and gradient are exactly the same even if the token target/prefix is \u200b\u200bchanged. Other models are selected as CRPS+0.1NLL, so there are differences in selection criteria.',
      '- CRPS is a probability prediction score of continuous speech feature values, not WER or word hit rate. Distinguish between the gain in this score and the gain in actual display quality.',
      '- Verification H4 ignites 256 model-selection. Separated from gate-fit1297 and calibration1414. test-clean2620/test-other2939 total evaluation.',
      '- The combined validation score of the learning curve and CSV is commonly calculated as CRPS+0.1NLL for all models. The actual training/checkpoint selection loss of CRPS-only is CRPS only and is distinguished from this common display score.',
      '- In dev, the standard is fixed below false introduced forecasts/100 native tokens=0.5. There is no guarantee that the test will have the same error rate.',
      '- The same 11 characteristics are provided to all new gates: prefix8 voting and 3 based on decoder probability. Deterministic models can also use probability-based certainty. The characteristics of the existing control group gate8/9 were fixed as is.',
      '- Persistence control group without learning: Copy the last arrived voice chunk as is to the future state and decode it. Examining the effect of rereading already seen voice states with a new prefix.',
      '- The existing native/full/band2 gate and threshold were reused as the original, and the display delay and false-forecast rate were accurately reproduced.','',
      '| Model | Complete Select epoch | Learn when select epoch | H | Verify CRPS | Verify token NLL |','|---|---:|---:|---:|---:|---:|']
    for t in training:lines.append(f"| {t['variant']} | {t['completed_epochs']} | {t['selected_epoch']} | {t['selected_horizon']} | {t['selected_validation']['state_crps']:.4f} | {t['selected_validation']['token_nll']:.4f} |")
    lines+=['','## Early display of a single token for the entire test','',
      'Display gain is the **stable-time difference for correct final words** relative to native eager output. WER remains 3.0584% clean / 7.8196% other in every configuration because final transcripts are fixed.','',
      'Token is a subword unit of BPE500 and is different from a word. The display gain is calculated in units of words, and the false report/retraction rate is calculated in units of native BPE tokens.','',
      '| split | model/learning gate | average display gain ms [95% CI] | false forecast/100 native token | withdrawal/100 native token | Additional CPU ms/chunk |','|---|---|---:|---:|---:|---:|']
    for split in cfg['evaluation_splits']:
        for kind in ORDER:
            v=metrics[f'{split}/{kind}/gate'];ci=v['stable_mean_gain_ci95_ms']
            lines.append(f"| {split} | {LABELS[kind]} | {v['stable_mean_gain_ms']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {v['wrong_forecasts_per_100_native_tokens']:.3f} | {v['retracted_per_100_native_tokens']:.3f} | {v['extra_ms_per_chunk']:.3f} |")
    lines+=['','## Differences from existing floor restriction predictors','',
      'Methods share a development risk budget, but test risk can differ. Latency differences alone do not establish a superior risk–latency curve. Intervals are paired utterance bootstrap; speaker-level intervals are stored separately in results.json.','',
      '| split | New configuration | Additional gain ms compared to band2 [95% CI] | test False report rate difference/100 token [95% CI] |','|---|---|---:|---:|']
    for split in cfg['evaluation_splits']:
        for kind in ['state','state38','ensemble','reset','deterministic','single_step','crps_only','persistence']:
            p=result['paired_comparisons'][f'{split}/{kind}-minus-band2'];ci=p['ci95_ms']
            risk=metrics[f'{split}/{kind}/gate']['wrong_forecasts_per_100_native_tokens']-metrics[f'{split}/band2/gate']['wrong_forecasts_per_100_native_tokens']
            rci=p['false_forecast_delta_ci95']
            lines.append(f"| {split} | {kind} | {p['delta_gain_ms']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | {risk:+.3f} [{rci[0]:+.3f}, {rci[1]:+.3f}] |")
    lines+=['','## Comparison of FGN elements','',
      'The difference between state and reset is persistent memory that contains old information, the difference between deterministic and shared noise, and the difference between single_step examines multi-step learning. Ensemble is an effect using two independent models, so it is distinguished from a single model effect. They are all control groups that started with the same initial value and sample order of seed37, and the second model in the ensemble is seed38.','',
      '| split | configuration − state | average gain difference ms [95% CI] |','|---|---|---:|']
    for split in cfg['evaluation_splits']:
        for kind in ['state38','ensemble','reset','deterministic','single_step','crps_only','persistence']:
            p=result['paired_comparisons'][f'{split}/{kind}-minus-state'];ci=p['ci95_ms']
            lines.append(f"| {split} | {kind} | {p['delta_gain_ms']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] |")
    lines+=['','## Maintain context size','',
      'Values sum logical retained tensor bytes, numel×element_size, at their maximum. Existing AR counts text and audio-projection KV; the new model counts two recent acoustic states and memory. Weights, allocator, temporary activations, backbone and offline caches are excluded; this is not complete process peak RAM.','',
      '| Configuration | Maximum maintenance context KiB of two test splits |','|---|---:|']
    for kind in ['full','band2','state','ensemble','persistence']:
        peak=max(metrics[f'{s}/{kind}/gate']['retained_forecaster_context_peak_bytes'] for s in cfg['evaluation_splits'])
        lines.append(f'| {LABELS[kind]} | {peak/1024:.2f} |')
    primary=metrics['test-clean/state/gate']
    lines+=['',f"The single state model's speculative branch state uses {primary['speculative_state_peak_bytes']/1024:.0f}KiB, a separate prediction state output tensor, is {primary['predicted_state_tensor_peak_bytes']/1024:.0f}It's KiB. This temporary space also does not increase depending on the length if the number of paths and rollout length are fixed, but is displayed separately from the maintenance context table above."]
    if (OUT/'storage_checks.json').exists():
        storage=json.loads((OUT/'storage_checks.json').read_text())['results']
        lines+=['',f"Unique tensor storage was checked over 10 minutes of observation: state {storage['state']['storage_peak_bytes']/1024:.0f}KiB, ensemble {storage['ensemble']['storage_peak_bytes']/1024:.0f}KiB, persistence {storage['persistence']['storage_peak_bytes']/1024:.0f}KiB. Unlike logical 33KiB, reset is due to the extra storage held by the view. {storage['reset']['storage_peak_bytes']/1024:.0f}It's KiB. It is not a process RSS measurement."]
    lines+=['','## Future Path Diagnosis','',
      'Auxiliary diagnostic generates 16 paths over 4 steps, 1.28 seconds, on model-selection utterances. Path hits mean agreement with future native tokens, not reference-word accuracy. Per-frame member shuffling preserves marginals and CRPS but breaks temporal coupling. Lower joint energy is better; this is not independent test evidence.','',
      '| Configuration | joint energy, coherent / shuffled ↓ | marginal CRPS ↓ | 2-token hit@16 % | 3-token hit@16 % | State standard deviation |','|---|---:|---:|---:|---:|---:|']
    for kind in ['state','state38','ensemble','reset','deterministic','single_step','crps_only','persistence']:
        c=trajectory['summary'][kind+'/coherent'];s=trajectory['summary'][kind+'/shuffled']
        lines.append(f"| {kind} | {c['energy_score']:.5f} / {s['energy_score']:.5f} | {c['marginal_crps']:.5f} | {c['path2_hit_any_percent']:.2f} | {c['path3_hit_any_percent']:.2f} | {c['sample_std']:.5f} |")
    lines+=['','coherent−shuffled difference in joint energy (coherent dominance if negative), paired utterance95%CI:','']
    for kind in ['state','state38','ensemble','reset','deterministic','single_step','crps_only','persistence']:
        c=trajectory['summary'][kind+'/coherent'];ci=c['coherent_minus_shuffled_energy_ci95']
        lines.append(f"- {kind}: {c['coherent_minus_shuffled_energy_mean']:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}].")
    lines+=['','## Tests and Limits','',
      '- Future input disturbance invariance, gradient of future state = 0, gradient leading to the next state through the predicted first state, and old input invariance of the reset control group were examined.',
      '- The maximum difference between the original ONNX decoder/joiner and the differentiable fixed implementation was approximately 2.6e−6/7.6e−6, respectively.',
      '- We checked that the streaming memory and the overall causal GRU calculation are consistent, that the prediction branch does not modify the deterministic memory, and that the retention tensor is fixed in 10 minutes worth of observations.',
      '- The above 10 test is a state size test. The display quality of the new gate has not been verified over a long period of actual continuous voice.',
      '- The maintenance state is a single model 33792byte (33KiB), and the ensemble is a shared two voice state + two memories 34816byte (34KiB). This counts only the live tensor of the new predictor and excludes backbone/weight/allocator/temporary activation/offline cache.',
      '- Fixed-size memory is lossy compression. This means that there is a transmission channel for past information, but it does not mean that all of the past is perfectly remembered.',
      "- reset only removes GRU paths from the distant past of the new predictor. The last two encoder states themselves retain historical information contained in the fixed Zipformer's causal cache. This is not a control group that changed the entire ASR to a model that only views the original sound of the recent 640ms.",
      '- CPU prediction time includes the cost of state update, path creation, and verification decoding. Concurrent execution/resource contention and independent phoneme alignment with the actual microphone/UI and backbone were not measured.',
      '- Average stable indication gain and irreversible commit latency are different indicators. Because only some words were advanced, the median gain for each word was also recorded separately.',
      '- Multi-level sequential states and paths are used for creation, learning, and diagnosis. The screen policy aligned with the existing method only displays 1 unconfirmed tokens at a time, and the user effect of displaying multiple tokens simultaneously is not what this claim is about.',
      '- The structure, objective function, and number of samples are different from existing predictors. Element-specific causal analysis takes precedence over new model internal controls, and multiple comparisons are exploratory.',
      '- Since the main model has a token auxiliary loss, this alone does not verify the marginals-only claim in the original text. In the separate CRPS-only control group, learning and selection of a new predictor is done only with marginal distribution scores, but the encoder expression used is already pre-trained with ASR.',
      '- The new control group has the same number of optimizer steps. Multi-step learning requires more computation as the number of steps, so it is not a comparison of the same FLOPs budget.',
      '- The learning loss of the fixed decoder uses the correct answer native future prefix as a condition. This is a critic condition for loss and is not included in the state transition input. Actual path decoding re-enters its prediction token.',
      '- The risk-benefit curve is at 5 points where the threshold of the dev budget 0.1/0.25/0.5/1/2 is fixed before test collection. The threshold was not optimized in the test.',
      '- The learning seeds of the component control group are 1, and the full state model is 2. The bootstrap CI is the test sample variation and is not the CI of the training seed variation.',
      f"- From preparation to final report {provenance['wall_elapsed_seconds']/3600:.2f}hours. Completed within the user-specified 6-hour limit: {provenance['wall_elapsed_seconds']<=21600}.",
      '- The above time is the new predictor implementation, learning, and evaluation time using the existing encoder feature cache. It does not include the original learning of the public backbone and the preparation of the previous source → feature cache.',
      '- Existing models, results, and documents were not overwritten, and git commit/push was not performed.','',
      '## file','',
      '- `experiments/fgn_state_v1/README.md`: Order of design, original text correspondence, and reproduction.',
      '- `runs/fgn_state_v1/{variant}/{best,last}.pt`, `log.jsonl`: Weights and learning records.',
      '- `runs/fgn_state_v1/{checks,runtime_checks,final_provenance}.json`: verification·selection epoch·hash.',
      '- `runs/fgn_state_v1/gates.json`: dev-only weights and fixed thresholds.',
      '- `runs/fgn_state_v1/{results,paired_word_gains,trajectory_diagnostics}.json`: Original statistics.',
      '- `runs/fgn_state_v1/{learning_curves,metrics}.csv`: Original numbers for tables and curves.',
      '- `runs/fgn_state_v1/figures/`: Performance illustration with learning curve and baseline.','',
      '- `experiments/fgn_state_v1/artifacts/`: Lightweight results/original figures/figure copies available for review in git. Large weight/speech feature cache/per-utterance events maintained in local runs.','',
      'FGN design source: [Original paper Sections2.1–2.4](https://arxiv.org/html/2506.10772v1).',
      f"Backbone source: [Zipformer paper]({backbone_cfg['paper']}), [Official recipe and results]({backbone_cfg['upstream_results']}), [original checkpoint]({backbone_cfg['original_checkpoint']}). ONNX distribution URL and settings were saved in artifacts/backbone_config.json.",'']
    interpretation=OUT/'interpretation.md'
    if interpretation.exists():lines[3:3]=interpretation.read_text().splitlines()+['']
    report='\n'.join(lines);(OUT/'summary.md').write_text(report)
    target=ROOT/'docs/fgn_state_pilot_2026-11-12.md'
    if target.exists():assert target.read_text().startswith('<!-- generated by experiments/fgn_state_v1/report.py -->'),'Existing peer-owned report; refusing overwrite'
    figures='\n## painting\n\n![Learning curve](../experiments/fgn_state_v1/artifacts/learning_curves.png)\n\n![Indicating gain and false forecast including existing baseline](../experiments/fgn_state_v1/artifacts/display_gain_and_risk.png)\n\n![risk–gain point fixed by dev](../experiments/fgn_state_v1/artifacts/risk_gain_curves.png)\n'
    target.write_text(report+figures)
    archive=ROOT/'experiments/fgn_state_v1/artifacts';archive.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'configs/zipformer_baseline.json',archive/'backbone_config.json')
    for name in ['summary.md','interpretation.md','results.json','gates.json','metrics.csv','learning_curves.csv','final_provenance.json',
                 'checks.json','runtime_checks.json','checks_crps_only.json','storage_checks.json','native_decode_audit.json','analysis_plan_before_test_v8.json']:
        if (OUT/name).exists():shutil.copy2(OUT/name,archive/name)
    for p in figdir.glob('*'):
        if p.name!='training_progress_only.png':shutil.copy2(p,archive/p.name)
    (archive/'trajectory_summary.json').write_text(json.dumps({k:v for k,v in trajectory.items() if k!='records'},indent=2)+'\n')
    print(json.dumps(dict(report=str(target),wall_hours=provenance['wall_elapsed_seconds']/3600,figures=str(figdir),reviewable_artifacts=str(archive))),flush=True)


if __name__=='__main__':main()
