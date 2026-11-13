import csv,json,os,platform,shutil,sys,time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from common import ROOT,OUT,HERE,config,sha,write

def main():
    cfg=config();art=HERE/'artifacts';art.mkdir(exist_ok=True)
    write(art/'environment.json',dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),
        logical_cpu_count=os.cpu_count(),mps_available=torch.backends.mps.is_available(),
        packages={p:version(p) for p in ['torch','numpy','sentencepiece','matplotlib']},
        evaluation_workers=4,training_precision='FP32',training_device='MPS',inference_device='CPU, one thread per worker'))
    metrics=json.loads((OUT/'results.json').read_text());curves=[];training={}
    for variant in cfg['variants']:
        log=[json.loads(s) for s in (OUT/variant/'log.jsonl').read_text().splitlines()]
        starts=[r for r in log if r.get('event')=='start'];ev=[r for r in log if r.get('event')=='eval']
        assert len(ev)==cfg['epochs']+1 and any(r.get('event')=='done' for r in log)
        ck=torch.load(OUT/variant/'best.pt',map_location='cpu',weights_only=False)
        selected=ck['epoch']+1
        training[variant]=dict(parameters=starts[-1]['parameters'],ar_parameters=starts[-1]['ar_parameters'],completed_epochs=cfg['epochs'],selected_epoch=selected,selected_dev=ck['dev'],selected_train_probe=ck['train_probe'],last_dev=ev[-1]['dev'],last_train_probe=ev[-1]['train_probe'],best_sha256=sha(OUT/variant/'best.pt'))
        for r in ev:curves.append(dict(variant=variant,epoch=r['epoch'],train_nll=r['train_probe']['nll'],dev_nll=r['dev']['nll'],dev_clean_nll=r['dev']['by_split']['dev-clean']['nll'],dev_other_nll=r['dev']['by_split']['dev-other']['nll'],gap_dev_clean_minus_train=r['generalization_gap_nll'],dev_hit1=r['dev']['hit1'],dev_crps=r['dev']['crps'],seconds=r['seconds']))
    with (art/'learning_curves.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    comparisons={}
    for size in ['small','large']:
        a=training[size+'_fgn'];b=training[size+'_det']
        comparisons[size]=dict(noisy_minus_deterministic_selected_dev_nll=a['selected_dev']['nll']-b['selected_dev']['nll'],
            noisy_minus_deterministic_last_dev_clean_nll=a['last_dev']['by_split']['dev-clean']['nll']-b['last_dev']['by_split']['dev-clean']['nll'],
            noisy_minus_deterministic_last_train_nll=a['last_train_probe']['nll']-b['last_train_probe']['nll'],
            scope='Descriptive one-seed continuation comparison, not a causal or universal overfitting claim.')
    write(art/'training_summary.json',dict(training=training,generalization=comparisons))
    exported=[]
    for key,r in metrics['results'].items():
        split,kind,policy=key.split('/')
        if policy!='gate':continue
        ci=r['stable_mean_gain_ci95_ms']
        exported.append(dict(split=split,model=kind,utterances=r['utterances'],wer_percent=r['wer_percent'],
            mean_display_gain_ms=r['stable_mean_gain_ms'],gain_ci95_low_ms=ci[0],gain_ci95_high_ms=ci[1],
            median_word_gain_ms=r['median_per_word_gain_ms'],advanced_correct_words_percent=r['correct_words_advanced_percent'],
            wrong_forecasts_per_100_native_tokens=r['wrong_forecasts_per_100_native_tokens'],
            retracted_per_100_native_tokens=r['retracted_per_100_native_tokens'],extra_ms_per_chunk=r['extra_ms_per_chunk'],
            retained_context_peak_bytes=r['retained_forecaster_context_peak_bytes']))
    with (art/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(exported[0]));w.writeheader();w.writerows(exported)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':150})
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for ax,v in zip(axes.flat,cfg['variants']):
        rr=[r for r in curves if r['variant']==v];x=[r['epoch'] for r in rr]
        for field,label in [('train_nll','train probe'),('dev_clean_nll','dev-clean'),('dev_other_nll','dev-other')]:ax.plot(x,[r[field] for r in rr],label=label,marker='.',linewidth=1.5)
        ax.axvline(training[v]['selected_epoch'],color='.7',linestyle=':');ax.set_title(v);ax.grid(alpha=.2);ax.set_xlabel('Continuation epoch');ax.set_ylabel('Next-token NLL')
    axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(art/'generalization_curves.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));names=['native','full','band2']+cfg['variants']
    for split,ax in zip(cfg['evaluation_splits'],axes):
        values=[metrics['results'][f'{split}/{v}/gate']['stable_mean_gain_ms'] for v in names]
        ax.barh(names,values,color=['#999999']*3+['#245b93','#559176','#d49b32']*2);ax.invert_yaxis();ax.set_title(split);ax.set_xlabel('Mean stable-display gain (ms)');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(art/'display_gain.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for row,size in enumerate(['small','large']):
        for col,split in enumerate(cfg['evaluation_splits']):
            ax=axes[row,col]
            for suffix,color,label in [('ar','#245b93','AR'),('det','#559176','AR + deterministic state'),('fgn','#d49b32','AR + noisy state')]:
                points=metrics['risk_gain_operating_points'][f'{split}/{size}_{suffix}']
                x=[p['wrong_forecasts_per_100_native_tokens'] for p in points];y=[p['stable_mean_gain_ms'] for p in points]
                ax.plot(x,y,marker='.',color=color,label=label)
                chosen=next(p for p in points if p['dev_budget']==.5)
                ax.scatter([chosen['wrong_forecasts_per_100_native_tokens']],[chosen['stable_mean_gain_ms']],color=color,s=45)
            ax.set_title(f'{size}: {split}');ax.set_xlabel('Actual test false forecasts / 100 native tokens');ax.set_ylabel('Mean stable-display gain (ms)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Dev-frozen operating points; larger dots = dev error budget 0.5')
    fig.tight_layout();fig.savefig(art/'risk_gain_curves.png');fig.savefig(art/'risk_gain_curves.svg');plt.close(fig)
    lines=['# Layer-limited AR + FGN combination experiment — 2026-11-12','',
      '**Judgment: Although the actual combined and joint learning was completed, it did not surpass the existing layer-band2 AR under this single token display condition. The basic configuration maintains the existing band2+gate. This is not an experiment that proves or disproves the general overfitting prevention effect of noise.**','',
      'The existing layer-band2 AR was retained and combined with continuous FGN acoustic-state prediction. AR token loss backpropagates into FGN and both modules are fine-tuned jointly. This differs from the preceding independent predictor.','',
      'The public 960h Zipformer is frozen. Six warm-start variants use all 28,539 train-clean-100 utterances for 8 epochs. Small AR/hybrid sizes are 4.60M/10.93M; large sizes 13.03M/19.35M. The large AR has 12 layers: four trained layers plus eight initially identity blocks.','',
      'Small stochastic hybrid display gain is 88.46/70.18ms versus same-run AR 89.63/70.30ms. Clean difference −1.17ms, CI −2.02~−0.31; other −0.12ms, CI −0.97~+0.72. Large stochastic hybrid gives 81.91/65.08ms and adds 11.87/12.23ms per chunk; small AR adds about 1.3ms. Actual test error rates differ; latency alone does not establish matched-error superiority.','',
      'The effects of Noise itself were also inconsistent. In the small combination, it was +1.93/+1.68ms ahead of the deterministic hybrid, but false forecasts increased, and in the large combination, it was −1.31/−1.21ms. Therefore, we did not observe a pattern that the display gain of noise increases when the model is expanded.','',
      '## display performance','',
      'Mean stable-display gain for **final words matching the reference**, relative to native eager display; higher is better. Final transcripts are fixed, so unchanged WER is by construction. False forecasts are counted per 100 native BPE tokens, not words.','',
      '| Configuration | clean gain ms | other gain ms | clean/other False forecast | clean/other Additional CPU ms/chunk |','|---|---:|---:|---:|---:|']
    for v in names:
        a=metrics['results'][f'test-clean/{v}/gate'];b=metrics['results'][f'test-other/{v}/gate']
        lines.append(f"| {v} | {a['stable_mean_gain_ms']:.2f} | {b['stable_mean_gain_ms']:.2f} | {a['wrong_forecasts_per_100_native_tokens']:.3f} / {b['wrong_forecasts_per_100_native_tokens']:.3f} | {a['extra_ms_per_chunk']:.2f} / {b['extra_ms_per_chunk']:.2f} |")
    lines+=['','## Same size internal comparison','',
      'Gates were selected under the same development error budget, but actual test errors differ. Gain differences alone do not establish matched-error Pareto superiority. Intervals use paired utterance bootstrap; speaker intervals are stored separately in the original JSON.','',
      '| split | comparison | average gain difference ms [95% CI] | false forecast difference/100 token |','|---|---|---:|---:|']
    for split in cfg['evaluation_splits']:
        for size in ['small','large']:
            for a,b in [(size+'_det',size+'_ar'),(size+'_fgn',size+'_ar'),(size+'_fgn',size+'_det')]:
                r=metrics['paired_comparisons'][f'{split}/{a}-minus-{b}'];ci=r['ci95_ms']
                lines.append(f"| {split} | {a} − {b} | {r['delta_gain_ms']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | {r['delta_false_forecasts_per_100_native_tokens']:+.3f} |")
    lines+=['','## Model augmentation and overfitting','',
      'Noise is a function that learns the probability distribution of future acoustic features and is not a device that automatically guarantees prevention of overfitting. [FGN original text2.2.3](https://arxiv.org/html/2506.10772v1#S2.SS2.SSS3) states that some loss weights were lowered to reduce overfitting when expanding the model.','',
      'Each epoch evaluates fixed 256-example train-clean and separate 256-example model-selection probes using the same sampling/evaluation mode. Define overfitting gap as dev-clean minus train-clean NLL; dev-other confounds domain difficulty. Select checkpoints on mixed-dev NLL, allowing epoch 0.','',
      '| configure | select epoch | select dev NLL | last train NLL | last dev-clean NLL | last gap |','|---|---:|---:|---:|---:|---:|']
    for v,t in training.items():
        tr=t['last_train_probe']['nll'];dv=t['last_dev']['by_split']['dev-clean']['nll']
        lines.append(f"| {v} | {t['selected_epoch']} | {t['selected_dev']['nll']:.4f} | {tr:.4f} | {dv:.4f} | {dv-tr:+.4f} |")
    lines+=['',
      'These curves describe one-seed continuation. A small gap alone is not evidence of less overfitting. Inspect whether training improves while validation worsens, and whether noise improves matched-size dev NLL and actual risk–display gain. Depth and effective text context changed together.','',
      'Noise lowered the mixed dev NLL of small/large combination to 1.0791→1.0687,1.0589→1.0450, respectively. dev-clean − The train gap also became slightly smaller from 0.0957 → 0.0851,0.1205 → 0.1074. However, both learning and validation losses were improved, and there was no clear overfitting worsening pattern in the control group. It was not possible to separate the effect of probability prediction/representation learning from overfitting mitigation, and the improvement of NLL did not lead to improvement in display performance.','',
      'Generated-state standard deviation of about 0.52~0.53 confirms active noise. However, branch top tokens agree with the mixture choice on average 99.4~99.7%, with predictive mutual information about 0.0003~0.0004. Feature variation does not produce large token differences. This diagnostic alone does not establish a cause.','',
      'This NLL evaluates one fixed sampled position per utterance; do not compare directly with prior all-token/multi-horizon NLL. All four hybrids selected the last epoch, leaving longer-run convergence unresolved.','',
      '## Methods and Limitations','',
      '- The AR weight started from the existing band2, and the state weights of both hybrids started from the same state_s37. It is not the result of learning a large model from scratch.',
      '- Shared noise, continuous state feedback and multi-level backpropagation are in the future speech features. In AR, predictive features are provided as cross-attention. This is not an experiment in which all normalization layers of AR itself are modulated with noise.',
      '- When enlarging, only the AR was changed from 4 to 12 layer, and the FGN state generator was maintained as 6.324M. This is not an experiment that directly verifies overfitting when the FGN generator itself is expanded.',
      '- Actual future acoustic states were used only as CRPS targets. The token prediction input contains only the arriving voice and the generated future features. We examined the FGN gradient and future input invariance of token loss.',
      '- For actual inference, native recognition prefix is \u200b\u200bused. Confirmed caches contain only observed audio; predicted features enter temporary branches only. Text KV retains 3 positions; audio KV grows with utterance length.',
      '- The teacher-forced token NLL in the train/development table is an indicator that is different from the false-forecast rate on the actual screen.',
      '- The preceding text state for learning is calculated by looking at the actual speech up to the selection cutoff. The actual inference stores the state at the time of past native release. In both cases, we do not add predictive speech to the pre-confirmation state, but this does not claim that the teacher forcing and streaming states are exactly the same across the varying cutoffs.',
      '- Gate weights were fit on 1297 dev utterances and thresholds on 1414 separate utterances before evaluating 5559 test utterances. The gate, timing, and standards of the existing 3 control group were reused as is, and 12 points were accurately reproduced.',
      '- We did not measure resource contention with the actual microphone/UI and backbone. The median display gain for each word and the actual test error rate were saved in results.json.',
      '- It is a warm-start pilot of one seed, two depths, 8epoch, and does not prove the general scaling law or the overfitting prevention effect of FGN.',
      '- Some independent runs were executed concurrently after checking available memory. Individual epoch times include contention, so this is not a standalone GPU speed comparison.',
      '- The initial weights of the state were taken from a model already trained with noise. Even if the parameters are the same, the starting function and score of noise0 and probability sampling are different, and the difference in final score is not directly interpreted as an overfitting prevention effect.',
      "- Original results were preserved; no commit/push was performed at that experiment's completion.",'',
      '## file','',
      '- `experiments/band_fgn_v1/README.md`: Actual hybrid architecture, controls and reproduction sequence.',
      '- `runs/band_fgn_v1/{variant}/{best,last}.pt`: Additional learning weights.',
      '- `runs/band_fgn_v1/checks.json`: Causality·gradient·cache inspection.',
      '- `experiments/band_fgn_v1/artifacts/`: Results JSON/CSV·Learning curve·Display comparison·hash.','',
      '![Generalization curves](../experiments/band_fgn_v1/artifacts/generalization_curves.png)','',
      '![Display gain](../experiments/band_fgn_v1/artifacts/display_gain.png)','',
      '![Dev-frozen risk and gain](../experiments/band_fgn_v1/artifacts/risk_gain_curves.png)']
    text='\n'.join(lines)+'\n';(ROOT/'docs/band_fgn_hybrid_2026-11-12.md').write_text(text)
    (art/'summary.md').write_text(text.replace('../experiments/band_fgn_v1/artifacts/',''))
    for f in ['results.json','gates.json','checks.json','initial_plan.json']:
        shutil.copy2(OUT/f,art/f)
    write(art/'provenance.json',dict(training=training,source_sha256={p.name:sha(p) for p in HERE.glob('*.py')},config_sha256=sha(HERE/'config.json'),
        dev_provenance=json.loads((OUT/'events_dev_provenance.json').read_text()),test_provenance=json.loads((OUT/'events_test_provenance.json').read_text()),
        completed_unix=time.time(),old_controls=metrics['old_control_reproduction']))
    print(json.dumps(dict(report='docs/band_fgn_hybrid_2026-11-12.md',variants=len(training),test_utterances=5559)),flush=True)
if __name__=='__main__':main()
