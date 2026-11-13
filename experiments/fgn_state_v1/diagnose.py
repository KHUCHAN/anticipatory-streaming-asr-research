"""Held-out continuous and discrete trajectory diagnostics, not gate tuning."""
import argparse,hashlib,json,random,time
import numpy as np
import torch
import collect
from data import ROOT,OUT,selections
from runtime import decode_states


def energy(samples,target):
    # Scale by sqrt(dimension), and exclude diagonal pairs (fair ensemble score).
    x=samples.flatten(1);y=target.flatten();n=len(x);d=x.shape[-1]**.5
    accuracy=(x-y).norm(dim=-1).mean()/d
    spread=torch.cdist(x,x).sum()/(n*(n-1)*d)
    return float(accuracy-.5*spread)


def crps(samples,target):
    n=len(samples)
    accuracy=(samples-target).abs().mean()
    # Fair ensemble marginal CRPS; two independent member indices.
    diversity=torch.stack([(samples[i]-samples[j]).abs().mean() for i in range(n) for j in range(i)]).mean()
    return float(accuracy-.5*diversity)


def shuffle_members_by_frame(x,generator):
    # Every frame's full 512-dimensional sample marginal is identical; only
    # cross-frame and cross-step member correspondence is destroyed.
    n,h,t,f=x.shape;out=x.clone()
    for step in range(h):
        for frame in range(t):out[:,step,frame]=x[torch.randperm(n,generator=generator),step,frame]
    return out


def add_discrete(record,prefix,native_future):
    first=[p[0] if p else 0 for p in prefix];counts={v:first.count(v) for v in set(first)}
    mode=sorted(counts,key=lambda v:(-counts[v],v))[0]
    for k in [1,2,3]:
        valid=len(native_future)>=k
        record[f'path{k}_valid']=int(valid)
        record[f'path{k}_hit_any']=int(valid and any(p[:k]==native_future[:k] for p in prefix))
    record['next_mode_hit']=int(bool(native_future) and mode==native_future[0])
    record['next_valid']=int(bool(native_future))
    record['empty_paths_fraction']=sum(not p for p in prefix)/len(prefix)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=256);a=p.parse_args()
    collect.initialize();cfg=collect.CFG;_,rows=selections(cfg);rows=rows[:a.limit]
    h=cfg['trajectory_diagnostic_steps'];n=cfg['trajectory_diagnostic_members'];records=[];started=time.time()
    for ri,row in enumerate(rows):
        uid=row['utterance_id']
        with np.load(ROOT/f"data/processed/zipformer_forecasts/{row['split']}/{uid}.npz") as v:
            states=torch.from_numpy(v['enc'].astype(np.float32)).view(-1,8,512)
            cuts=v['cuts'].tolist();native=v['ids'].tolist();duration=float(v['duration'])
        last=min(len(states)-h-1,int((duration-.48)/.32)-h)
        if last<1:continue
        rng=random.Random(int(hashlib.sha256(uid.encode()).hexdigest()[:8],16));cut=rng.randint(1,last)
        count=cuts[cut];context=native[count-2:count] if count>=2 else ([0,native[0]] if count else [-1,0])
        target=states[cut+1:cut+h+1];native_future=native[count:cuts[cut+h]]
        for kind in collect.KINDS:
            runtime=collect.make_runtime(kind)
            for x in states[:cut+1]:runtime.observe(x)
            generator=torch.Generator().manual_seed(int(hashlib.sha256((uid+'diagnostics').encode()).hexdigest()[:12],16))
            prediction=runtime.sample_states(generator,steps=h,members=n)
            shuffled=shuffle_members_by_frame(prediction,generator)
            normalized=(prediction-collect.MEAN)/collect.STD;truth=(target-collect.MEAN)/collect.STD
            for condition,x in [('coherent',prediction),('shuffled',shuffled)]:
                standardized=(x-collect.MEAN)/collect.STD
                r=dict(id=uid,split=row['split'],kind=kind,condition=condition,cut_chunk=cut,
                       energy_score=energy(standardized,truth),marginal_crps=crps(standardized,truth),
                       sample_std=float(standardized.std(0).mean()),mean_rmse=float((standardized.mean(0)-truth).square().mean().sqrt()))
                for step in [1,2,4]:r[f'energy_h{step}']=energy(standardized[:,:step],truth[:step])
                paths=decode_states(x,collect.BRIDGE,context,first_only=False);add_discrete(r,paths,native_future)
                records.append(r)
        if (ri+1)%25==0:print(json.dumps(dict(utterances=ri+1,seconds=time.time()-started)),flush=True)
    summary={}
    for kind in collect.KINDS:
        for condition in ['coherent','shuffled']:
            chosen=[r for r in records if r['kind']==kind and r['condition']==condition]
            if not chosen:continue
            keys=['energy_score','marginal_crps','sample_std','mean_rmse','energy_h1','energy_h2','energy_h4','empty_paths_fraction']
            s={k:float(np.mean([r[k] for r in chosen])) for k in keys};s['utterances']=len(chosen)
            for k in [1,2,3]:
                denom=sum(r[f'path{k}_valid'] for r in chosen)
                s[f'path{k}_eligible']=denom;s[f'path{k}_hit_any_percent']=100*sum(r[f'path{k}_hit_any'] for r in chosen)/max(1,denom)
            s['next_mode_hit_percent']=100*sum(r['next_mode_hit'] for r in chosen)/max(1,sum(r['next_valid'] for r in chosen))
            summary[kind+'/'+condition]=s
    # Shuffle construction must preserve CRPS, but need not preserve joint scores.
    for kind in collect.KINDS:
        if kind+'/coherent' in summary:
            assert abs(summary[kind+'/coherent']['marginal_crps']-summary[kind+'/shuffled']['marginal_crps'])<1e-5
            coherent={r['id']:r for r in records if r['kind']==kind and r['condition']=='coherent'}
            shuffled={r['id']:r for r in records if r['kind']==kind and r['condition']=='shuffled'}
            delta=np.array([coherent[uid]['energy_score']-shuffled[uid]['energy_score'] for uid in coherent])
            rng=np.random.default_rng(416);draw=rng.integers(0,len(delta),(1000,len(delta)))
            summary[kind+'/coherent']['coherent_minus_shuffled_energy_mean']=float(delta.mean())
            summary[kind+'/coherent']['coherent_minus_shuffled_energy_ci95']=np.percentile(delta[draw].mean(1),[2.5,97.5]).tolist()
    result=dict(summary=summary,records=records,seconds=time.time()-started,
                scope='Fixed held-out model_selection utterances; four free-running 320ms states, 16 members. Discrete path hit is native-decoder agreement within that future window, NOT reference-word accuracy.',
                shuffle='Independent member permutations per predicted frame preserve every frame-vector ensemble marginal exactly and destroy cross-frame/step correspondence. Fair multivariate energy score probes joint forecast structure; marginal CRPS alone cannot certify it.',
                caveats='Selection/checkpoint data also used for training validation; diagnostic rather than an independent test. One cutoff per utterance; no claim of calibrated path probabilities from only 16 samples.')
    (OUT/'trajectory_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
