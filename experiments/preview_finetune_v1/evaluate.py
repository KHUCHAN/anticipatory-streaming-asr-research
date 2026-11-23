"""Display delay, rewrites and gates for original-encoder previews and fine-tuned previews on the 20M model (adapted from preview_filler_v1/stage2.py)."""
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from shared import HERE,OUT,ROOT,config,edit_pairs,write

cfg=config();S=cfg['stage2'];H=cfg['feature_hop'];SHIFT=cfg['frame_shift_ms'];OFFSET=cfg['frame_audio_end_offset_ms'];W=cfg['feature_window']
ALIGN=json.loads((ROOT/cfg['alignments']).read_text())


def load(role,tags):
    base=json.loads((OUT/f'collect_{role}_{tags[0]}.json').read_text())
    for tag in tags[1:]:
        other=json.loads((OUT/f'collect_{role}_{tag}.json').read_text())
        for a,b in zip(base['records'],other['records']):
            assert a['utterance_id']==b['utterance_id'] and len(a['chunks'])==len(b['chunks'])
            for x,y in zip(a['chunks'],b['chunks']):
                assert x['ids']==y['ids'];x['previews'].update(y['previews'])
    return base


def token_rows(records,key,start):
    X=[];y=[]
    for r in records:
        for ch in r['chunks']:
            p=ch['previews'][key];low=1.0
            for j,t in enumerate(p['ids']):
                low=min(low,p['p'][j]);X.append([p['p'][j],p['pb'][j],p['mg'][j],p['fr'][j],j,len(p['ids']),low,float(start[t])])
                y.append(j<len(ch['ids']) and ch['ids'][j]==t)
    return np.array(X,np.float32).reshape(-1,8),np.array(y,bool)


def attach(records,key,model,start):
    for r in records:
        for ch in r['chunks']:
            p=ch['previews'][key];low=1.0;X=[]
            for j,t in enumerate(p['ids']):
                low=min(low,p['p'][j]);X.append([p['p'][j],p['pb'][j],p['mg'][j],p['fr'][j],j,len(p['ids']),low,float(start[t])])
            p['score']=model.predict_proba(np.array(X,np.float32))[:,1].tolist() if X else []


def simulate(record,words,name,ks,tau):
    audio=1000*record['audio_seconds'];free=0.0;committed=[];events=[];compute=0.0
    for c,chunk in enumerate(record['chunks']):
        for k in ks:
            if name=='zero&repeat':
                a=chunk['previews'][f'zero|{k}'];b=chunk['previews'][f'repeat|{k}'];ids=[]
                for x,y in zip(a['ids'],b['ids']):
                    if x!=y:break
                    ids.append(x)
                seconds=a['seconds']+b['seconds']
            else:
                p=chunk['previews'][f'{name}|{k}'];ids=p['ids'];seconds=p['seconds']
                if tau is not None:
                    cut=next((j for j,v in enumerate(p['score']) if v<tau),len(ids));ids=ids[:cut]
            arrival=min((c*H+k-1)*SHIFT+OFFSET,audio);free=max(arrival,free)+1000*seconds;compute+=seconds;events.append((free,words(committed+ids)))
        arrival=min((c*H+W-1)*SHIFT+OFFSET,audio);free=max(arrival,free)+1000*chunk['seconds'];compute+=chunk['seconds']
        committed=committed+chunk['ids'];events.append((free,words(committed)))
    final=words(committed);ref=ALIGN[record['utterance_id']];errors,pairs=edit_pairs(ref['words'],final);stable=[None]*len(final)
    for i in range(len(final)):
        for t,shown in reversed(events):
            if i<len(shown) and shown[i]==final[i]:stable[i]=t
            else:break
    rewrites=0
    for (_,a),(_,b) in zip(events[:-1],events[1:]):rewrites+=sum(1 for i,w in enumerate(a) if i>=len(b) or not b[i].startswith(w))
    return dict(delay=np.array([stable[i]-ref['end_ms'][j] for i,j in pairs]),errors=errors,reference_words=len(ref['words']),final_words=len(final),
        rewrites=rewrites,compute=compute+record['feature_seconds'],audio=record['audio_seconds'],chunks=len(record['chunks']))


def summarize(stats):
    d=np.concatenate([s['delay'] for s in stats])
    return dict(delay_mean=float(d.mean()),delay_p50=float(np.percentile(d,50)),delay_p90=float(np.percentile(d,90)),
        rewrites_per_100=100*sum(s['rewrites'] for s in stats)/sum(s['final_words'] for s in stats),
        wer_percent=100*sum(s['errors'] for s in stats)/sum(s['reference_words'] for s in stats),rtf=sum(s['compute'] for s in stats)/sum(s['audio'] for s in stats))


def main():
    names=S['free_fillers']+S['learned_fillers'];train=load('train',['all']);sel=load('selection',['all']);ev=json.loads((OUT/'collect_eval_all.json').read_text())
    tokens={int(k):v for k,v in ev['tokens'].items()};start={k:v.startswith('▁') for k,v in tokens.items()}

    def words(ids):return ''.join(tokens[i] for i in ids).replace('▁',' ').upper().split()

    gates={}
    for name in names:
        for k in S['ks']:
            key=f'{name}|{k}';X,y=token_rows(train['records'],key,start);model=HistGradientBoostingClassifier(max_iter=150,learning_rate=0.1,max_leaf_nodes=15,random_state=cfg['seed']).fit(X,y)
            Xs,ys=token_rows(sel['records'],key,start);gates[key]=dict(train_tokens=int(len(y)),train_correct=float(y.mean()),selection_correct=float(ys.mean()),
                selection_auc=float(roc_auc_score(ys,model.predict_proba(Xs)[:,1])) if 0<ys.mean()<1 else None)
            attach(sel['records'],key,model,start);attach(ev['records'],key,model,start)
    grid=[.3,.5,.6,.7,.8,.85,.9,.93,.95,.97,.98,.99];chosen={}
    for name in names:
        for label,ks in S['schedules'].items():
            sweep=[(tau,summarize([simulate(r,words,name,ks,tau) for r in sel['records']])) for tau in grid]
            for target in S['rewrite_targets_per_100']:
                ok=[(v['delay_mean'],tau) for tau,v in sweep if v['rewrites_per_100']<=target]
                chosen[f'{label}|{name}|R{target:g}']=min(ok)[1] if ok else grid[-1]
    rng=np.random.default_rng(cfg['seed']);results={'gates':gates,'chosen_thresholds':chosen,'finetune':json.loads((OUT/'student.json').read_text())}
    lines=[]
    for split in cfg['splits']:
        records=[r for r in ev['records'] if r['split']==split];draws=rng.integers(0,len(records),(cfg['bootstrap_replicates']//2,len(records)));out={}
        base=[simulate(r,words,'repeat',[],None) for r in records];out['none']=summarize(base)
        for label,ks in S['schedules'].items():
            ref=[simulate(r,words,'repeat',ks,None) for r in records]
            policies=[(name,None,'ungated') for name in names]+[(name,chosen[f'{label}|{name}|R{t:g}'],f'R{t:g}') for name in names for t in S['rewrite_targets_per_100']]+[('zero&repeat',None,'agree')]
            for name,tau,mode in policies:
                stats=ref if (name,mode)==('repeat','ungated') else [simulate(r,words,name,ks,tau) for r in records];row=summarize(stats);row['threshold']=tau
                dd=[];dr=[]
                for index in draws:
                    a=[stats[i] for i in index];b=[ref[i] for i in index]
                    dd.append(np.concatenate([s['delay'] for s in a]).mean()-np.concatenate([s['delay'] for s in b]).mean())
                    dr.append(100*sum(s['rewrites'] for s in a)/sum(s['final_words'] for s in a)-100*sum(s['rewrites'] for s in b)/sum(s['final_words'] for s in b))
                row['vs_repeat_ungated']=dict(delay_ms_ci95=[float(v) for v in np.percentile(dd,[2.5,97.5])],rewrites_ci95=[float(v) for v in np.percentile(dr,[2.5,97.5])])
                out[f'{label}|{name}|{mode}']=row
        results[split]=out
    write(HERE/'artifacts/results.json',results)
    a,b=[results[s] for s in cfg['splits']];fmt=lambda key:'%3.0f/%4.1f %3.0f/%4.1f'%(a[key]['delay_mean'],a[key]['rewrites_per_100'],b[key]['delay_mean'],b[key]['rewrites_per_100'])
    lines.append('columns: mean stable delay ms / rewrites per 100 words, dev-clean then dev-other')
    lines.append('none  '+fmt('none')+'  WER %.2f/%.2f RTF %.3f'%(a['none']['wer_percent'],b['none']['wer_percent'],a['none']['rtf']))
    for label in S['schedules']:
        for name in names:
            lines.append('%-7s %-6s ung %s | R5 t%.2f %s | R2 t%.2f %s | rtf %.3f'%(label,name,fmt(f'{label}|{name}|ungated'),chosen[f'{label}|{name}|R5'],fmt(f'{label}|{name}|R5'),
                chosen[f'{label}|{name}|R2'],fmt(f'{label}|{name}|R2'),a[f'{label}|{name}|ungated']['rtf']))
        lines.append('%-7s zero&repeat agree %s | rtf %.3f'%(label,fmt(f'{label}|zero&repeat|agree'),a[f'{label}|zero&repeat|agree']['rtf']))
    lines.append('WER identical across policies: '+str(len({round(v['wer_percent'],6) for v in a.values()})==1 and len({round(v['wer_percent'],6) for v in b.values()})==1))
    lines.append('finetune validation initial -> final: '+' '.join('%s kl %.3f->%.3f recall %.2f->%.2f precision %.2f->%.2f'%(k,a0['kl'],a1['kl'],a0['recall'],a1['recall'],a0['precision'],a1['precision']) for (k,a0),a1 in zip(results['finetune']['initial'].items(),results['finetune']['final'].values())))
    for s,r in zip(cfg['splits'],[a,b]):
        for key in ['k7|zero|R2','k7|ft|ungated','k7|ft|R5','k7|ft|R2','k7_23|ft|ungated','k7_23|ft|R5']:lines.append('%s %s p90 %.0f vs repeat ungated delay CI %s rewrites CI %s'%(s,key,r[key]['delay_p90'],[round(x) for x in r[key]['vs_repeat_ungated']['delay_ms_ci95']],[round(x,1) for x in r[key]['vs_repeat_ungated']['rewrites_ci95']]))
    lines.append('gate selection AUC: '+' '.join('%s %.2f'%(k,v['selection_auc']) for k,v in gates.items() if v['selection_auc']))
    lines.append('preview token correct rate (selection): '+' '.join('%s %.2f'%(k,v['selection_correct']) for k,v in gates.items()))
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
