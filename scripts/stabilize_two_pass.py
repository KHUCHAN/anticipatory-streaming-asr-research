#!/usr/bin/env python3
"""Causal two-pass display policies selected on dev, then frozen on test.

Count repeated agreement only at increasing acoustic coverage, not at forecast
events or a second pass rereading the same audio. The slow pass owns confirmed
prefixes; its completed final transcript is flushed when that pass finishes.
"""
from collections import deque
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_final2 as analysis
from analyze_final import Gate,SETS,labelled,tok

BASE_REPLAY=analysis.replay
ALIGNMENT_SECONDS=[]


def lcp(a,b):
    n=0
    while n<min(len(a),len(b)) and a[n]==b[n]: n+=1
    return n


def overlap_end(slow,fast,expected):
    """Align an observed slow suffix near the fast audio-coverage boundary."""
    if not slow or not expected: return expected
    begin=max(0,expected-24); segment=fast[begin:min(len(fast),expected+8)]; tail=slow[-16:]
    previous=[0]*(len(segment)+1)
    for i,token in enumerate(tail,1):
        current=[i]
        for j,other in enumerate(segment,1):
            current.append(min(previous[j]+1,current[j-1]+1,previous[j-1]+(token!=other)))
        previous=current
    j=min(range(len(previous)),key=lambda j:(previous[j],abs(begin+j-expected)))
    return begin+j if previous[j]<=max(3,.35*len(tail)) else expected


def overlap_replay(uf,us,cf,cs,score,th,extra):
    """Same causal clocks as the old merge; replace only its splice boundary."""
    ev=[]; fb=[]; sb=[]; n=0
    for k,b in enumerate(uf['bounds']):
        n+=len(b['new']); fb.append(((k+1)*cf if k<len(uf['bounds'])-1 else 10**9,n))
        ev.append((b['t'],0,'f',k,None))
        if b['f'] is not None and score is not None and score(b['f'])>=th:
            ev.append((b['t']+b['tg']+(b['tn'] if extra else 0),1,'c',k,[b['g'][0][0]]))
    n=0
    for k,b in enumerate(us['bounds']):
        n+=len(b['new']); sb.append(((k+1)*cs if k<len(us['bounds'])-1 else 10**9,n))
        ev.append((b['t'],2,'s',k,None))
    fk=sk=-1; fc=[]; out=[]; previous=None
    for t,_,kind,k,payload in sorted(ev,key=lambda row:(row[0],row[1])):
        if kind=='f': fk,fc=k,[]
        elif kind=='c': fc=payload if k==fk else fc
        else: sk=k
        cover,ns=sb[sk] if sk>=0 else (0,0)
        ids=us['ids'][:ns]
        if fk>=0 and fb[fk][0]>cover:
            j=next((i for i,(end,_) in enumerate(fb) if end>cover),None)
            expected=fb[j-1][1] if j else 0
            fast=uf['ids'][:fb[fk][1]]
            start=time.perf_counter(); splice=overlap_end(ids,fast,expected)
            ALIGNMENT_SECONDS.append(time.perf_counter()-start)
            ids=ids+fast[splice:]+fc
        if ids!=previous:
            out.append((t,analysis.to_words(ids))); previous=ids
    return out,analysis.to_words(us['ids'])


def stabilize(events,final,uf,us,cf,cs,confirmations,drop_last):
    by_time={t:words for t,words in events}
    acoustic={}
    for u,c in ((uf,cf),(us,cs)):
        for k,b in enumerate(u['bounds']):
            acoustic[b['t']]=max(acoustic.get(b['t'],0),min((k+1)*c*.04,u['audio_s']))
    slow=[]; n=0
    for b in us['bounds']:
        n+=len(b['new']); slow.append((b['t'],n))
    timeline=sorted(set(by_time)|set(acoustic))
    hist=deque(maxlen=confirmations); coverage=-1.; words=[]; previous=None; out=[]; slow_k=-1
    final_ready=us['bounds'][-1]['t']
    for t in timeline:
        if t in by_time: words=by_time[t]
        while slow_k+1<len(slow) and slow[slow_k+1][0]<=t: slow_k+=1
        if t>=final_ready:
            shown=final
        else:
            candidate=words[:-1] if drop_last and words else words
            if t in acoustic and acoustic[t]>coverage+1e-9:
                coverage=acoustic[t]; hist.append(list(candidate))
            count=0
            if len(hist)==confirmations:
                count=len(candidate)
                for old in hist: count=min(count,lcp(candidate,old))
            confirmed=analysis.to_words(us['ids'][:slow[slow_k][1]])[:-1] if slow_k>=0 else []
            if confirmed and words[:len(confirmed)]!=confirmed:
                raise AssertionError('Raw merge disagrees with its completed slow prefix')
            shown=words[:max(count,len(confirmed))]
        if shown!=previous:
            out.append((t,list(shown))); previous=list(shown)
    if not out or out[-1][1]!=final:
        raise AssertionError('Stabilization changed the final transcript')
    return out,final


def causal_smoke():
    ids=tok.encode('A B')
    uf={'audio_s':.64,'bounds':[{'t':t,'new':[]} for t in (.16,.32,.48,.64)]}
    us={'audio_s':.64,'ids':ids,'bounds':[{'t':.64,'new':ids}]}
    events=[(.16,['A']),(.32,['A','X']),(.48,['A','B']),(.64,['A','B'])]
    out,_=stabilize(events,['A','B'],uf,us,4,16,2,False)
    if any('X' in words for _,words in out): raise AssertionError('Unstable word passed confirmation')
    if next(t for t,words in out if words==['A','B'])!=.64: raise AssertionError('Used future final transcript')


def measure(split,data,gate,policy,forecast):
    confirmations,drop_last,overlap=policy
    def replay(uf,us,cf,cs,score,th,extra):
        events,final=(overlap_replay if overlap else BASE_REPLAY)(uf,us,cf,cs,score,th,extra)
        return (events,final) if not confirmations else stabilize(events,final,uf,us,cf,cs,confirmations,drop_last)
    analysis.replay=replay
    fast=data[4]['utts'][:300]; slow=data[16]['utts'][:300]
    if [u['utt'] for u in fast]!=[u['utt'] for u in slow]: raise AssertionError('Pass utterance orders differ')
    _,metrics=analysis.screen_metrics((split,fast,slow,4,16,gate if forecast else None,.87,forecast))
    return metrics


def main():
    torch.set_num_threads(1); causal_smoke()
    gate_sets=[analysis.load('gate_dev-clean'),analysis.load('gate_dev-other')]
    X=[]; y=[]
    for c in sorted(gate_sets[0]):
        a,b=labelled([u for g in gate_sets for u in g[c]['utts']]); X+=a; y+=b
    gate=Gate(SETS['lr_all']).fit(X,y)
    policies={'raw':(0,False,False),'confirm2':(2,False,False),'confirm3':(3,False,False),
              'confirm2_complete_words':(2,True,False),'confirm3_complete_words':(3,True,False),
              'overlap':(0,False,True),'overlap_confirm2':(2,False,True)}
    out={'protocol':{'dev':'dev-clean, first 300 saved utterances','test':['test-clean','test-other'],
                     'n':300,'gate':'same lr_all reconstructed from original gate data','threshold':.87,
                     'selection':'minimize mean rewrites over forecast/no-forecast, subject to mean latency increase <=80ms and P95 increase <=160ms on dev; baseline eligible',
                     'scope':'offline screen-event replay; not a physical UI or parallel-pass contention benchmark'},
         'dev':{},'test':{}}
    dev=analysis.load('eval_dev-clean')
    for name,policy in policies.items():
        out['dev'][name]={str(fc):measure('dev-clean',dev,gate,policy,fc) for fc in (False,True)}
        print(json.dumps({'split':'dev-clean','policy':name,'results':out['dev'][name]}),flush=True)
    baseline=out['dev']['raw']
    eligible=[]
    for name,metrics in out['dev'].items():
        if all(metrics[str(fc)]['stable_mean']-baseline[str(fc)]['stable_mean']<=80 and
               metrics[str(fc)]['stable_p95']-baseline[str(fc)]['stable_p95']<=160 for fc in (False,True)):
            eligible.append(name)
    chosen=min(eligible,key=lambda name:sum(out['dev'][name][str(fc)]['rewritten_words_per_100'] for fc in (False,True)))
    out['chosen_policy']=chosen
    for split in ('test-clean','test-other'):
        data=analysis.load('eval_'+split)
        out['test'][split]={}
        for name in dict.fromkeys(('raw',chosen,'confirm2','confirm3','overlap')):
            out['test'][split][name]={str(fc):measure(split,data,gate,policies[name],fc) for fc in (False,True)}
            print(json.dumps({'split':split,'policy':name,'results':out['test'][split][name]}),flush=True)
    analysis.replay=BASE_REPLAY
    if ALIGNMENT_SECONDS:
        ordered=sorted(ALIGNMENT_SECONDS)
        out['alignment_compute_ms']={'p50':1000*ordered[len(ordered)//2],
                                      'p95':1000*ordered[int(.95*len(ordered))],
                                      'max':1000*ordered[-1],
                                      'note':'Measured separately; not added to saved replay clocks; concurrent seed training'}
    target=ROOT/'runs/followups/two_pass_stability.json'; target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,indent=2)+'\n')
    print('TWO PASS STABILITY COMPLETE; chosen='+chosen,flush=True)


if __name__=='__main__': main()
