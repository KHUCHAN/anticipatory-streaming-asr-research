"""Reference-anchored display delay, rewrite events, final WER and complete compute for every filler and schedule."""
import json
import sys

import numpy as np

from shared import HERE,OUT,config,edit_pairs,write


def main():
    cfg=config();name=sys.argv[1] if len(sys.argv)>1 else 'replay.json'
    data=json.loads((OUT/name).read_text());align=json.loads((OUT/'alignments.json').read_text());tokens={int(k):v for k,v in data['tokens'].items()}
    H=cfg['feature_hop'];shift=cfg['frame_shift_ms'];offset=cfg['frame_audio_end_offset_ms']

    def words(ids):return ''.join(tokens[i] for i in ids).replace('▁',' ').upper().split()

    def simulate(record,filler,precision,ks):
        """Returns per-utterance statistics for one display policy."""
        audio=1000*record['audio_seconds'];free=0.0;committed=[];events=[];compute=0.0
        for c,chunk in enumerate(record['chunks']):
            for k in list(ks)+[45]:
                source=chunk if k==45 else chunk['previews'][f'{filler}|{k}|{precision}']
                arrival=min((c*H+k-1)*shift+offset,audio);free=max(arrival,free)+1000*source['seconds'];compute+=source['seconds']
                events.append((free,words(committed+source['ids'])))
            committed=committed+chunk['ids']
        final=words(committed);ref=align[record['utterance_id']];errors,pairs=edit_pairs(ref['words'],final)
        stable=[None]*len(final);first=[None]*len(final)
        for i in range(len(final)):
            for t,shown in reversed(events):
                if i<len(shown) and shown[i]==final[i]:stable[i]=t
                else:break
            for t,shown in events:
                if i<len(shown) and shown[i]==final[i]:first[i]=t;break
        rewrites=0
        for (_,a),(_,b) in zip(events[:-1],events[1:]):
            rewrites+=sum(1 for i,w in enumerate(a) if i>=len(b) or not b[i].startswith(w))
        return dict(stable=[stable[i]-ref['end_ms'][j] for i,j in pairs],first=[first[i]-ref['end_ms'][j] for i,j in pairs],errors=errors,
            reference_words=len(ref['words']),final_words=len(final),rewrites=rewrites,compute=compute,feature=record['feature_seconds'],
            audio=record['audio_seconds'],chunks=len(record['chunks']))

    policies=[('none','-','fp32',[])]
    for label,ks in cfg['schedules'].items():
        if not ks:continue
        policies+=[(label,f,'fp32',ks) for f in cfg['fillers']]+[(label,cfg['int8_preview_filler'],'int8',ks)]
    rng=np.random.default_rng(cfg['seed']);results={};reps=cfg['bootstrap_replicates']
    for split in cfg['splits']:
        records=[r for r in data['records'] if r['split']==split];per={}
        for label,filler,precision,ks in policies:per[(label,filler,precision)]=[simulate(r,filler,precision,ks) for r in records]
        draws=rng.integers(0,len(records),(reps,len(records)))

        def pooled(stats,index,key,fn):return fn(np.concatenate([stats[i][key] for i in index]))

        base=per[('none','-','fp32')];out={}
        for (label,filler,precision),stats in per.items():
            delay=np.concatenate([s['stable'] for s in stats]);first=np.concatenate([s['first'] for s in stats])
            row=dict(utterances=len(stats),wer_percent=100*sum(s['errors'] for s in stats)/sum(s['reference_words'] for s in stats),
                word_coverage=len(delay)/sum(s['reference_words'] for s in stats),
                stable_delay_ms=dict(mean=float(delay.mean()),p50=float(np.percentile(delay,50)),p90=float(np.percentile(delay,90)),p95=float(np.percentile(delay,95))),
                first_correct_delay_ms=dict(mean=float(first.mean()),p50=float(np.percentile(first,50)),p90=float(np.percentile(first,90))),
                rewrites_per_100_final_words=100*sum(s['rewrites'] for s in stats)/sum(s['final_words'] for s in stats),
                rtf=sum(s['compute']+s['feature'] for s in stats)/sum(s['audio'] for s in stats),
                compute_ms_per_chunk=1000*sum(s['compute'] for s in stats)/sum(s['chunks'] for s in stats))
            if label!='none':
                for ref_name,ref in [('none',base)]+([('zero_same_schedule',per[(label,'zero','fp32')])] if filler!='zero' else []):
                    dm=[];dp=[];dq=[]
                    for index in draws:
                        a=np.concatenate([stats[i]['stable'] for i in index]);b=np.concatenate([ref[i]['stable'] for i in index])
                        dm.append(a.mean()-b.mean());dp.append(np.percentile(a,50)-np.percentile(b,50));dq.append(np.percentile(a,90)-np.percentile(b,90))
                    row[f'stable_delay_change_vs_{ref_name}_ms']={k:dict(estimate=float(e),ci95=[float(x) for x in np.percentile(v,[2.5,97.5])])
                        for k,v,e in [('mean',dm,delay.mean()-np.concatenate([s['stable'] for s in ref]).mean()),
                                      ('p50',dp,np.percentile(delay,50)-np.percentile(np.concatenate([s['stable'] for s in ref]),50)),
                                      ('p90',dq,np.percentile(delay,90)-np.percentile(np.concatenate([s['stable'] for s in ref]),90))]}
            out[f'{label}|{filler}|{precision}']=row
        results[split]=out
    results['native_equal']=dict(equal=sum(r['native_equal'] for r in data['records']),utterances=len(data['records']))
    if name=='replay.json':write(HERE/'artifacts/results.json',results)
    for split in cfg['splits']:
        print(split)
        for key,row in results[split].items():
            d=row['stable_delay_ms'];print('  %-28s WER %.2f cov %.3f | stable delay mean %4.0f p50 %4.0f p90 %4.0f p95 %4.0f | first %4.0f | rewrites/100 %5.2f | RTF %.4f | %.1f ms/chunk'%(
                key,row['wer_percent'],row['word_coverage'],d['mean'],d['p50'],d['p90'],d['p95'],row['first_correct_delay_ms']['mean'],row['rewrites_per_100_final_words'],row['rtf'],row['compute_ms_per_chunk']))
    print('native equal',results['native_equal'])


if __name__=='__main__':main()
