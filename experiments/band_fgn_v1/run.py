import argparse,json,subprocess,sys,time
from common import OUT,HERE,config,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','evaluate','all'],default='all');a=p.parse_args();cfg=config()
    stages=[]
    if a.stage in ['train','all']:stages += [(v,['train.py','--variant',v]) for v in cfg['variants']]
    if a.stage in ['evaluate','all']:stages += [('smoke',['collect.py','--part','dev','--limit','2','--workers','1','--smoke']),
      ('dev',['collect.py','--part','dev','--workers','4']),('calibration',['analyze.py','calibrate']),
      ('test',['collect.py','--part','test','--workers','4']),('analysis',['analyze.py','evaluate']),('report',['report.py']),('verification',['verify.py'])]
    started=time.time();records=[]
    for name,args in stages:
        t=time.time();print(json.dumps(dict(event='stage_start',stage=name)),flush=True)
        with (OUT/(name+'.pipeline.log')).open('a',buffering=1) as log:
            result=subprocess.run([sys.executable,'-u',str(HERE/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(stage=name,returncode=result.returncode,seconds=time.time()-t,ended_unix=time.time()))
        write(OUT/('queue_'+a.stage+'.json'),records);print(json.dumps(records[-1]),flush=True)
        if result.returncode:print((OUT/(name+'.pipeline.log')).read_text()[-5000:],flush=True);raise SystemExit(result.returncode)
    write(OUT/('complete_'+a.stage+'.json'),dict(stages=records,seconds=time.time()-started,complete=True))
if __name__=='__main__':main()
