"""Shared paths and helpers; reuses the frozen A2 feature/decoder adapters without modifying them."""
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'runs/preview_finetune_v1'
sys.path.insert(0,str(ROOT/'experiments/predict_correct_a2_v1'))
import common as a2  # noqa: E402

features=a2.features;write=a2.write;sha=a2.sha


def config():return json.loads((HERE/'config.json').read_text())


def rows(cfg):
    """Fixed-seed subset of the existing dev evaluation role, sorted for a stable order."""
    import random
    a2cfg=json.loads((ROOT/cfg['a2_config']).read_text())
    index=json.loads((ROOT/a2cfg['input_cache']/'index.json').read_text())
    ids=set(json.loads((ROOT/cfg['protocol']).read_text())['evaluation_ids']);out=[]
    for split in cfg['splits']:
        own=sorted([r for r in index if r['utterance_id'] in ids and r['split']==split],key=lambda r:r['utterance_id'])
        random.Random(cfg['seed']).shuffle(own);out+=sorted(own[:cfg['utterances_per_split']],key=lambda r:r['utterance_id'])
    return out


def edit_pairs(reference,hypothesis):
    """Levenshtein backtrace. Returns (errors, [(hypothesis index, reference index)] for exact matches)."""
    a=reference;b=hypothesis;n=len(a);m=len(b)
    d=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):d[i][0]=i
    for j in range(m+1):d[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]!=b[j-1]))
    i=n;j=m;pairs=[]
    while i>0 and j>0:
        if d[i][j]==d[i-1][j-1]+(a[i-1]!=b[j-1]):
            if a[i-1]==b[j-1]:pairs.append((j-1,i-1))
            i-=1;j-=1
        elif d[i][j]==d[i-1][j]+1:i-=1
        else:j-=1
    return d[n][m],pairs[::-1]


def role_rows(cfg,role):
    """eval: stage-1 evaluation subset; selection: 128 dev model-selection utterances; train: fixed-seed train subset."""
    import random
    if role=='eval':return rows(cfg)
    a2cfg=json.loads((ROOT/cfg['a2_config']).read_text());index=json.loads((ROOT/a2cfg['input_cache']/'index.json').read_text())
    if role=='selection':
        ids=set(json.loads((ROOT/cfg['protocol']).read_text())['selection_ids']);return sorted([r for r in index if r['utterance_id'] in ids],key=lambda r:r['utterance_id'])
    s=cfg['stage2'];own=sorted([r for r in index if r['split']==s['train_split']],key=lambda r:r['utterance_id'])
    random.Random(s['train_seed']).shuffle(own);return sorted(own[:s['train_utterances']],key=lambda r:r['utterance_id'])
