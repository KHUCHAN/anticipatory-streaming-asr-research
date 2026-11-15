import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'runs/predict_correct_v1'
sys.path.insert(0,str(ROOT))


def config():return json.loads((HERE/'config.json').read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


state_module=module('corrector_original_state','experiments/fgn_state_v1/model.py')


def word_error(reference,hypothesis):
    ref=reference.split();hyp=hypothesis.split();old=list(range(len(hyp)+1))
    for i,a in enumerate(ref,1):
        new=[i]
        for j,b in enumerate(hyp,1):new.append(min(new[-1]+1,old[j]+1,old[j-1]+(a!=b)))
        old=new
    return old[-1],len(ref)
