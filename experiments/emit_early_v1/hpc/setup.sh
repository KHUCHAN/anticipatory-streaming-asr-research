#!/bin/bash
#SBATCH --job-name=asr-setup
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch1/%u/%x-%j.out
# Environment, upstream sources, public weights, LibriSpeech and features. Everything stays under /scratch1/$USER/asr.
set -uo pipefail
W=/scratch1/$USER/asr; mkdir -p $W/src $W/data $W/logs $W/ckpt $W/code; cd $W
curl -sI -m 20 https://huggingface.co | head -1 || { echo NO_INTERNET; exit 3; }
module purge 2>/dev/null; module load python/3.11.9 2>/dev/null || module load gcc/13.3.0 python/3.11.9; python3 --version
( cd data; for s in dev-clean dev-other train-clean-100; do
    [ -d LibriSpeech/$s ] && continue
    for host in us.openslr.org www.openslr.org openslr.elda.org; do curl -fsL -C - --retry 3 -o $s.tar.gz https://$host/resources/12/$s.tar.gz && break; done
    tar xzf $s.tar.gz && rm $s.tar.gz; echo downloaded $s $(date +%T)
  done ) > logs/download.log 2>&1 &
[ -d venv ] || python3 -m venv venv; source venv/bin/activate; pip -q install --upgrade pip
python - > k2.choice <<'P'
import re,urllib.request
html=urllib.request.urlopen('https://k2-fsa.github.io/k2/cuda.html',timeout=60).read().decode();c=[]
for m in re.finditer(r'href="([^"]*?k2-[0-9.]+dev(\d+)(?:\+|%2B)cuda([0-9.]+)\.torch([0-9.]+)-cp311-cp311-[^"]*?linux[^"]*?\.whl)"',html):
    if m.group(3).startswith('12.'):c.append((tuple(map(int,m.group(4).split('.'))),m.group(2),m.group(3),m.group(1)))
if not c:print('NONE',[l for l in html.splitlines() if 'cp311' in l][:3])
else:
    t,d,cuda,url=max(c);print('.'.join(map(str,t)),cuda,url if url.startswith('http') else 'https://k2-fsa.github.io/k2/'+url)
P
cat k2.choice | cut -c1-300; read TV CV URL < k2.choice; [ "$TV" = NONE ] && exit 4
pip -q install torch==$TV --index-url https://download.pytorch.org/whl/cu${CV//./} || exit 5
pip -q install "$URL" numpy sentencepiece soundfile kaldi-native-fbank || exit 6
B=https://raw.githubusercontent.com/k2-fsa/icefall/179b5eef823e8c2c38a7972f4762e22c5dd23973/egs/librispeech/ASR
for f in pruned_transducer_stateless7_streaming/zipformer.py pruned_transducer_stateless7/scaling.py pruned_transducer_stateless7/decoder.py pruned_transducer_stateless7/joiner.py pruned_transducer_stateless7/optim.py pruned_transducer_stateless7/model.py; do curl -fsL -o src/$(basename $f) $B/$f || exit 7; done
H=https://huggingface.co/desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small/resolve/main
[ -s src/pretrained.pt ] || curl -fsL -o src/pretrained.pt $H/exp/pretrained.pt; [ -s src/bpe.model ] || curl -fsL -o src/bpe.model $H/data/lang_bpe_500/bpe.model
ls -l src | awk '{print $5,$9}'; grep -n "^import\|^from" src/optim.py src/model.py | tr '\n' ';'; echo
python - <<'P'
import inspect,torch,k2
print('torch',torch.__version__,'k2',k2.__version__ if hasattr(k2,'__version__') else '?')
for f in ['rnnt_loss_smoothed','rnnt_loss_pruned','get_rnnt_prune_ranges','do_rnnt_pruning']:print(f,list(inspect.signature(getattr(k2,f)).parameters))
P
cat > code/prep_features.py <<'P'
__PREP_FEATURES__
P
wait; cat logs/download.log
for s in dev-clean dev-other train-clean-100; do [ -s data/index_$s.json ] || python code/prep_features.py $W/data $s 16; done
ls -l data | awk '{print $5,$9}'; echo SETUP_DONE $(date +%T)
