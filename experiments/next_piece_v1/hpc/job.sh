#!/bin/bash
#SBATCH --job-name=asr-@TAG@
#SBATCH --partition=gpu
#SBATCH --gres=gpu:@GPU@:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=@TIME@
#SBATCH --dependency=afterok:@DEP@
#SBATCH --output=/scratch1/%u/asr/logs/%x-%j.out
# One job: optional training (fine-tuning with the next-piece loss, or the head alone on a frozen recognizer), then replay, gate, full-dev WER and comparisons. Code is unpacked per job.
# The job fails (non-zero exit) when any step fails, so that jobs depending on a smoke run do not start after a broken one.
set -uo pipefail
export ASR=/scratch1/$USER/asr TRAIN_SPLITS=train-clean-100,train-clean-360,train-other-500 SA=1 SC=1 DECAY=1 MAX_FRAMES=@MAXF@ GATE_TRAIN=@GTRAIN@ GATE_SELECT=@GSEL@ WER_LIMIT=@WERLIMIT@
export CTX=@CTX@ CTXLM=@CTXLM@ NXW=@NXW@ FREEZE=@FREEZE@ INIT=@INIT@ INITKEY=@INITKEY@ ANT=@ANT@ LMX=@LMX@ LAM=0 LAMN=0 COMP=0
cd $ASR; module purge 2>/dev/null; module load python/3.11.9 2>/dev/null || module load gcc/13.3.0 python/3.11.9; source venv/bin/activate
C=$ASR/code/$SLURM_JOB_ID; mkdir -p $C; export PYTHONPATH=$C
cat > $C/model_def.py <<'P'
__MODEL_DEF__
P
cat > $C/train.py <<'P'
__TRAIN__
P
cat > $C/evaluate.py <<'P'
__EVALUATE__
P
cat > $C/gate.py <<'P'
__GATE__
P
cat > $C/wer_dev.py <<'P'
__WER_DEV__
P
cat > $C/compare.py <<'P'
__COMPARE__
P
cat > $C/align_eval.b64 <<'P'
__ALIGN_EVAL__
P
mv -f $C/align_eval.b64 code/align_eval.b64
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
rc=0
if [ "@MIN@" != 0 ]; then python $C/train.py @TAG@ @DP@ @MIN@ 0.0005 && python $C/evaluate.py @TAG@_ema ckpt/@TAG@/last.pt ema || rc=1; fi
if [ "@GATE@" != 0 ]; then { python -c "import sklearn" 2>/dev/null || pip -q install scikit-learn; } && python $C/gate.py @GATE@ @GCKPT@ @GKEY@ || rc=1; fi
if [ "@WERDEV@" != 0 ]; then python $C/wer_dev.py @WERDEV@ @GCKPT@ @GKEY@ || rc=1; fi
if [ "@CMP@" != 0 ]; then python $C/compare.py @CMP@ || rc=1; fi
if [ "@CMPB@" != 0 ]; then python $C/compare.py @CMPB@ || rc=1; fi
echo JOB_DONE rc=$rc $(date +%T)
exit $rc
