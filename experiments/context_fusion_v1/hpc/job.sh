#!/bin/bash
#SBATCH --job-name=asr-@TAG@
#SBATCH --partition=gpu
#SBATCH --gres=gpu:@GPU@:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=@TIME@
#SBATCH --dependency=afterok:@DEP@
#SBATCH --output=/scratch1/%u/asr/logs/%x-%j.out
# One job: optional LM training, optional fine-tuning, then replay, gate, full-dev WER and comparisons. Code is unpacked per job.
set -uo pipefail
export ASR=/scratch1/$USER/asr TRAIN_SPLITS=train-clean-100,train-clean-360,train-other-500 SA=1 SC=1 DECAY=1 MAX_FRAMES=@MAXF@ GATE_TRAIN=@GTRAIN@
export CTX=@CTX@ CTXLM=@CTXLM@ LMX=@LMX@ LAM=@LAM@ LAMN=@LAMN@ COMP=6
cd $ASR; module purge 2>/dev/null; module load python/3.11.9 2>/dev/null || module load gcc/13.3.0 python/3.11.9; source venv/bin/activate
C=$ASR/code/$SLURM_JOB_ID; mkdir -p $C; export PYTHONPATH=$C
cat > $C/model_def.py <<'P'
__MODEL_DEF__
P
cat > $C/lm.py <<'P'
__LM__
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
[ "@LMMIN@" != 0 ] && python $C/lm.py @LMTAG@ @LMMIN@
[ "@MIN@" != 0 ] && python $C/train.py @TAG@ @DP@ @MIN@ 0.0005 && LAMN=0 python $C/evaluate.py @TAG@_ema ckpt/@TAG@/last.pt ema
[ "@GATE@" != 0 ] && { python -c "import sklearn" 2>/dev/null || pip -q install scikit-learn; python $C/gate.py @GATE@ @GCKPT@ @GKEY@; }
[ "@WERDEV@" != 0 ] && python $C/wer_dev.py @WERDEV@ @GCKPT@ @GKEY@
[ "@CMP@" != 0 ] && python $C/compare.py @CMP@
echo JOB_DONE $(date +%T)
