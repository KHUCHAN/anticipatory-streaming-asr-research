#!/bin/bash
#SBATCH --job-name=asr-@TAG@
#SBATCH --partition=gpu
#SBATCH --gres=gpu:@GPU@:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=@TIME@
#SBATCH --output=/scratch1/%u/asr/logs/%x-%j.out
# Fine-tune with a delay penalty, then replay the fixed dev subset. Code is unpacked per job.
set -uo pipefail
export ASR=/scratch1/$USER/asr TRAIN_SPLITS=@SPLITS@ SA=@SA@ SC=@SC@ DECAY=@DECAY@ GATE_TRAIN=@GTRAIN@; cd $ASR; module purge 2>/dev/null; module load python/3.11.9 2>/dev/null || module load gcc/13.3.0 python/3.11.9; source venv/bin/activate
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
cat > $C/align_eval.b64 <<'P'
__ALIGN_EVAL__
P
mv -f $C/align_eval.b64 code/align_eval.b64
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ "@BASE@" = 1 ] && python $C/evaluate.py base
[ "@MIN@" != 0 ] && python $C/train.py @TAG@ @DP@ @MIN@ @LR@ && python $C/evaluate.py @TAG@ ckpt/@TAG@/last.pt && python $C/evaluate.py @TAG@_ema ckpt/@TAG@/last.pt ema
[ "@GATE@" != 0 ] && { python -c "import sklearn" 2>/dev/null || pip -q install scikit-learn; python $C/gate.py @GATE@ @GCKPT@ @GKEY@; }
[ "@WERDEV@" != 0 ] && python $C/wer_dev.py @WERDEV@ @GCKPT@ @GKEY@
echo JOB_DONE $(date +%T)
