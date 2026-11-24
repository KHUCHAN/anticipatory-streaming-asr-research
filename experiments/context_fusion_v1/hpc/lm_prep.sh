#!/bin/bash
#SBATCH --job-name=asr-lmprep
#SBATCH --partition=main
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch1/%u/asr/logs/%x-%j.out
# Downloads the LibriSpeech LM corpus and writes the LM token file (training transcripts first). Falls back to the transcripts alone if the download fails.
set -uo pipefail
export ASR=/scratch1/$USER/asr; cd $ASR; module purge 2>/dev/null; module load python/3.11.9 2>/dev/null || module load gcc/13.3.0 python/3.11.9; source venv/bin/activate
C=$ASR/code/$SLURM_JOB_ID; mkdir -p $C
cat > $C/lm_prep.py <<'P'
__LM_PREP__
P
F=data/librispeech-lm-norm.txt.gz
[ -s $F ] || for host in us.openslr.org www.openslr.org openslr.elda.org; do curl -fsL -C - --retry 3 -o $F https://$host/resources/11/librispeech-lm-norm.txt.gz && break; done
ls -l $F
if gzip -t $F 2>/dev/null; then python $C/lm_prep.py 32 $F; else echo NO_CORPUS; python $C/lm_prep.py 32; fi
ls -l data/lm_tokens.u16; echo JOB_DONE $(date +%T)
