#!/bin/bash
#SBATCH --job-name=asr-prep-@NAME@
#SBATCH --partition=main
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch1/%u/asr/logs/%x-%j.out
# Downloads more LibriSpeech training splits and computes the same features.
set -uo pipefail
W=/scratch1/$USER/asr; cd $W; module purge 2>/dev/null; module load python/3.11.9 2>/dev/null || module load gcc/13.3.0 python/3.11.9; source venv/bin/activate
C=$W/code/$SLURM_JOB_ID; mkdir -p $C
cat > $C/prep_features.py <<'P'
__PREP_FEATURES__
P
for s in @SPLITS@; do
  [ -s data/index_$s.json ] && continue
  ( cd data; [ -d LibriSpeech/$s ] || { for host in us.openslr.org www.openslr.org openslr.elda.org; do curl -fsL -C - --retry 3 -o $s.tar.gz https://$host/resources/12/$s.tar.gz && break; done; tar xzf $s.tar.gz && rm $s.tar.gz; } )
  echo downloaded $s $(date +%T); python $C/prep_features.py $W/data $s 32
done
ls -l data | awk '{print $5,$9}'; echo JOB_DONE $(date +%T)
