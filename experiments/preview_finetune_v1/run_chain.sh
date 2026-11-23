#!/bin/zsh
# Waits for fine-tuning, then collects previews (train, selection, eval) and evaluates. Single CPU thread per pass.
cd "$(dirname "$0")"; R=../../runs/preview_finetune_v1; PY=../../.local/venv-fgn-state/bin/python
until [ -f $R/student.json ]; do sleep 60; done
for role in train selection eval; do $PY collect.py $role zero,repeat,ft >> $R/collect.log 2>&1 || { echo "collect $role failed" >> $R/collect.log; touch $R/chain.finished; exit 1; }; done
$PY evaluate.py > $R/evaluate.log 2>&1
touch $R/chain.finished
