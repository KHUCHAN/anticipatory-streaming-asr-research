#!/bin/zsh
# Unattended stage-2 chain. GPU trains the two predictors while one CPU thread collects training-free previews.
cd "${0:A:h}"; PY=../../.local/venv-fgn-state/bin/python; R=../../runs/preview_filler_v1
( $PY predictor.py det > $R/predictor_det.log 2>&1; $PY predictor.py fgn > $R/predictor_fgn.log 2>&1 ) &
GPU=$!
FREE=zero,repeat,mean,fade; LEARNED=det,fgn1,fgnm
$PY collect.py train free $FREE > $R/collect.log 2>&1
$PY collect.py selection free $FREE >> $R/collect.log 2>&1
wait $GPU
$PY collect.py train learned $LEARNED >> $R/collect.log 2>&1
$PY collect.py selection learned $LEARNED >> $R/collect.log 2>&1
$PY collect.py eval all $FREE,$LEARNED >> $R/collect.log 2>&1   # nothing else runs here: timing is taken from this pass
$PY stage2.py > $R/stage2.log 2>&1
touch $R/stage2.finished
