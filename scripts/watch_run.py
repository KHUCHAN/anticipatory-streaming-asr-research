#!/usr/bin/env python3
"""Low-cost training watcher. Blocks up to --wait seconds, then prints ONE status line.

First word of the output: OK (still healthy), DONE (finished) or ALERT (needs attention).
Read-only: it never touches the run.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path


def status(run, pattern, stall_min):
    log = run / 'log.jsonl'
    recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()] if log.exists() else []
    trains = [r for r in recs if r['event'] == 'train']
    evals = [r for r in recs if r['event'] == 'eval']
    summary = ''
    if trains:
        t = trains[-1]
        summary += f"epoch={t['epoch']} step={t['step']} loss={t['loss']} acc={t.get('acc')} xRT={t.get('x_realtime')} mem={t.get('mem_gb')}GB"
    if evals:
        summary += f" | last_eval(ep{evals[-1]['epoch']})={json.dumps(evals[-1].get('wer', evals[-1].get('metrics')))}"
    if any(r['event'] == 'done' for r in recs):
        return 'DONE', summary
    if any(r['event'] == 'abort' for r in recs):
        return 'ALERT', 'run aborted itself: ' + summary
    alive = subprocess.run(['pgrep', '-f', pattern], capture_output=True).returncode == 0
    if not alive:
        tail = ''
        out = run / 'stdout.log'
        if out.exists():
            tail = ' || stdout tail: ' + ' / '.join(out.read_text().splitlines()[-6:])[-900:]
        return 'ALERT', 'process not running and no done event. ' + summary + tail
    if log.exists() and time.time() - log.stat().st_mtime > stall_min * 60:
        return 'ALERT', f'no log progress for {stall_min} min. ' + summary
    bad = sum(r['event'] == 'nonfinite' for r in recs)
    if bad >= 10:
        return 'ALERT', f'{bad} non-finite losses so far. ' + summary
    if len(trains) >= 3 and (all(t.get('sec_per_step', 0) > 2 for t in trains[-3:]) or trains[-1].get('mps_gb', 0) > 12):
        return 'ALERT', f"slow or memory-heavy: sec_per_step={trains[-1].get('sec_per_step')} mps_gb={trains[-1].get('mps_gb')}. " + summary
    if len(trains) >= 30:
        recent = sum(t['loss'] for t in trains[-5:]) / 5
        best = min(sum(t['loss'] for t in trains[i:i + 5]) / 5 for i in range(0, len(trains) - 9))
        if recent > 1.6 * best:
            return 'ALERT', f'loss diverging: recent={recent:.3f} best={best:.3f}. ' + summary
    return 'OK', summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--pattern', required=True, help='pgrep -f pattern of the training process')
    ap.add_argument('--wait', type=int, default=540)
    ap.add_argument('--stall-min', type=int, default=20)
    a = ap.parse_args()
    run = Path(a.run)
    end = time.time() + a.wait
    while True:
        state, msg = status(run, a.pattern, a.stall_min)
        if state != 'OK' or time.time() >= end:
            print(state, msg)
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
