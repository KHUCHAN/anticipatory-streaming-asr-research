#!/usr/bin/env python3
"""Run frozen evaluation with NumPy-safe aggregate serialization.

The original evaluation completed all timed decodes before encountering a
NumPy-bool JSON error. Keep that frozen code and every raw record unchanged.
The recovery flag aggregates only existing records and finishes reporting.
"""
import argparse
import json
import time

import numpy as np

from common import HERE,OUT,config,sha,write
import evaluate


def scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--finish-existing',action='store_true')
    args=parser.parse_args();cfg=config();original=evaluate.write
    def safe_write(path,value):
        return original(path,json.loads(json.dumps(value,default=scalar)))
    evaluate.write=safe_write
    if args.finish_existing:
        paths=[OUT/'evaluation'/f'{variant}_repeat{repeat}.json' for variant in ['full_encoder',*cfg['variants']] for repeat in range(cfg['timing_repetitions'])]
        assert all(p.exists() for p in paths)
        before={str(p.relative_to(OUT)):sha(p) for p in paths}
        def forbid_decode(*args,**kwargs):raise RuntimeError('Recovery must not rerun inference')
        evaluate.Inference=forbid_decode
    evaluate.main()
    if args.finish_existing:
        assert before=={str(p.relative_to(OUT)):sha(p) for p in paths}
        import verify
        verify.main()
        runner=[json.loads(line) for line in (OUT/'runner.jsonl').read_text().splitlines()]
        recovery=dict(event='reporting_recovery',reason='Convert NumPy aggregate scalars to JSON-native values after all timed decodes completed.',
            frozen_evaluation_source_unchanged=True,raw_records_unchanged=True,inference_rerun=False,
            serializer_sha256=sha(HERE/'finalize.py'),raw_record_sha256=before)
        write(HERE/'artifacts/reporting_recovery.json',recovery)
        with (OUT/'runner.jsonl').open('a') as log:
            log.write(json.dumps(recovery)+'\n')
            log.write(json.dumps(dict(event='done',elapsed_seconds=time.time()-runner[0]['started_unix'],
                reporting_serialization_recovered=True,test_used=False))+'\n')
        import report
        report.main()
        import analyze
        analyze.main()


if __name__=='__main__':main()
