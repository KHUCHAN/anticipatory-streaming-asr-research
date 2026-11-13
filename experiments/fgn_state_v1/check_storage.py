"""Audit actual retained tensor storages separately from logical element bytes."""
import json,time
import numpy as np
import torch
import collect
from data import ROOT,OUT,selections


@torch.inference_mode()
def main():
    collect.initialize();_,rows=selections(collect.CFG);row=rows[0]
    with np.load(ROOT/f"data/processed/zipformer_forecasts/{row['split']}/{row['utterance_id']}.npz") as v:
        states=torch.from_numpy(v['enc'][:16].astype(np.float32)).view(2,8,512)
    report={};started=time.time()
    for kind in collect.KINDS:
        runtime=collect.make_runtime(kind);snapshots=[]
        for i in range(1875):
            runtime.observe(states[i%2])
            if i in [0,1,9,99,999,1874]:
                tensors=[getattr(runtime,'previous',None),runtime.current,*getattr(runtime,'memories',[])]
                tensors=[x for x in tensors if x is not None]
                storages={x.untyped_storage().data_ptr():x.untyped_storage().nbytes() for x in tensors}
                snapshots.append(dict(chunk=i+1,logical_bytes=sum(x.numel()*x.element_size() for x in tensors),unique_storage_bytes=sum(storages.values())))
        assert len({r['unique_storage_bytes'] for r in snapshots if r['chunk']>=10})==1
        report[kind]=dict(snapshots=snapshots,storage_peak_bytes=max(r['unique_storage_bytes'] for r in snapshots))
    assert report['state']['storage_peak_bytes']==33792
    assert report['ensemble']['storage_peak_bytes']==34816
    assert report['persistence']['storage_peak_bytes']==16384
    result=dict(status='pass',observed_seconds_equivalent=600,results=report,seconds=time.time()-started,
                scope='Actual unique retained tensor storage buffers, including unused view storage, not process RSS, weights, allocator pools or temporary activations. Repeated observations test storage retention, not long-stream recognition quality.')
    (OUT/'storage_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
