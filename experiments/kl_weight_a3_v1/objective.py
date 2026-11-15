import torch
from torch.nn import functional as F


def losses(model,bridge,batch,cfg,mean,std,with_kl):
    features,fvalid,target,contexts,tokens,valid=batch
    predicted=model.sequence(features,fvalid,target.shape[1]);normalized=(target-mean)/std
    mae=((predicted-normalized).abs().mean(-1)*valid).sum()/valid.sum().clamp_min(1)
    if not with_kl:return mae,None
    with torch.no_grad():pn=bridge.prediction(contexts);teacher=bridge(target,pn)
    temperature=cfg['kl_temperature'];student=bridge(predicted*std+mean,pn)
    kl=F.kl_div(F.log_softmax(student/temperature,-1),F.softmax(teacher/temperature,-1),reduction='none').sum(-1)*temperature**2
    weight=torch.where(tokens==0,cfg['blank_frame_weight'],1.0)*valid
    return mae,(kl*weight).sum()/weight.sum().clamp_min(1e-6)
