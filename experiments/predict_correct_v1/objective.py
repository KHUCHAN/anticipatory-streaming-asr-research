"""Teacher outputs supervise losses only, never recurrent inference inputs."""
import torch
from torch.nn import functional as F


def objective(model,bridge,batch,cfg,mean,std):
    features,fvalid,target,contexts,tokens,valid,_=batch
    predicted=model.sequence(features,fvalid,target.shape[1])
    normalized=(target-mean)/std
    mae=(predicted-normalized).abs().mean(-1)
    representation=(mae*valid).sum()/valid.sum().clamp_min(1)
    # Free-running inference does not use these teacher contexts. They are
    # offline distillation labels, like the full encoder's output targets.
    with torch.no_grad():
        pn=bridge.prediction(contexts)
        teacher=bridge(target,pn)
    student=bridge(predicted*std+mean,pn)
    temperature=cfg['distillation_temperature']
    kl=F.kl_div(F.log_softmax(student/temperature,-1),F.softmax(teacher/temperature,-1),reduction='none').sum(-1)*(temperature**2)
    weight=torch.where(tokens==0,cfg['blank_frame_weight'],1.)*valid
    distillation=(kl*weight).sum()/weight.sum().clamp_min(1e-6)
    loss=cfg['representation_weight']*representation+cfg['distillation_weight']*distillation
    return loss,dict(mae=representation.detach(),kl=distillation.detach())
