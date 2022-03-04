
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from collections import Counter
from copy import deepcopy

def l2_norm(r):
    r /= torch.norm(r, dim=1, keepdim=True) + 1e-8
    return r

class FlexMatch(nn.Module):

    def __init__(self, args):
        super(FlexMatch, self).__init__()
        self.num_classes = args.num_classes
        self.momentum = args.momentum
        self.conf_cutoff = args.conf_cutoff
        self.flex_iter = args.flex_iter 
        self.train_iter = args.iter_per_epoch
        self.thresh_warmup = args.thresh_warmup

    def forward(self, model, selected_label, classwise_acc, x_ul_w, x_ul_s, n_l, n_ul):

        pseudo_counter = Counter(selected_label.tolist())
        if max(pseudo_counter.values()) < len(self.unlabeled_loader):
            if self.thresh_warmup:
                for i in range(self.num_classes):
                    classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
            else:
                wo_negative_one = deepcopy(pseudo_counter)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                for i in range(self.num_classes):
                    classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

        preds_w = model(x_ul_w)
        preds_s = model(x_ul_s)

        self.conf_cutoff = 0

        loss, select, pseudo_lb = consistency_loss(preds_w, preds_s, classwise_acc
                                            , 'ce', self.conf_cutoff)



        return loss, select, pseudo_lb


def consistency_loss(logits_s, logits_w, class_acc, name='ce',
                     p_cutoff=0.0):
    assert name in ['ce', 'L2']
    ce_loss = F.cross_entropy()
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)

        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx]))).float()  # convex
        select = max_probs.ge(p_cutoff).long()
        masked_loss = ce_loss(logits_s, max_idx, reduction='none') * mask
        
        return masked_loss.mean(), select, max_idx.long()

    else:
        assert Exception('Not Implemented consistency_loss')