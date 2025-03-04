import torch
import torch.nn.functional as F
from train_utils import ce_loss
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


def one_hot(targets, nClass, gpu):
    logits = torch.zeros(targets.size(0), nClass).cuda(gpu)
    return logits.scatter_(1, targets.unsqueeze(1), 1)




def consistency_loss(logits_w1, logits_w2):
    logits_w2 = logits_w2.detach()
    assert logits_w1.size() == logits_w2.size()
    return F.mse_loss(torch.softmax(logits_w1,dim=-1), torch.softmax(logits_w2,dim=-1), reduction='mean')


import torch



def class_contrastive_loss(z_labeled, labels, temperature=0.5):

    device = z_labeled.device
    batch_size = z_labeled.size(0)

    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
    mask = mask - torch.eye(batch_size, device=device)

    sim_matrix = torch.mm(z_labeled, z_labeled.T) / temperature

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()
    exp_sim = torch.exp(sim_matrix)

    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    positive_pairs = (mask * log_prob).sum(dim=1)

    valid_pos_counts = mask.sum(dim=1)
    loss = - (positive_pairs / (valid_pos_counts + 1e-8)).mean()
    return loss


def instance_contrastive_loss(z_unlabeled, temperature=0.5):

    batch_size = z_unlabeled.size(0) // 2
    device = z_unlabeled.device

    z1, z2 = torch.chunk(z_unlabeled, 2, dim=0)

    labels = torch.arange(batch_size, device=device)
    labels = torch.cat([labels, labels])

    sim_matrix = torch.mm(z_unlabeled, z_unlabeled.T) / temperature

    loss = F.cross_entropy(sim_matrix, labels)
    return loss