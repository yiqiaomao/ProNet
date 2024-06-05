from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch.nn as nn
import torch as th
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import torch


class En_decoder(nn.Module):
    def __init__(self, input_dimension, feature_dimension, project_dimension):
        super(En_decoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dimension),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(feature_dimension, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, input_dimension),
            nn.ReLU(inplace=True)
        )
        self.projection = nn.Sequential(
            nn.Linear(feature_dimension, project_dimension),
            nn.BatchNorm1d(project_dimension),
            nn.ReLU(inplace=True),
        )
        self.projection1 = nn.Sequential(
            nn.Linear(input_dimension, project_dimension),
            nn.BatchNorm1d(project_dimension),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f = self.encoder(x)
        p = self.projection(f)
        x_dec = self.decoder(f)
        return f, p, x_dec


class Encoder_shared(nn.Module):
    def __init__(self, project_dimension, cluster_num):
        super(Encoder_shared, self).__init__()
        self.encoder_S = nn.Sequential(
            nn.Linear(project_dimension, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, project_dimension),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(project_dimension)
        )
        self.projection = nn.Sequential(
            nn.Linear(project_dimension, project_dimension),
            nn.BatchNorm1d(project_dimension),
            nn.ReLU(inplace=True),
        )
        self.clustering = nn.Sequential(nn.Linear(project_dimension, cluster_num))

    def forward(self, x1, x2):
        f1 = self.encoder_S(x1)
        f2 = self.encoder_S(x2)
        p1 = self.projection(f1)
        p2 = self.projection(f2)
        clustering = torch.softmax(self.clustering(p1), dim=1)
        clustering_nn = torch.softmax(self.clustering(p2), dim=1)
        return f1, f2, p1, p2, clustering, clustering_nn




def UD_constraint(classer):
    CL = classer.detach().cpu().numpy()
    N, K = CL.shape
    CL = CL.T
    r = np.ones((K, 1)) / K
    c = np.ones((N, 1)) / N
    CL **= 10
    inv_K = 1. / K
    inv_N = 1. / N
    err = 1e3
    _counter = 0
    while err > 1e-2 and _counter < 75:
        r = inv_K / (CL @ c)
        c_new = inv_N / (r.T @ CL).T
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    CL *= np.squeeze(c)
    CL = CL.T
    CL *= np.squeeze(r)
    CL = CL.T
    try:
        argmaxes = np.nanargmax(CL, 0)
    except:
        argmaxes = np.argmax(CL, 0)
    newL = th.LongTensor(argmaxes)
    return newL


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    import math

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class NTXentLoss(nn.Module):
    """ NTXentLoss
    Args:
        tau: The temperature parameter.
    """

    def __init__(self, bs, tau=0.5, cos_sim=True, gpu=True, eps=1e-8):
        super(NTXentLoss, self).__init__()
        self.name = 'NTXentLoss_Org'
        self.tau = tau
        self.use_cos_sim = cos_sim
        self.gpu = gpu
        self.eps = eps

        if cos_sim:
            self.cosine_similarity = nn.CosineSimilarity(dim=-1)
            self.name += '_CosSim'

        # Get pos and neg mask

        print(self.name)

    def forward(self, zi, zj, target=None):
        '''
        input: {'zi': out_feature_1, 'zj': out_feature_2}
        target: one_hot lbl_prob_mat
        '''
        # zi, zj = F.normalize(input['zi'], dim=1), F.normalize(input['zj'], dim=1)
        bs = zi.shape[0]
        self.pos_mask, self.neg_mask = get_pos_and_neg_mask(bs)

        if self.gpu:
            self.pos_mask = self.pos_mask.cuda()
            self.neg_mask = self.neg_mask.cuda()

        z_all = torch.cat([zi, zj], dim=0)  # input1,input2: z_i,z_j
        # [2*bs, 2*bs] -  pairwise similarity
        if self.use_cos_sim:
            sim_mat = self.cosine_similarity(
                z_all.unsqueeze(1), z_all.unsqueeze(0)) / self.tau  # s_(i,j)
        else:
            sim_mat = torch.mm(z_all, z_all.t().contiguous()) / self.tau  # s_(i,j)

        sim_pos = torch.exp(sim_mat.masked_select(self.pos_mask).view(2*bs).clone())
        # [2*bs, 2*bs-1]
        sim_neg = torch.exp(sim_mat.masked_select(self.neg_mask).view(2*bs, -1).clone())

        # Compute loss
        loss = (- torch.log(sim_pos / (sim_neg.sum(dim=-1) + self.eps))).mean()

        return loss

def get_pos_and_neg_mask(bs):
    ''' Org_NTXentLoss_mask '''
    zeros = torch.zeros((bs, bs), dtype=torch.uint8)
    eye = torch.eye(bs, dtype=torch.uint8)
    pos_mask = torch.cat([
        torch.cat([zeros, eye], dim=0), torch.cat([eye, zeros], dim=0),
    ], dim=1)
    neg_mask = (torch.ones(2*bs, 2*bs, dtype=torch.uint8) - torch.eye(
        2*bs, dtype=torch.uint8))
    pos_mask = mask_type_transfer(pos_mask)
    neg_mask = mask_type_transfer(neg_mask)
    return pos_mask, neg_mask

def mask_type_transfer(mask):
    mask = mask.type(torch.bool)
    # mask = mask.type(torch.uint8)
    return mask


class Pseudo_Label_Loss(nn.Module):

    def __init__(self, bs, tau=0.5, gpu=True, eps=1e-8):
        super(Pseudo_Label_Loss, self).__init__()
        self.tau = tau
        self.gpu = gpu
        self.eps = eps
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_pos_and_neg_mask(self, bs1, label):
        eye = torch.eye(bs1, dtype=torch.uint8).to(self.device)
        label = label.unsqueeze(dim=1)
        pos_mask = label.eq(label.t()) ^ eye.type(torch.bool)
        neg_mask = ~pos_mask ^ eye.type(torch.bool)
        return pos_mask, neg_mask

    def forward(self, feature, label):
        label = torch.argmax(label, dim=1)
        bs1, bs2 = feature.shape[0], label.shape[0]
        assert bs1 == bs2
        self.pos_mask, self.neg_mask = self.get_pos_and_neg_mask(bs1, label)

        sim_mat = self.cosine_similarity(
            feature.unsqueeze(1), feature.unsqueeze(0)) / self.tau  # s_(i,j)
        sim_pos = torch.exp(sim_mat.masked_select(self.pos_mask).clone())
        # [2*bs, 2*bs-1]
        sim_neg = torch.exp(sim_mat.masked_select(self.neg_mask).clone())

        # Compute loss
        loss = (- torch.log(sim_pos / (sim_neg.sum(dim=-1) + self.eps))).mean()

        return loss


class loss_NTXent_Class(nn.Module):

    def __init__(self, bs, tau=0.5, gpu=True, eps=1e-8):
        super(loss_NTXent_Class, self).__init__()
        self.tau = tau
        self.gpu = gpu
        self.eps = eps
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_pos_and_neg_mask(self, bs1, label1, label2):
        label1 = torch.argmax(label1, dim=1)
        label2 = torch.argmax(label2, dim=1)
        eye = torch.eye(bs1, dtype=torch.uint8).to(self.device)

        zeros = torch.zeros((bs1, bs1), dtype=torch.uint8).to(self.device)

        label1_, label2_ = self.mapL2toL1(label1, label2)
        # label1_, label2_ = label1, label2
        label2_ = torch.tensor(label2_).to(self.device)
        label1 = label1_.unsqueeze(dim=1)
        label2 = label2_.unsqueeze(dim=1)
        pos_mask1 = label1.eq(label2.t()) | eye.type(torch.bool)
        pos_mask = torch.cat([
            torch.cat([zeros, pos_mask1], dim=0), torch.cat([pos_mask1, zeros], dim=0),
        ], dim=1).type(torch.bool)

        label_sum = torch.cat([label1_, label2_], 0)
        label_sum = label_sum.unsqueeze(dim=1)
        neg_mask = ~label_sum.eq(label_sum.t())
        return pos_mask, neg_mask

    def mapL2toL1(self, label1, label2):
        D = max(label1.max(), label2.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        try:
            for i in range(label1.size(0)):
                w[label1[i], label2[i]] += 1
        except:
            for i in range(label1.size):
                w[label1[i], label2[i]] += 1
        from scipy.optimize import linear_sum_assignment
        ind = linear_sum_assignment(w.max() - w)
        ind = np.asarray(ind)
        ind = np.transpose(ind)

        map1 = {a2: a1 for a1, a2 in ind}
        try:
            label2 = [map1[l] for l in label2.cpu().numpy().squeeze()]
        except:
            label2 = [map1[l] for l in label2.squeeze()]
        return label1, label2

    def forward(self, label1, label2):

        bs1, bs2 = label1.shape[0], label2.shape[0]
        assert bs1 == bs2
        self.pos_mask, self.neg_mask = self.get_pos_and_neg_mask(bs1, label1, label2)

        z_all = torch.cat([label1, label2], dim=0)

        sim_mat = self.cosine_similarity(
            z_all.unsqueeze(1), z_all.unsqueeze(0)) / self.tau  # s_(i,j)
        sim_mat = torch.log(sim_mat)
        sim_pos = torch.exp(sim_mat.masked_select(self.pos_mask).clone())
        # [2*bs, 2*bs-1]
        sim_neg = torch.exp(sim_mat.masked_select(self.neg_mask).clone())

        # Compute loss
        loss = (- torch.log(sim_pos / (sim_neg.sum(dim=-1) + self.eps))).mean()

        return loss



class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum