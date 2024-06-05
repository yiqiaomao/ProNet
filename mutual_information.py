import sys
import torch
import torch.nn

EPS = sys.float_info.epsilon


def mutual_information(x_img, x_txt):
    _, k = x_img.size()
    p_i_j = compute_joint(x_img, x_txt)
    assert (p_i_j.size() == (k, k))
    temp1 = p_i_j.sum(dim=1).view(k, 1)
    p_i = temp1.expand(k, k).clone()
    temp2 = p_i_j.sum(dim=0).view(1, k)
    p_j = temp2.expand(k, k).clone()
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS
    loss = - p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))
    loss = loss.sum()
    return loss


def compute_joint(x_img, x_txt):
    bn, k = x_img.size()
    assert (x_txt.size(0) == bn and x_txt.size(1) == k)
    p_i_j = x_img.unsqueeze(2) * x_txt.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.
    p_i_j = p_i_j / p_i_j.sum()
    return p_i_j


def cluster_centre_distillation(x, y, z):
    temp_t = 0.25
    x_ = torch.einsum('nd,cd->nc', x, z)
    y_ = torch.einsum('nd,cd->nc', y, z)
    y_ /= temp_t

    x_ = torch.softmax(x_ / temp_t, dim=1)

    # loss computation, use log_softmax for stable computation
    loss = - torch.mul(x_, torch.log_softmax(y_, dim=1)).sum() / x_.shape[0]

    return loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simility_distillation(f_cur,f_past,z):

    features1_prev_task = torch.cat([f_cur, z], dim=0)
    features1_prev_task = torch.nn.functional.normalize(features1_prev_task, dim=1)
    features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), 0.2)
    logits_mask = torch.scatter(
        torch.ones_like(features1_sim).to(device),
        1,
        torch.arange(features1_sim.size(0)).view(-1, 1).to(device),
        0
    )
    logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
    features1_sim = features1_sim - logits_max1.detach()
    row_size = features1_sim.size(0)
    logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
        features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

    features2_prev_task = torch.cat([f_past, z], dim=0)
    features2_prev_task = torch.nn.functional.normalize(features2_prev_task, dim=1)

    features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), 0.2)
    logits_max2, _ = torch.max(features2_sim * logits_mask, dim=1, keepdim=True)
    features2_sim = features2_sim - logits_max2.detach()
    logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
        features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

    loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()

    return loss_distill

