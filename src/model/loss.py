import torch
import random

"""
  Binary cross-entropy loss function.
 """
def bce_loss(input, target):
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

"""
  GAN Generator loss
"""
def gan_g_loss(scores_fake):
        y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)

"""
 GAN Discriminator loss
"""
def gan_d_loss(scores_real, scores_fake):
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


"""
L2 loss (L2 norm)- how far off is our prediction from the ground truth
"""

def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)

"""
Displacement error - Euclidean displacement error (x-x1)^2 
Used for average over all points

"""
def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):

    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

"""
Final Displacement error (The final position of the object in cosideration)- Euclidean displacement error (x-x1)^2
"""
def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)
