# training/loss.py

import torch.nn as nn

stress_loss = nn.MSELoss()
force_loss = nn.MSELoss()


def compute_loss(pred_stress,true_stress,pred_force,true_force):

    loss_s = stress_loss(pred_stress,true_stress)

    loss_f = force_loss(pred_force,true_force)

    loss = loss_s + 10*loss_f

    return loss
