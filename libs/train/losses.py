__all__ = ['get_loss_func']


import torch
import torch.nn as nn


def weighted_error(error, obs_mask, obs_weight=1.0):
    weight = obs_mask * obs_weight - (obs_mask - 1)
    return error * weight


class L1Loss(nn.Module):
    def __init__(self, obs_weight=1.0):
        super(L1Loss, self).__init__()
        self.obs_weight = obs_weight

    def forward(self, target, pred, obs_mask=None, data_mask=None):
        error = torch.abs(target - pred)

        if obs_mask is not None:
            error = weighted_error(error, obs_mask, self.obs_weight)

        if data_mask is None:
            return torch.mean(error)
        else:
            return torch.sum(error * data_mask) / torch.sum(data_mask)


class L2Loss(nn.Module):
    def __init__(self, obs_weight=1.0):
        super(L2Loss, self).__init__()
        self.obs_weight = obs_weight

    def forward(self, target, pred, obs_mask=None, data_mask=None):
        error = (target - pred) ** 2

        if obs_mask is not None:
            error = weighted_error(error, obs_mask, self.obs_weight)

        if data_mask is None:
            return torch.mean(error)
        else:
            return torch.sum(error * data_mask) / torch.sum(data_mask)


class LogCoshLoss(nn.Module):
    def __init__(self, obs_weight=1.0):
        super(LogCoshLoss, self).__init__()
        self.obs_weight = obs_weight

    def forward(self, target, pred, obs_mask=None, data_mask=None):
        error = target - pred
        error = torch.log(torch.cosh(error + 1e-12))

        if obs_mask is not None:
            error = weighted_error(error, obs_mask, self.obs_weight)

        if data_mask is None:
            return torch.mean(error)
        else:
            return torch.sum(error * data_mask) / torch.sum(data_mask)


class XTanhLoss(nn.Module):
    def __init__(self, obs_weight=1.0):
        super(XTanhLoss, self).__init__()
        self.obs_weight = obs_weight

    def forward(self, target, pred, obs_mask=None, data_mask=None):
        error = target - pred
        error = error * torch.tanh(error)

        if obs_mask is not None:
            error = weighted_error(error, obs_mask, self.obs_weight)

        if data_mask is None:
            return torch.mean(error)
        else:
            return torch.sum(error * data_mask) / torch.sum(data_mask)


class XSigmoidLoss(nn.Module):
    def __init__(self, obs_weight=1.0):
        super(XSigmoidLoss, self).__init__()
        self.obs_weight = obs_weight

    def forward(self, target, pred, obs_mask=None, data_mask=None):
        error = target - pred
        error = 2 * error / (1 + torch.exp(-error)) - error

        if obs_mask is not None:
            error = weighted_error(error, obs_mask, self.obs_weight)

        if data_mask is None:
            return torch.mean(error)
        else:
            return torch.sum(error * data_mask) / torch.sum(data_mask)


class L2NormLoss(nn.Module):

    def forward(self, gt, pred, data_mask=None):
        loss = 0
        diff = gt - pred
        for d, g in zip(diff, gt):
            loss += self.l2_norm(d, data_mask) / self.l2_norm(g, data_mask)
        return loss / len(gt)

    @staticmethod
    def l2_norm(x, data_mask=None):
        if data_mask is None:
            return torch.sqrt(torch.sum(x ** 2))
        else:
            return torch.sqrt(torch.sum(x ** 2 * data_mask))


def get_loss_func(loss_type='mse', obs_weight=1.0):
    assert loss_type in ['mae', 'mse', 'l2norm', 'logcosh', 'xtanh', 'xsigmoid'], \
        'loss_type should be one of mae, mse, l2norm, logcosh, xtanh or xsigmoid'

    if loss_type == 'mae':
        loss_func = L1Loss(obs_weight)
    elif loss_type == 'mse':
        loss_func = L2Loss(obs_weight)
    elif loss_type == 'l2norm':
        loss_func = L2NormLoss()
    elif loss_type == 'logcosh':
        loss_func = LogCoshLoss(obs_weight)
    elif loss_type == 'xtanh':
        loss_func = XTanhLoss(obs_weight)
    else:  # loss_type == 'xsigmoid'
        loss_func = XSigmoidLoss(obs_weight)

    return loss_func
