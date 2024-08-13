__all__ = ['train_one_epoch', 'val_one_epoch']


import sys
import math
import torch

from .losses import *
from .util import misc
from .util import lr_sched


def train_one_epoch(model, data_loader, optimizer, device, epoch,
                    loss_scaler, args, opt_func, eval_func, mean_std=None):
    print_freq = args.print_freq
    accum_iter = args.accum_iter
    model_name = args.model_name

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if len(data) == 3:
            # for simulation
            gt, feature, obs_mask = data
            data_mask = None
        else:  # for voronoicnn, aq_highres
            gt, feature, obs_mask, data_mask = data

        gt = gt.to(device, non_blocking=True)
        feature = feature.to(device, non_blocking=True)
        obs_mask = obs_mask.to(device, non_blocking=True)
        if data_mask is not None:
            data_mask = data_mask.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            if 'vitae' in model_name:
                pred, pred_enc = model(feature)
            else:
                pred = model(feature)
                pred_enc = None

            loss4opt = opt_func(gt, pred, obs_mask, data_mask)
            if pred_enc is not None:
                loss4opt_enc = opt_func(gt, pred_enc, obs_mask, data_mask)
                loss4opt = 0.8 * loss4opt + 0.2 * loss4opt_enc
            loss4opt_value = loss4opt.item()

            if mean_std is not None:
                gt = gt * mean_std['std'] + mean_std['mean']
                pred = pred * mean_std['std'] + mean_std['mean']

            loss4eval = eval_func(gt, pred, data_mask)
            loss4eval_value = loss4eval.item()

        if not math.isfinite(loss4opt_value):
            print('Loss is {}, stopping training'.format(loss4opt_value))
            sys.exit(1)

        loss4opt /= accum_iter
        loss_scaler(loss4opt, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        metric_logger.update(opt=loss4opt_value, eval=loss4eval_value)
        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)

    print('>>> Train <<< Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(model, data_loader, device, epoch,
                  args, opt_func, eval_func, mean_std=None):
    print_freq = args.print_freq
    model_name = args.model_name

    model.eval()
    metric_logger = misc.MetricLogger(delimiter='  ')
    header = 'Epoch: [{}]'.format(epoch)

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if len(data) == 3:
            # for simulation dataset
            gt, feature, obs_mask = data
            data_mask = None
        else:  # for voronoicnn, aq_highres
            gt, feature, obs_mask, data_mask = data

        gt = gt.to(device, non_blocking=True)
        feature = feature.to(device, non_blocking=True)
        obs_mask = obs_mask.to(device, non_blocking=True)
        if data_mask is not None:
            data_mask = data_mask.to(device, non_blocking=True)

        if 'vitae' in model_name:
            pred, pred_enc = model(feature)
        else:
            pred = model(feature)
            pred_enc = None

        loss4opt = opt_func(gt, pred, obs_mask, data_mask)
        if pred_enc is not None:
            loss4opt_enc = opt_func(gt, pred_enc, obs_mask, data_mask)
            loss4opt = 0.8 * loss4opt + 0.2 * loss4opt_enc
        loss4opt_value = loss4opt.item()

        if mean_std is not None:
            gt = gt * mean_std['std'] + mean_std['mean']
            pred = pred * mean_std['std'] + mean_std['mean']

        loss4eval = eval_func(gt, pred, data_mask)
        loss4eval_value = loss4eval.item()

        metric_logger.update(opt=loss4opt_value, eval=loss4eval_value)

    print('>>> Val <<< Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
