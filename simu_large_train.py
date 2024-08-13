import os
import json
import time
import torch
import random
import argparse
import datetime
import numpy as np
import libs.models as models_vitae
import libs.train.util.misc as misc

from libs.train import *
from libs.models import *
from libs.dataset import *
from libs.train.util.misc import NativeScalerWithGradNormCount as NativeScaler

try:
    import timm
    assert timm.__version__ == '0.3.2'
    import timm.optim.optim_factory as optim_factory
except Exception:
    import sys
    sys.exit(1)


def get_args_parser():

    parser = argparse.ArgumentParser('ViT-AE training for simulation dataset', add_help=False)

    # Dataset parameters
    parser.add_argument('--data_root', type=str, help='dataset path')
    parser.add_argument('--data_dir', type=str, help='dataset dir name')
    parser.add_argument('--train_prop', type=float, help='proportion of training data')
    parser.add_argument('--val_prop', type=float, help='proportion of validation data')
    parser.add_argument('--augment', action='store_true', help='augment training data')

    # Model parameters
    parser.add_argument('--model_root', default='./outputs/models', type=str,
                        help='root path where to save, empty for no saving')
    parser.add_argument('--model_name', default='vitae_base', type=str,
                        metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', type=int, help='images input size')
    parser.add_argument('--patch_size', type=int, help='patch size in vit encoder')
    parser.add_argument('--input_type', type=str, default='sparse', help='input features')

    # Output parameters
    parser.add_argument('--log_root', default='./outputs/logs', type=str,
                        help='root path where to tensorboard log')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--ckpt_freq', default=20, type=int)
    parser.add_argument('--print_freq', default=20, type=int)

    # Training parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Optimizer parameters
    parser.add_argument('--loss_type', type=str, default='mae',
                        help='loss function for optimization')
    parser.add_argument('--obs_weight', type=float, default=1.0,
                        help='weight of observed points in loss computation')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    args = parser.parse_args()

    assert args.model_name in models_vitae.__dict__.keys(), f'model_name:{args.model_name} was not found'
    assert args.input_size in [256, 512], 'input_size should be 256 or 512'
    assert args.input_type in ['sparse', 'voronoi'], 'input_type should be sparse or voronoi'

    return args


def train(args):

    print('-' * 100)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))

    # --------------------------------------------------------------------------
    # settings

    # env
    seed = args.seed
    device_no = args.device
    log_root = args.log_root
    model_root = args.model_root

    # data file
    data_dir = args.data_dir
    data_root = args.data_root

    # input data
    augment = args.augment
    val_prop = args.val_prop
    train_prop = args.train_prop
    model_name = args.model_name
    input_type = args.input_type
    input_size = args.input_size
    patch_size = args.patch_size
    
    if 'vitae' in model_name:
        model_dir_name = f'{model_name}_patch{patch_size}'
    else:
        model_dir_name = model_name

    # hyperparameters
    lr = args.lr
    epochs = args.epochs
    pin_mem = args.pin_mem
    loss_type = args.loss_type
    ckpt_freq = args.ckpt_freq
    obs_weight = args.obs_weight
    batch_size = args.batch_size
    num_workers = args.num_workers
    start_epoch = args.start_epoch
    weight_decay = args.weight_decay

    # --------------------------------------------------------------------------
    # preparation

    # environment setup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(device_no)

    # model dir
    model_dir = os.path.join(model_root, data_dir, model_dir_name, input_type)
    os.makedirs(model_dir, exist_ok=True)

    # log dir
    logs_dir = os.path.join(log_root, data_dir, model_dir_name, input_type)
    os.makedirs(logs_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # load dataset

    # source data
    data_path = os.path.join(data_root, data_dir)
    train_list, val_list, _ = load_pickle_and_split(data_path, train_prop, val_prop)

    # dataset
    dataset_train = SimuDataset(train_list, input_type, augment)
    dataset_val = SimuDataset(val_list, input_type)

    # dataloader
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_mem)
    loader_train = get_dataloader(dataset_train, is_train=True, **loader_kwargs)
    loader_val = get_dataloader(dataset_val, is_train=False, **loader_kwargs)

    # --------------------------------------------------------------------------
    # define the model and  optimizer

    # define the model
    model_func = getattr(models_vitae, model_name)
    in_chans = 1 if input_type == 'sparse' else 2
    if 'vitae' in model_name:
        model = model_func(
            input_size=input_size,
            in_chans=in_chans,
            patch_size=patch_size
        )
    else:  # model_name == 'voronoi_cnn'
        model = model_func(in_chans=in_chans)
    model.to(device)
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # load checkpoint
    misc.load_model(args=args, model_without_ddp=model,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    # --------------------------------------------------------------------------
    # define loss functions

    opt_func = get_loss_func(loss_type, obs_weight)
    eval_func = get_loss_func('l2norm')

    # --------------------------------------------------------------------------
    # information

    print('-' * 100)
    print(f'Start training for {epochs} epochs')
    print(f'Dataset: {data_path}')
    print(f'Model: {model_name}')
    print(f'Input: {input_type}')
    print(f'Training: {len(train_list)} - Validation: {len(val_list)}')
    print(f'Model dir: {model_dir}')
    print(f'Log dir: {logs_dir}\n')

    # --------------------------------------------------------------------------
    # training and validation

    best_val_eval = np.inf
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        # epoch for training
        train_stats = train_one_epoch(
            model, loader_train, optimizer, device, epoch,
            loss_scaler, args, opt_func, eval_func
        )

        # epoch for validation
        val_stats = val_one_epoch(
            model, loader_val, device, epoch,
            args, opt_func, eval_func
        )

        # save checkpoint regularly
        if epoch % ckpt_freq == 0 or epoch + 1 == epochs:
            misc.save_model(
                args=args, output_dir=model_dir, model=model,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
            )

        # save checkpoint with best val loss
        if val_stats['eval'] < best_val_eval:
            best_val_eval = val_stats['eval']
            misc.save_model(
                args=args, output_dir=model_dir,
                model=model, epoch='best'
            )
            print('>>> Best Val Epoch - Lowest Eval Loss - Save Model <<<')

        # write log
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'epoch': epoch
        }
        with open(os.path.join(logs_dir, 'log.txt'), mode='a', encoding='utf-8') as f:
            f.write(json.dumps(log_stats) + '\n')
        print()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('-' * 100, '\n')


if __name__ == '__main__':
    args = get_args_parser()
    train(args)
