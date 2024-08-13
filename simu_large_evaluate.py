import os
import sys
import GPy
import time
import torch
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import libs.models as models_vitae

from tqdm import tqdm
from libs.dataset import load_pickle_and_split
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


warnings.filterwarnings('ignore')


def metrics_stats(metrics):
    return np.mean(metrics), np.std(metrics)


def gpy_interpolate(data, field_size, scale=1, kernel_type='rbf'):
    assert kernel_type in ['rbf', 'exp']

    X = data[:, :2]
    Y = data[:, 2][..., None] / scale

    # define kernel
    if kernel_type == 'rbf':
        ker = GPy.kern.RBF(input_dim=2)
    else:  # kernel_type == 'exp
        ker = GPy.kern.Exponential(input_dim=2)

    # create simple GP model
    m = GPy.models.GPRegression(X, Y, ker)

    # optimize and plot
    m.optimize(messages=False, max_f_eval=1000)

    # generate grid points
    gridx = np.arange(0.0, field_size, 1.0)
    gridy = np.arange(0.0, field_size, 1.0)
    cols, rows = np.meshgrid(gridx, gridy)
    cols, rows = cols.flatten(), rows.flatten()
    points = np.stack([rows, cols]).T

    interp, _ = m.predict(points)
    interp = interp.reshape((field_size,) * 2).T

    return interp * scale


def l2_norm(x):
    return np.sqrt(np.sum(x ** 2))


def prepare_model(model_path, model_name, input_size,
                  patch_size, input_type, device):
    if not os.path.isfile(model_path):
        raise ValueError(f'{model_path} was not found')

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

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    return model.to(device)


def load_sample(data, input_type):

    obs = data['obs']
    gt = np.array(data['gt'])
    voronoi = data['voronoi']

    sparse = np.zeros_like(gt)
    obs_mask = np.zeros_like(gt)
    for x, y, v in obs:
        sparse[int(y), int(x)] = v
        obs_mask[int(y), int(x)] = 1.0

    gt = gt.astype(np.float32)
    obs_mask = obs_mask.astype(np.float32)

    if input_type == 'sparse':
        feature = sparse.astype(np.float32)
    else:  # self.input_type == 'voronoi'
        feature = voronoi.astype(np.float32)

    return gt, feature, obs_mask, obs


def get_args_parser():
    parser = argparse.ArgumentParser('ViT-AE evaluation for simulation dataset', add_help=False)

    # Dataset parameters
    parser.add_argument('--data_root', type=str, help='dataset path')
    parser.add_argument('--data_dir', type=str, help='dataset dir name')
    parser.add_argument('--train_prop', type=float, help='proportion of training data')
    parser.add_argument('--val_prop', type=float, help='proportion of validation data')

    # Model parameters
    parser.add_argument('--model_root', default='./outputs/models', type=str,
                        help='root path where models were saved')
    parser.add_argument('--model_name', default='vitae_base', type=str,
                        metavar='MODEL', help='Name of model to train')
    parser.add_argument('--model_file', default='checkpoint-best.pth', type=str,
                        help='file name of model')
    parser.add_argument('--patch_size', type=int, help='patch size in vit encoder')
    parser.add_argument('--input_size', type=int, help='images input size')
    parser.add_argument('--input_type', type=str, default='sparse', help='input features')
    parser.add_argument('--device', default='cpu', type=str,
                        help='device to use for evaluation')

    # Output parameters
    parser.add_argument('--metric_root', default='./outputs/metrics', type=str,
                        help='root path where metrics will be saved')
    parser.add_argument('--metrics_file', default='metrics.csv', type=str,
                        help='csv file name of metrics')
    parser.add_argument('--save_metrics', action='store_true', help='if save metrics')
    parser.add_argument('--kriging_num', type=int, default=100, help='number of kriging interpolation, -1 for all')
    parser.add_argument('--kriging_kernel', type=str, default='rbf', help='type of kernel for kriging interpolation')

    # visualize
    parser.add_argument('--visualize', action='store_true', help='if visualize outputs')
    parser.add_argument('--visual_root', type=str, default='', help='output dir of visualization')
    parser.add_argument('--visual_num', type=int, default=10, help='number of visualization')
    parser.add_argument('--visual_kriging', action='store_true', help='visual kriging')

    args = parser.parse_args()

    assert args.model_name in models_vitae.__dict__.keys(), f'model_name:{args.model_name} was not found'
    assert args.input_size in [256, 512], 'input_size should be 256 or 512'
    assert args.input_type in ['sparse', 'voronoi'], 'input_type should be sparse or voronoi'
    assert args.kriging_kernel in ['rbf', 'exp'], 'kriging_kernel should be oe of rbf or exp'

    if args.visualize:
        assert args.visual_root, 'args.visual_root should be set if args.visualize is true'

    return args


def evaluate(args):

    # --------------------------------------------------------------------------
    # settings

    # env
    device_no = args.device
    visualize = args.visualize
    visual_num = args.visual_num
    model_root = args.model_root
    metric_root = args.metric_root
    visual_root = args.visual_root
    kriging_num = args.kriging_num
    save_metrics = args.save_metrics
    metrics_file = args.metrics_file
    visualize_kriging = args.visual_kriging

    # data file
    data_root = args.data_root
    data_dir = args.data_dir

    # input data
    val_prop = args.val_prop
    train_prop = args.train_prop
    model_name = args.model_name
    model_file = args.model_file
    input_size = args.input_size
    input_type = args.input_type
    patch_size = args.patch_size
    kriging_kernel = args.kriging_kernel

    if 'vitae' in model_name:
        model_dir_name = f'{model_name}_patch{patch_size}'
    else:
        model_dir_name = model_name

    kriging_dir = os.path.join(f'./kriging_{kriging_kernel}', f'simu{input_size}', data_dir)
    os.makedirs(kriging_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # load test data

    data_path = os.path.join(data_root, data_dir)
    _, _, test_list = load_pickle_and_split(data_path, train_prop, val_prop)
    if visualize:
        test_list = test_list[:visual_num]

    # --------------------------------------------------------------------------
    # load model

    device = torch.device(device_no)
    model_dir = os.path.join(model_root, data_dir)
    model_path = os.path.join(model_dir, model_dir_name, input_type, model_file)
    model = prepare_model(model_path, model_name, input_size,
                          patch_size, input_type, device)

    # --------------------------------------------------------------------------
    # init output csv file

    if save_metrics:
        columns = [
            'model_name',   'input_type',
            'vae_l2p_avg',  'vae_l2p_std',
            'vae_ssim_avg', 'vae_ssim_std',
            'vae_psnr_avg', 'vae_psnr_std',
            'krg_l2p_avg',  'krg_l2p_std',
            'krg_ssim_avg', 'krg_ssim_std',
            'krg_psnr_avg', 'krg_psnr_std',
            'duration_avg', 'duration_std'
        ]
        metric_dir = os.path.join(metric_root, data_dir)
        os.makedirs(metric_dir, exist_ok=True)
        metrics_path = os.path.join(metric_dir, metrics_file)
        if not os.path.isfile(metrics_path):
            metrics_df = pd.DataFrame(data=[], columns=columns)
            metrics_df.to_csv(metrics_path, index=False)

    # --------------------------------------------------------------------------
    # inference and evaluation

    print('-' * 100)
    print(f'Dataset: {data_dir}')
    print(f'Model: {model_dir_name}')
    print(f'Input: {input_type}')

    # dir of visualization
    if visualize:
        visual_total = 0
        visual_dir = os.path.join(visual_root, data_dir, model_dir_name, input_type)
        os.makedirs(visual_dir, exist_ok=True)
        print('Visualize in:', visual_dir)

    vae_l2p_list  = []
    vae_ssim_list = []
    vae_psnr_list = []
    krg_l2p_list  = []
    krg_ssim_list = []
    krg_psnr_list = []
    duration_list = []

    for ith_data, test_data in tqdm(enumerate(test_list), total=len(test_list), ncols=100):

        gt, feature, obs_mask, obs = load_sample(test_data, input_type)
        gt_range = np.max(gt) - np.min(gt)

        begin = time.time()

        if input_type == 'sparse':
            feature_tensor = torch.tensor(feature[None, None, ...]).to(device)
        else:  # input_type == 'voronoi'
            feature_tensor = torch.cat([
                torch.tensor(feature[None, None, ...]),
                torch.tensor(obs_mask[None, None, ...]),
            ], dim=1).to(device)

        with torch.no_grad():
            if 'vitae' in model_name:
                pred, pred_enc = model(feature_tensor)
                pred = pred.data.cpu().numpy()[0][0]
                pred_enc = pred_enc.data.cpu().numpy()[0][0]
            else:
                pred = model(feature_tensor)
                pred = pred.data.cpu().numpy()[0][0]

        duration = time.time() - begin
        duration_list.append(duration)

        vae_diff = gt - pred
        vae_l2p = l2_norm(vae_diff) / l2_norm(gt)
        vae_ssim = structural_similarity(pred, gt, data_range=gt_range)
        vae_psnr = peak_signal_noise_ratio(pred, gt, data_range=gt_range)
        vae_l2p_list.append(vae_l2p)
        vae_ssim_list.append(vae_ssim)
        vae_psnr_list.append(vae_psnr)

        if kriging_num == -1:
            make_kriging = True
        else:
            assert kriging_num > 0
            if len(krg_l2p_list) < kriging_num:
                make_kriging = True
            else:
                make_kriging = False

        if make_kriging:
            kriging_path = os.path.join(kriging_dir, f'{ith_data:06d}.npy')
            if os.path.isfile(kriging_path):
                kriging = np.load(kriging_path)
            else:
                kriging = gpy_interpolate(obs, input_size, kernel_type=kriging_kernel)
                np.save(kriging_path, kriging)

            krg_diff = gt - kriging
            krg_l2p = l2_norm(krg_diff) / l2_norm(gt)
            krg_ssim = structural_similarity(kriging, gt, data_range=gt_range)
            krg_psnr = peak_signal_noise_ratio(kriging, gt, data_range=gt_range)
            krg_l2p_list.append(krg_l2p)
            krg_ssim_list.append(krg_ssim)
            krg_psnr_list.append(krg_psnr)

        if visualize:
            # visualization
            subject_name = f'{visual_total:03d}'
            subject_dir = os.path.join(visual_dir, subject_name)
            os.makedirs(subject_dir, exist_ok=True)
            fig_path = os.path.join(subject_dir, f'image_with_kriging_{kriging_kernel}.jpg')
            data_path = os.path.join(subject_dir, f'data_with_kriging_{kriging_kernel}.pkl')

            if visualize_kriging:

                data_dict = {
                    'gt':         gt,
                    'kriging':    kriging,
                    'model_name': model_name,
                    'model_pred': pred,
                    'observed':   feature,
                    'krg_diff':   krg_diff,
                    'model_diff': vae_diff,
                    'krg_l2p':    krg_l2p,
                    'model_l2p':  vae_l2p,
                    'num_obs':    int(np.sum(obs_mask))
                }
                with open(data_path, 'wb') as f:
                    pickle.dump(data_dict, f)

                plt.figure(figsize=(12, 8))
                imshow_kwargs = dict(vmin=gt.min(), vmax=gt.max(), cmap='coolwarm')
                plt.subplot(231)
                plt.title(f'GT ({input_size}x{input_size})')
                plt.imshow(gt, **imshow_kwargs)
                if make_kriging:
                    plt.subplot(232)
                    plt.title('Kriging')
                    plt.imshow(kriging, **imshow_kwargs)
                plt.subplot(233)
                plt.title(f'{model_name} pred')
                plt.imshow(pred, **imshow_kwargs)
                plt.subplot(234)
                plt.title(f'Observed (num={int(np.sum(obs_mask))})')
                plt.imshow(feature, **imshow_kwargs)
                plt.subplot(235)
                plt.title(f'GT - Kriging | L2P {krg_l2p:.5f}')
                im = plt.imshow(krg_diff, cmap='coolwarm')
                plt.colorbar(im, fraction=0.0275, pad=0.04)
                plt.subplot(236)
                plt.title(f'GT - {model_name} | L2P {vae_l2p:.5f}')
                im = plt.imshow(vae_diff, cmap='coolwarm')
                plt.colorbar(im, fraction=0.0275, pad=0.04)
                plt.tight_layout()
                # plt.show()
                plt.savefig(fig_path)
                plt.close()
            else:
                data_dict = {
                    'gt':         gt,
                    'model_name': model_name,
                    'model_pred': pred,
                    'observed':   feature,
                    'model_diff': vae_diff,
                    'model_l2p':  vae_l2p,
                    'num_obs':    int(np.sum(obs_mask))
                }
                with open(data_path, 'wb') as f:
                    pickle.dump(data_dict, f)

                plt.figure(figsize=(8.85, 8))
                imshow_kwargs = dict(vmin=gt.min(), vmax=gt.max(), cmap='coolwarm')
                plt.subplot(221)
                plt.title(f'GT ({input_size}x{input_size})')
                plt.imshow(gt, **imshow_kwargs)
                plt.subplot(222)
                plt.title(f'{model_name} pred')
                plt.imshow(pred, **imshow_kwargs)
                plt.subplot(223)
                plt.title(f'Observed (num={int(np.sum(obs_mask))})')
                plt.imshow(feature, **imshow_kwargs)
                plt.subplot(224)
                plt.title(f'GT - {model_name} | L2P {vae_l2p:.5f}')
                im = plt.imshow(vae_diff, cmap='coolwarm')
                plt.colorbar(im, fraction=0.0275, pad=0.04)
                plt.tight_layout()
                # plt.show()
                plt.savefig(fig_path)
                plt.close()

            visual_total += 1
    
    if visualize:
        print('-' * 100, '\n')
        sys.exit(1)

    # --------------------------------------------------------------------------
    # compute average metrics

    # plt.figure()
    # plt.title('Air Quality', fontsize=16)
    # plt.hist(krg_l2p_list, bins=50)
    # plt.yscale('log')
    # plt.xlabel('Kriging L2P', fontsize=16)
    # plt.ylabel('number of samples', fontsize=16)
    # plt.show()

    vae_l2p_avg, vae_l2p_std   = metrics_stats(vae_l2p_list)
    vae_ssim_avg, vae_ssim_std = metrics_stats(vae_ssim_list)
    vae_psnr_avg, vae_psnr_std = metrics_stats(vae_psnr_list)

    krg_l2p_avg, krg_l2p_std   = metrics_stats(krg_l2p_list)
    krg_ssim_avg, krg_ssim_std = metrics_stats(krg_ssim_list)
    krg_psnr_avg, krg_psnr_std = metrics_stats(krg_psnr_list)

    duration_avg, duration_std = metrics_stats(duration_list)

    print(f'{model_name} L2P:  {vae_l2p_avg:.6f}+/-{vae_l2p_std:.6f}')
    print(f'{model_name} SSIM: {vae_ssim_avg:.6f}+/-{vae_ssim_std:.6f}')
    print(f'{model_name} PSNR: {vae_psnr_avg:.6f}+/-{vae_psnr_std:.6f}')
    print()
    print(f'KRG L2P:  {krg_l2p_avg:.6f}+/-{krg_l2p_std:.6f}')
    print(f'KRG SSIM: {krg_ssim_avg:.6f}+/-{krg_ssim_std:.6f}')
    print(f'KRG PSNR: {krg_psnr_avg:.6f}+/-{krg_psnr_std:.6f}')
    print()
    print(f'Duration: {duration_avg:.6f}+/-{duration_std:.6f} seconds')
    print('-' * 100, '\n')

    # --------------------------------------------------------------------------
    # append metrics to the output csv file

    if save_metrics:
        metrics_data = [
            model_dir_name, input_type,
            vae_l2p_avg,    vae_l2p_std,
            vae_ssim_avg,   vae_ssim_std,
            vae_psnr_avg,   vae_psnr_std,
            krg_l2p_avg,    krg_l2p_std,
            krg_ssim_avg,   krg_ssim_std,
            krg_psnr_avg,   krg_psnr_std,
            duration_avg,   duration_std
        ]
        metrics_df = pd.DataFrame(data=[metrics_data], columns=columns)
        metrics_df.to_csv(metrics_path, mode='a', index=False, header=False)

    return


if __name__ == '__main__':

    args = get_args_parser()
    evaluate(args)
