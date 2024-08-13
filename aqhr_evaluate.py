import os
import sys
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
from libs.dataset import *
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


warnings.filterwarnings('ignore')


def metrics_stats(metrics):
    return np.mean(metrics), np.std(metrics)


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


def prep_obs_mask(obs_mask, patch_size):

    def get_padding_params(n, patch_size):
        if n % patch_size != 0:
            pad = (n // patch_size + 1) * patch_size - n
            pad1 = pad // 2
            pad2 = pad - pad1
            pad = (pad1, pad2)
            return pad
        else:
            return None

    # calculate padding parameters
    h, w = obs_mask.shape
    hpad = get_padding_params(h, patch_size)
    wpad = get_padding_params(w, patch_size)
    if (hpad is not None) or (wpad is not None):
        pad_params = (hpad, wpad)
        new_obs_mask = np.pad(obs_mask, pad_params)
    else:
        pad_params = None
        new_obs_mask = obs_mask.copy()
    
    input_size = new_obs_mask.shape
    new_obs_mask = new_obs_mask.astype(np.float32)
    return new_obs_mask, input_size, pad_params


def load_sample(data, input_type, obs_mask, pad_params, mean_std):

    if mean_std is not None:
        mean = mean_std['mean']
        std = mean_std['std']
    else:
        mean, std = 0.0, 1.0

    gt = np.array(data['gt'])
    gt_tmp = (gt - mean) / std
    gt_tmp = np.nan_to_num(gt_tmp)

    data_mask = 1.0 - np.isnan(gt)
    voronoi = (data['voronoi'][..., 0] - mean) / std
    kriging = np.array(data['kriging'])

    if pad_params is not None:
        gt_tmp = np.pad(gt_tmp, pad_params)
        voronoi = np.pad(voronoi, pad_params)

    sparse = obs_mask * gt_tmp

    gt = gt.astype(np.float32)
    kriging = kriging.astype(np.float32)
    if input_type == 'sparse':
        feature = sparse.astype(np.float32)
    else:  # self.input_type == 'voronoi'
        feature = voronoi.astype(np.float32)

    return gt, feature, kriging, data_mask


def get_args_parser():
    parser = argparse.ArgumentParser('ViT-AE evaluation for air quality high resolution dataset', add_help=False)

    # Dataset parameters
    parser.add_argument('--data_root', type=str, help='dataset path')
    parser.add_argument('--data_name', type=str, help='dataset name')
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
    parser.add_argument('--input_type', type=str, default='sparse', help='input features')
    parser.add_argument('--normalize', action='store_true', help='if normalize inputs')
    parser.add_argument('--device', default='cpu', type=str,
                        help='device to use for evaluation')

    # Output parameters
    parser.add_argument('--metric_root', default='./outputs/metrics', type=str,
                        help='root path where metrics will be saved')
    parser.add_argument('--metrics_file', default='metrics.csv', type=str,
                        help='csv file name of metrics')
    parser.add_argument('--save_metrics', action='store_true', help='if save metrics')

    # visualize
    parser.add_argument('--visualize', action='store_true', help='if visualize outputs')
    parser.add_argument('--visual_root', type=str, default='', help='output dir of visualization')
    parser.add_argument('--visual_num', type=int, default=10, help='number of visualization')

    args = parser.parse_args()

    valid_data_names = [d for d in os.listdir(args.data_root) if d != 'source']
    assert args.data_name in valid_data_names, f'{args.data_name} was not found in {args.data_root}'
    assert args.model_name in models_vitae.__dict__.keys(), f'model_name:{args.model_name} was not found'
    assert args.input_type in ['sparse', 'voronoi'], 'input_type should be sparse or voronoi'

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
    save_metrics = args.save_metrics
    metrics_file = args.metrics_file

    # data file
    data_root = args.data_root
    data_name = args.data_name
    val_prop = args.val_prop
    train_prop = args.train_prop

    # input data
    normalize = args.normalize
    model_name = args.model_name
    model_file = args.model_file
    input_type = args.input_type
    patch_size = args.patch_size

    if 'vitae' in model_name:
        model_dir_name = f'{model_name}_patch{patch_size}'
    else:
        model_dir_name = model_name

    # --------------------------------------------------------------------------
    # load test data

    data_dir = os.path.join(data_root, data_name)
    data_path = os.path.join(data_dir, 'data.pkl')
    _, _, test_list = load_pickle_and_split(data_path, train_prop, val_prop)
    mean_std = load_pickle(os.path.join(data_dir, 'data_mean_std.pkl')) if normalize else None
    obs_mask = np.load(os.path.join(data_dir, 'data_mask.npy'))
    obs_mask, input_size, pad_params = prep_obs_mask(obs_mask, patch_size)

    if visualize:
        visual_index = [i * (len(test_list) // visual_num) for i in range(visual_num)]
        test_list = [test_list[i] for i in visual_index]

    # --------------------------------------------------------------------------
    # load model

    device = torch.device(device_no)
    model_dir = os.path.join(model_root, data_name)
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
        metric_dir = os.path.join(metric_root, data_name)
        os.makedirs(metric_dir, exist_ok=True)
        metrics_path = os.path.join(metric_dir, metrics_file)
        if not os.path.isfile(metrics_path):
            metrics_df = pd.DataFrame(data=[], columns=columns)
            metrics_df.to_csv(metrics_path, index=False)

    # --------------------------------------------------------------------------
    # inference and evaluation

    print('-' * 100)
    print(f'Dataset: {data_name}')
    print(f'Model: {model_dir_name}')
    print(f'Input: {input_type}')
    print(f'Normalization using: {mean_std}\n')

    # dir of visualization
    if visualize:
        visual_total = 0
        visual_dir = os.path.join(visual_root, data_name, model_dir_name, input_type)
        os.makedirs(visual_dir, exist_ok=True)
        print('Visualize in:', visual_dir)

    vae_l2p_list  = []
    vae_ssim_list = []
    vae_psnr_list = []
    krg_l2p_list  = []
    krg_ssim_list = []
    krg_psnr_list = []
    duration_list = []

    for test_data in tqdm(test_list, ncols=100):

        gt, feature, kriging, data_mask = \
            load_sample(test_data, input_type, obs_mask, pad_params, mean_std)

        # fill nan
        gt_tmp = np.nan_to_num(gt, nan=0)
        gt_tmp_range = np.nanmax(gt_tmp) - np.nanmin(gt_tmp)

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

        if pad_params is not None:
            hpad, wpad = pad_params
            pred = pred[hpad[0]:-hpad[1], wpad[0]:-wpad[1]]
            feature = feature[hpad[0]:-hpad[1], wpad[0]:-wpad[1]]

        if normalize:
            pred = pred * mean_std['std'] + mean_std['mean']
            feature = feature * mean_std['std'] + mean_std['mean']

        duration = time.time() - begin
        duration_list.append(duration)

        vae_diff2d = gt - pred
        vae_diff = gt[data_mask == 1] - pred[data_mask == 1]
        vae_l2p = l2_norm(vae_diff) / l2_norm(gt[data_mask == 1])
        vae_l2p_list.append(vae_l2p)

        pred_tmp = pred.copy()
        pred_tmp[np.isnan(gt)] = 0
        vae_ssim = structural_similarity(gt_tmp, pred_tmp, data_range=gt_tmp_range)
        vae_psnr = peak_signal_noise_ratio(gt_tmp, pred_tmp, data_range=gt_tmp_range)
        vae_ssim_list.append(vae_ssim)
        vae_psnr_list.append(vae_psnr)

        krg_diff2d = gt - kriging
        krg_diff = gt[data_mask == 1] - kriging[data_mask == 1]
        krg_l2p = l2_norm(krg_diff) / l2_norm(gt[data_mask == 1])
        krg_l2p_list.append(krg_l2p)

        kriging_tmp = kriging.copy()
        kriging_tmp[np.isnan(gt)] = 0
        krg_ssim = structural_similarity(gt_tmp, kriging_tmp, data_range=gt_tmp_range)
        krg_psnr = peak_signal_noise_ratio(gt_tmp, kriging_tmp, data_range=gt_tmp_range)
        krg_ssim_list.append(krg_ssim)
        krg_psnr_list.append(krg_psnr)

        if visualize:
            # visualization
            subject_name = f'{visual_total:03d}'
            subject_dir = os.path.join(visual_dir, subject_name)
            os.makedirs(subject_dir, exist_ok=True)
            fig_path = os.path.join(subject_dir, 'image.jpg')
            vis_data_path = os.path.join(subject_dir, 'data.pkl')

            data_dict = {
                'gt':         gt,
                'kriging':    kriging,
                'model_name': model_dir_name,
                'model_pred': pred,
                'observed':   feature,
                'krg_diff':   krg_diff2d,
                'model_diff': vae_diff2d,
                'krg_l2p':    krg_l2p,
                'model_l2p':  vae_l2p,
                'num_obs':    int(np.sum(obs_mask))
            }
            with open(vis_data_path, 'wb') as f:
                pickle.dump(data_dict, f)

            plt.figure(figsize=(15, 6))
            imshow_kwargs = dict(vmin=np.nanmin(gt), vmax=np.nanmax(gt), cmap='coolwarm')
            plt.subplot(231)
            plt.title(f'GT ({input_size})')
            plt.imshow(gt, **imshow_kwargs)
            plt.axis('off')
            plt.subplot(232)
            plt.title('Kriging')
            plt.imshow(kriging, **imshow_kwargs)
            plt.axis('off')
            plt.subplot(233)
            plt.title(f'{model_dir_name} pred')
            pred_temp = pred.copy()
            pred_temp[np.isnan(gt)] = np.nan
            plt.imshow(pred_temp, **imshow_kwargs)
            plt.axis('off')
            plt.subplot(234)
            plt.title(f'Observed (num={int(np.sum(obs_mask))})')
            feature_tmp = feature.copy()
            feature_tmp[np.isnan(gt)] = np.nan
            plt.imshow(feature_tmp, **imshow_kwargs)
            plt.axis('off')
            plt.subplot(235)
            plt.title(f'GT - Kriging | L2P {krg_l2p:.5f}')
            im = plt.imshow(krg_diff2d, cmap='coolwarm')
            plt.colorbar(im, fraction=0.0275, pad=0.04)
            plt.axis('off')
            plt.subplot(236)
            plt.title(f'GT - {model_dir_name} | L2P {vae_l2p:.5f}')
            im = plt.imshow(vae_diff2d, cmap='coolwarm')
            plt.colorbar(im, fraction=0.0275, pad=0.04)
            plt.axis('off')
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
    # plt.hist(krg_l2p_list, bins=50, alpha=0.75, label='Kriging')
    # plt.hist(vae_l2p_list, bins=50, alpha=0.75, label=f'{model_dir_name} Pred')
    # plt.yscale('log')
    # plt.xlabel('L2P', fontsize=16)
    # plt.ylabel('number of samples', fontsize=16)
    # plt.legend()
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
