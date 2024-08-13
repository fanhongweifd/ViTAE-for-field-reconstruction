import os
import time
import torch
import pickle
import warnings
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import libs.models as models_vitae

from tqdm import tqdm
from libs.dataset import load_pickle
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


warnings.filterwarnings('ignore')


INPUT_SIZE = {
    'noaa': (180, 360),
    'cylinder': (112, 192),
    'ch2dxysec': (48, 128)
}

TEST_FILES = {
    'noaa': ['test_trained.pkl', 'test_unseen.pkl'],
    'cylinder': ['test_obs8.pkl', 'test_obs16.pkl'],
    'ch2dxysec': ['test_trained.pkl', 'test_unseen.pkl']
}

NOAA_NUM_OBS_LIST = {
    'trained': [10, 20, 30, 50, 100],
    'unseen': [10, 20, 30, 50, 70, 100, 200]
}

CH2D_NUM_OBS_LIST = {
    'trained': [50, 100, 200],
    'unseen': [50, 100, 150, 200, 250]
}


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


def load_sample(data, input_type, mean_std):

    if mean_std is not None:
        mean = mean_std['mean']
        std = mean_std['std']
    else:
        mean, std = 0.0, 1.0

    gt = np.array(data['gt'])
    num_obs = data['obs_value']
    obs = data['obs']
    data_mask = 1.0 - np.isnan(gt)
    gt = np.nan_to_num(gt)
    voronoi = (data['voronoi'] - mean) / std

    sparse = np.zeros_like(gt)
    obs_mask = np.zeros_like(gt)
    for r, c, v in obs:
        sparse[int(r), int(c)] = (v - mean) / std
        obs_mask[int(r), int(c)] = 1.0

    if input_type == 'sparse':
        feature = sparse
    else:  # self.input_type == 'voronoi'
        feature = voronoi

    gt = gt.astype(np.float32)
    obs_mask = obs_mask.astype(np.float32)
    data_mask = data_mask.astype(np.float32)

    if input_type == 'sparse':
        feature = sparse.astype(np.float32)
    else:  # self.input_type == 'voronoi'
        feature = voronoi.astype(np.float32)

    return num_obs, gt, feature, obs_mask, data_mask


def get_args_parser():
    parser = argparse.ArgumentParser('ViT-AE evaluation for Voronoi-CNN dataset', add_help=False)

    # Dataset parameters
    parser.add_argument('--data_root', type=str, help='dataset path')
    parser.add_argument('--data_name', type=str, help='dataset name')

    # Model parameters
    parser.add_argument('--model_root', default='./outputs/models', type=str,
                        help='root path where to save, empty for no saving')
    parser.add_argument('--model_name', default='vitae_base', type=str,
                        metavar='MODEL', help='Name of model to train')
    parser.add_argument('--model_file', default='checkpoint-best.pth', type=str,
                        help='file name of model')
    parser.add_argument('--patch_size', type=int, help='patch size in vit encoder')
    parser.add_argument('--input_type', type=str, default='s', help='input features')
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

    assert args.data_name in ['noaa', 'cylinder', 'ch2dxysec'], \
        'data_name must be one of noaa, cylinder or ch2dxysec'
    assert args.model_name in models_vitae.__dict__.keys(), f'model_name:{args.model_name} was not found'
    assert args.input_type in ['sparse', 'voronoi'], 'input_type should be sparse or voronoi'

    if args.model_name != 'voronoi_cnn':
        if args.data_name == 'noaa':
            assert args.patch_size == 9, f'{args.data_name} only supports patch_size 9'
        else:  # args.data_name in ['cylinder', 'ch2dxysec']
            assert args.patch_size in [4, 8], f'{args.data_name} only supports patch_size 4 or 8'

    if args.visualize:
        assert args.visual_root, 'args.visual_root should be set if args.visualize is true'

    return args


def evaluate(args):

    # --------------------------------------------------------------------------
    # settings

    # env
    device_no = args.device
    visualize = args.visualize
    model_root = args.model_root
    visual_num = args.visual_num
    visual_root = args.visual_root
    metric_root = args.metric_root
    save_metrics = args.save_metrics
    metrics_file = args.metrics_file

    # data file
    data_root = args.data_root
    data_name = args.data_name

    # input data
    normalize= args.normalize
    model_name = args.model_name
    model_file = args.model_file
    patch_size = args.patch_size
    input_type = args.input_type
    input_size = INPUT_SIZE[data_name]

    if 'vitae' in model_name:
        model_dir_name = f'{model_name}_patch{patch_size}'
    else:
        model_dir_name = model_name

    # --------------------------------------------------------------------------
    # test data information

    data_path = os.path.join(data_root, data_name)

    if data_name == 'noaa':
        data_num_obs_list = NOAA_NUM_OBS_LIST
    elif data_name == 'ch2dxysec':
        data_num_obs_list = CH2D_NUM_OBS_LIST
    else:  # data_name == 'cylinder'
        data_num_obs_list = None

    mean_std = load_pickle(os.path.join(data_path, 'mean_std.pkl')) if normalize else None

    # --------------------------------------------------------------------------
    # load model

    device = torch.device(device_no)
    model_dir = os.path.join(model_root, data_name)
    model_path = os.path.join(model_dir, model_dir_name, input_type, model_file)
    model = prepare_model(model_path, model_name, input_size,
                          patch_size, input_type, device)

    for test_file in TEST_FILES[data_name]:

        # ----------------------------------------------------------------------
        # load test data

        test_name = test_file[:-4]
        test_path = os.path.join(data_path, test_file)
        test_list = load_pickle(test_path)

        # ----------------------------------------------------------------------
        # output csv file

        if 'trained' in test_name:
            num_obs_list = data_num_obs_list['trained']
            more_columns = [[f'l2p_obs{nobs}_avg', f'l2p_obs{nobs}_std']
                            for nobs in num_obs_list]
            more_columns += [[f'ssim_obs{nobs}_avg', f'ssim_obs{nobs}_std']
                             for nobs in num_obs_list]
            more_columns += [[f'psnr_obs{nobs}_avg', f'psnr_obs{nobs}_std']
                             for nobs in num_obs_list]
            more_columns = list(itertools.chain(*more_columns))
        elif 'unseen' in test_name:
            num_obs_list = data_num_obs_list['unseen']
            more_columns = [[f'l2p_obs{nobs}_avg', f'l2p_obs{nobs}_std']
                            for nobs in num_obs_list]
            more_columns += [[f'ssim_obs{nobs}_avg', f'ssim_obs{nobs}_std']
                             for nobs in num_obs_list]
            more_columns += [[f'psnr_obs{nobs}_avg', f'psnr_obs{nobs}_std']
                             for nobs in num_obs_list]
            more_columns = list(itertools.chain(*more_columns))
        elif 'obs8' in test_name:
            num_obs_list = [8]
            more_columns = ['l2p_obs8_avg', 'l2p_obs8_std',
                            'ssim_obs8_avg', 'ssim_obs8_std',
                            'psnr_obs8_avg', 'psnr_obs8_std']
        elif 'obs16' in test_name:
            num_obs_list = [16]
            more_columns = ['l2p_obs16_avg', 'l2p_obs16_std',
                            'ssim_obs16_avg', 'ssim_obs16_std',
                            'psnr_obs16_avg', 'psnr_obs16_std']
        else:
            raise ValueError(f'{test_name} was not supported')

        if save_metrics:
            metric_dir = os.path.join(metric_root, data_name)
            os.makedirs(metric_dir, exist_ok=True)
            metrics_name = '_'.join([test_name, metrics_file])
            metrics_path = os.path.join(metric_dir, metrics_name)
            columns = ['model_name', 'input_type', 'duration_avg', 'duration_std']
            columns = columns + more_columns
            if not os.path.isfile(metrics_path):
                metrics_df = pd.DataFrame(data=[], columns=columns)
                metrics_df.to_csv(metrics_path, index=False)

        # --------------------------------------------------------------------------
        # inference and evaluation

        print('-' * 100)
        print(f'Dataset: {data_name}')
        print(f'Test: {test_name}[{len(test_list)}]')
        print(f'Model: {model_dir_name}')
        print(f'Input: {input_type}')
        print(f'Normalization using: {mean_std}\n')

        # dir of visualization
        if visualize:
            visual_total = 0
            visual_obs_total = {n: 0 for n in num_obs_list}
            visual_dir = os.path.join(visual_root, data_name, input_type, model_name, test_name)
            os.makedirs(visual_dir, exist_ok=True)
            print('Visualize in:', visual_dir)

        l2p_dict = {}
        ssim_dict = {}
        psnr_dict = {}
        duration_list = []

        for test_data in tqdm(test_list, ncols=100):
    
            num_obs, gt, feature, obs_mask, data_mask = load_sample(test_data, input_type, mean_std)
            assert num_obs in num_obs_list

            # fill nan
            gt_tmp = np.nan_to_num(gt, nan=0)
            gt_tmp_range = np.nanmax(gt_tmp) - np.nanmin(gt_tmp)

            begin = time.time()

            if visualize:
                if all([visual_obs_total[n] == visual_num for n in num_obs_list]):
                    break
                if visual_obs_total[num_obs] == visual_num:
                    continue

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

            if mean_std is not None:
                pred = pred * mean_std['std'] + mean_std['mean']

            duration = time.time() - begin
            duration_list.append(duration)

            gt[data_mask == 0] = np.nan
            pred[data_mask == 0] = np.nan

            vae_diff2d = gt - pred
            vae_diff = gt[data_mask == 1] - pred[data_mask == 1]
            vae_l2p = l2_norm(vae_diff) / l2_norm(gt[data_mask == 1])

            pred_tmp = pred.copy()
            pred_tmp[np.isnan(gt)] = 0
            vae_ssim = structural_similarity(gt_tmp, pred_tmp, data_range=gt_tmp_range)
            vae_psnr = peak_signal_noise_ratio(gt_tmp, pred_tmp, data_range=gt_tmp_range)

            if num_obs not in l2p_dict.keys():
                l2p_dict[num_obs]  = [vae_l2p]
                ssim_dict[num_obs] = [vae_ssim]
                psnr_dict[num_obs] = [vae_psnr]
            else:
                l2p_dict[num_obs].append(vae_l2p)
                ssim_dict[num_obs].append(vae_ssim)
                psnr_dict[num_obs].append(vae_psnr)

            if visualize:
                # visualization
                subject_name = f'{visual_total:03d}'
                subject_dir = os.path.join(visual_dir, subject_name)
                os.makedirs(subject_dir, exist_ok=True)
                fig_path = os.path.join(subject_dir, 'image.jpg')
                vis_data_path = os.path.join(subject_dir, 'data.pkl')

                data_dict = {
                    'gt':         gt,
                    'model_name': model_name,
                    'model_pred': pred,
                    'observed':   feature,
                    'num_obs':    num_obs,
                    'model_diff': vae_diff2d,
                    'model_l2p':  vae_l2p,
                }
                with open(vis_data_path, 'wb') as f:
                    pickle.dump(data_dict, f)

                plt.figure(figsize=(12, 8))
                imshow_kwargs = dict(cmap='seismic', vmin=np.nanmin(gt), vmax=np.nanmax(gt))
                plt.subplot(221)
                plt.title(f'GT')
                plt.imshow(gt, **imshow_kwargs)
                plt.subplot(222)
                plt.title(f'Observed (num={num_obs})')
                plt.imshow(feature, **imshow_kwargs)
                plt.subplot(223)
                plt.title(f'{model_name} pred')
                plt.imshow(pred, **imshow_kwargs)
                plt.subplot(224)
                plt.title(f'GT - {model_name} | L2P {vae_l2p:.5f}')
                im = plt.imshow(vae_diff2d, cmap='seismic')
                plt.colorbar(im, fraction=0.0275, pad=0.04)
                plt.tight_layout()
                # plt.show()
                plt.savefig(fig_path)
                plt.close()

                visual_total += 1
                visual_obs_total[num_obs] += 1

        if visualize:
            continue

        l2p_metrics = []
        ssim_metrics = []
        psnr_metrics = []
        for num_obs in num_obs_list:
            l2p_avg, l2p_std = metrics_stats(l2p_dict[num_obs])
            l2p_metrics += [l2p_avg, l2p_std]
            print(f'OBS{num_obs} L2P:  {l2p_avg:.8f}+/-{l2p_std:.8f}')

            ssim_avg, ssim_std = metrics_stats(ssim_dict[num_obs])
            ssim_metrics += [ssim_avg, ssim_std]
            print(f'OBS{num_obs} SSIM: {ssim_avg:.8f}+/-{ssim_std:.8f}')

            psnr_avg, psnr_std = metrics_stats(psnr_dict[num_obs])
            psnr_metrics += [psnr_avg, psnr_std]
            print(f'OBS{num_obs} PSNR: {psnr_avg:.8f}+/-{psnr_std:.8f}')

        duration_avg, duration_std = metrics_stats(duration_list)
        print(f'Duration: {duration_avg:.6f}+/-{duration_std:.6f} seconds')
        print('-' * 100, '\n')

        if save_metrics:
            metrics_data = [model_dir_name, input_type, duration_avg, duration_std]
            metrics_data = metrics_data + l2p_metrics + ssim_metrics + psnr_metrics
            metrics_df = pd.DataFrame(data=[metrics_data], columns=columns)
            metrics_df.to_csv(metrics_path, mode='a', index=False, header=False)

    return


if __name__ == '__main__':

    args = get_args_parser()
    evaluate(args)
