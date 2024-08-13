import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset


class AQHighResDataset(Dataset):

    def __init__(self, data_list, obs_mask, patch_size,
                 input_type='sparse', mean_std=None):
        super(AQHighResDataset, self).__init__()
        assert input_type in ['sparse', 'voronoi']

        self.obs_mask = obs_mask
        self.data_list = data_list
        self.input_type = input_type
        self.patch_size = patch_size

        if mean_std is not None:
            self.mean = mean_std['mean']
            self.std = mean_std['std']
        else:
            self.mean, self.std = 0.0, 1.0

        # calculate padding parameters
        h, w = self.obs_mask.shape
        hpad = self.get_padding_params(h, patch_size)
        wpad = self.get_padding_params(w, patch_size)
        if (hpad is not None) or (wpad is not None):
            self.pad = (hpad, wpad)
            self.obs_mask = np.pad(self.obs_mask, self.pad)
        else:
            self.pad = None
        
        self.input_size = self.obs_mask.shape

    @staticmethod
    def get_padding_params(n, patch_size):
        if n % patch_size != 0:
            pad = (n // patch_size + 1) * patch_size - n
            pad1 = pad // 2
            pad2 = pad - pad1
            pad = (pad1, pad2)
            return pad
        else:
            return None

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):

        data = self.data_list[index]
        gt = np.array(data['gt'])
        data_mask = 1.0 - np.isnan(gt)
        gt = (gt - self.mean) / self.std
        gt = np.nan_to_num(gt)

        obs_mask = self.obs_mask.copy()
        voronoi = (data['voronoi'][..., 0] - self.mean) / self.std

        if self.pad is not None:
            gt = np.pad(gt, self.pad)
            voronoi = np.pad(voronoi, self.pad)
            data_mask = np.pad(data_mask, self.pad)

        sparse = self.obs_mask * gt
        if self.input_type == 'sparse':
            feature = sparse
        else:  # self.input_type == 'voronoi'
            feature = voronoi
        
        # plt.figure(figsize=(12, 8))
        # plt.subplot(221)
        # plt.title(gt.shape)
        # plt.imshow(gt, cmap='coolwarm')
        # plt.subplot(222)
        # plt.imshow(feature, cmap='coolwarm')
        # plt.subplot(223)
        # plt.imshow(obs_mask)
        # plt.subplot(224)
        # plt.imshow(data_mask)
        # plt.tight_layout()
        # plt.show()

        gt = gt[None, ...].astype(np.float32)
        feature = feature[None, ...].astype(np.float32)
        obs_mask = obs_mask[None, ...].astype(np.float32)
        data_mask = data_mask[None, ...].astype(np.float32)

        if self.input_type == 'voronoi':
            feature = np.concatenate([feature, obs_mask], axis=0)

        return gt, feature, obs_mask, data_mask
