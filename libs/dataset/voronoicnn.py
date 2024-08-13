import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from .utils import real_augment_transform


class VCNNDataset(Dataset):

    def __init__(self, data_list, input_type='sparse',
                 mean_std=None, augment=False):
        super(VCNNDataset, self).__init__()
        assert input_type in ['sparse', 'voronoi']

        self.augment = augment
        self.data_list = data_list
        self.input_type = input_type

        if mean_std is not None:
            self.mean = mean_std['mean']
            self.std = mean_std['std']
        else:
            self.mean, self.std = 0.0, 1.0
        
        if augment:
            self.transform = real_augment_transform()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        data = self.data_list[index]
        gt = np.array(data['gt'])
        gt = (gt - self.mean) / self.std
        obs = data['obs']
        data_mask = 1.0 - np.isnan(gt)
        gt = np.nan_to_num(gt)
        voronoi = (data['voronoi'] - self.mean) / self.std

        sparse = np.zeros_like(gt)
        obs_mask = np.zeros_like(gt)
        for r, c, v in obs:
            sparse[int(r), int(c)] = (v - self.mean) / self.std
            obs_mask[int(r), int(c)] = 1.0

        if self.input_type == 'sparse':
            feature = sparse
        else:  # self.input_type == 'voronoi'
            feature = voronoi

        if self.augment:
            transformed = self.transform(
                image=gt, image0=feature,
                image1=obs_mask, image2=data_mask
            )

            gt = transformed['image']
            feature = transformed['image0']
            obs_mask = transformed['image1']
            data_mask = transformed['image2']

        # plt.figure(figsize=(12, 8))
        # plt.subplot(221)
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
