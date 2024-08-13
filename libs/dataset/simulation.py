import numpy as np
# import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from .utils import simu_augment_transform


class SimuDataset(Dataset):

    def __init__(self, data_list, input_type='sparse', augment=False):
        super(SimuDataset, self).__init__()
        assert input_type in ['sparse', 'voronoi']

        self.augment = augment
        self.data_list = data_list
        self.input_type = input_type

        if augment:
            self.transform = simu_augment_transform()

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):

        data = self.data_list[index]
        obs = data['obs']
        gt = np.array(data['gt'])
        voronoi = data['voronoi']

        sparse = np.zeros_like(gt)
        obs_mask = np.zeros_like(gt)
        for x, y, v in obs:
            sparse[int(y), int(x)] = v
            obs_mask[int(y), int(x)] = 1.0

        if self.input_type == 'sparse':
            feature = sparse
        else:  # self.input_type == 'voronoi'
            feature = voronoi

        if self.augment:
            transformed = self.transform(
                image=gt, image0=feature, image1=obs_mask
            )

            gt = transformed['image']
            feature = transformed['image0']
            obs_mask = transformed['image1']

        # plt.figure(figsize=(20, 10))
        # plt.subplot(221)
        # plt.imshow(gt)
        # plt.subplot(222)
        # plt.imshow(feature)
        # plt.subplot(223)
        # plt.imshow(obs_mask)
        # plt.tight_layout()
        # plt.show()

        gt = gt[None, ...].astype(np.float32)
        feature = feature[None, ...].astype(np.float32)
        obs_mask = obs_mask[None, ...].astype(np.float32)

        if self.input_type == 'voronoi':
            feature = np.concatenate([feature, obs_mask], axis=0)

        return gt, feature, obs_mask