__all__ = ['load_pickle_and_split', 'load_pickle',
           'get_dataloader', 'simu_augment_transform',
           'real_augment_transform']

import os
import pickle
import albumentations as A

from torch.utils.data import RandomSampler, DataLoader


def load_pickle_and_split(data_path, train_prop, val_prop):
    assert os.path.isfile(data_path), f'{data_path} was not found'
    data_list = pickle.load(open(data_path, 'rb'))
    num_data = len(data_list)
    num_train = int(num_data * train_prop)
    num_val = int(num_data * val_prop)

    train_list = data_list[:num_train]
    val_list = data_list[num_train:num_train + num_val]
    test_list = data_list[num_train + num_val:]
    return train_list, val_list, test_list


def load_pickle(data_path):
    assert os.path.isfile(data_path), f'{data_path} was not found'
    return pickle.load(open(data_path, 'rb'))


def get_dataloader(dataset, is_train=False, **kwargs):
    sampler = RandomSampler(dataset) if is_train else None
    dataloader = DataLoader(dataset, sampler=sampler,
                            drop_last=is_train, **kwargs)
    return dataloader


def simu_augment_transform():
    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5)
        ],
        additional_targets={
            'image0': 'image',
            'image1': 'image'
        }
    )


def real_augment_transform():
    return A.Compose(
        [A.Flip(p=0.5)],
        additional_targets={
            'image0': 'image',
            'image1': 'image',
            'image2': 'image'
        }
    )
