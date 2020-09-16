#!/usr/bin/env python3

import torch
import numpy as np
import os.path as osp
from torchvision import datasets
from torchvision import transforms
from torch.utils import data

dataset_choices = ['MNIST', 'FashionMNIST', 'SVHN', 'STL10', 'CIFAR10', 'CIFAR100', 'TinyImageNet', 'CelebA', 'LSUN']

def get_dataset(dataset_name, dataset_root, train_transform, test_transform):

    if dataset_name == 'MNIST':
        train_data = datasets.MNIST(osp.join(dataset_root, 'MNIST'), train=True, transform=train_transform, download=False)
        test_data = datasets.MNIST(osp.join(dataset_root, 'MNIST'), train=False, transform=test_transform, download=False)
    elif dataset_name == 'FashionMNIST':
        train_data = datasets.FashionMNIST(osp.join(dataset_root, 'FashionMNIST'), train=True, transform=train_transform, download=False)
        test_data = datasets.FashionMNIST(osp.join(dataset_root, 'FashionMNIST'), train=False, transform=test_transform, download=False)
    elif dataset_name == 'CIFAR10':
        train_data = datasets.CIFAR10(osp.join(dataset_root, 'CIFAR10'), train=True, transform=train_transform, download=False)
        test_data = datasets.CIFAR10(osp.join(dataset_root, 'CIFAR10'), train=False, transform=test_transform, download=False)
    elif dataset_name == 'SVHN':
        train_data = datasets.SVHN(osp.join(dataset_root, 'SVHN'), split='train', transform=train_transform, download=False)
        test_data = datasets.SVHN(osp.join(dataset_root, 'SVHN'), split='test', transform=test_transform, download=False)
    else:
        raise Exception('Unknown dataset: {}'.format(dataset_name))

    return train_data, test_data

def get_mean_std(dataset):

    if dataset == 'MNIST':
        mean = (0.1307,)
        std = (0.3081,)
    elif dataset == 'FashionMNIST':
        mean = (0.5,)
        std  = (0.5,)
    elif dataset == 'CIFAR10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'SVHN':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        raise Exception('Unknown dataset: {}'.format(dataset))

    return mean, std

def get_dataset_transforms(mean, std, grayscale=False, augment=False):

    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = test_transform

    return train_transform, test_transform

def get_dataset_config(dataset_name):

    config = dict()
    if dataset_name in ['MNIST', 'FashionMNIST']:

        config['size']        = 28
        config['ch']          = 1
        config['num_classes'] = 10
        # config['train']       = 6e4
        # config['test']        = 1e4
        config['dataset_size']= {'train': 6e4, 'test': 1e4}

    elif dataset_name in ['CIFAR10', 'CIFAR100']:

        config['size']        = 32
        config['ch']          = 3
        config['num_classes'] = 10
        if dataset_name == 'CIFAR100':
            config['num_classes'] = 100
        # config['train']       = 5e4
        # config['test']        = 1e4
        config['dataset_size']= {'train': 5e4, 'test': 1e4}

    elif dataset_name in ['STL10']:

        config['size']        = 96 
        config['ch']          = 3
        config['num_classes'] = 10
        # config['train']       = 5e4
        # config['test']        = 1e4
        config['dataset_size']= {'train': 5e4, 'test': 8e4}

    elif dataset_name in ['SVHN']:

        config['size']        = 32
        config['ch']          = 3
        config['num_classes'] = 11
        # config['train']       = 73257
        # config['test']        = 26032
        config['dataset_size']= {'train': 73257, 'test': 26032}

    elif dataset_name in ['TinyImageNet']:
        config['size']        = 64
        config['ch']          = 3
        config['num_classes'] = 200
        # config['train']       = 1e5
        # config['test']        = 1e4
        config['dataset_size']= {'train': 1e5, 'test': 1e4}

    elif dataset_name in ['CelebA']:
        config['size']        = 64
        config['ch']          = 3
        config['num_classes'] = None

        config['dataset_size']= {'train': 202599}

    elif dataset_name in ['LSUN']:
        config['size'] = 64
        config['ch']   = 3
        config['num_classes'] = None
        
        config['dataset_size'] = {'train': 168103}

    else:
        raise Exception('unknown dataset: {}'.format(dataset_name))

    return config

def get_directories(run_dir):

    ckpt_dir = osp.join(run_dir, 'ckpt')
    images_dir = osp.join(run_dir, 'images')
    log_dir = osp.join(run_dir, 'log')

    return (ckpt_dir, images_dir, log_dir)

def get_weight_dataset(ckpt_dir, seeds, accs, total_parameter_count, transform):

    n_samples = len(seeds)

    x_index = 0
    y_index = 0
    x = np.zeros((n_samples, total_parameter_count))
    y = np.zeros(n_samples)
    for i, seed in enumerate(seeds):
        # read data
        weights_path = osp.join(ckpt_dir, f'seed_{seed}.tar')
        assert osp.exists(weights_path), f'{weights_path} was not found'
        weights = torch.load(weights_path)

        # fill data
        with torch.no_grad():

            # transform before filling
            x[x_index] = transform(weights)
            x_index += 1
            y[y_index] = accs[i]
            y_index += 1

    assert x_index == x.shape[0]
    assert y_index == y.shape[0]

    return x, y

def get_validation_split(X, Y, index):

    train_x = X[:index] + X[index+1:]
    train_y = Y[:index] + Y[index+1:]

    test_x = X[index]
    test_y = Y[index]

    if isinstance(test_x, np.ndarray):
        return np.concatenate(train_x, axis=0), test_x, np.concatenate(train_y, axis=0), test_y
    elif isinstance(test_x, torch.tensor):
        return torch.cat(train_x, axis=0), test_x, torch.cat(train_y, axis=0), test_y
    else:
        raise Exception('unknown data type: {}'.format(type(x[0]).__class__))