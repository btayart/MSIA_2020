#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:37:22 2020

@author: btayart
"""
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np
from torch.utils.data import Subset, Sampler, Dataset

class ContinuousSampler(Sampler):
    """
    Iterator to provide random indices from the supervised train set
    The iterator never stops, do not call alone in a for loop! Use:

        for index, something_else in zip(
                ContinuousSampler(dataset_length, batch_size),
                iterator_that_will_stop
                ):
            do_something(index, something_else)
        
    or use:
        sampler = ContinuousSampler(dataset_length, batch_size)
        while condition:
            sample = next(sampler)
            condition = something(sample)
    """

    def __init__(self, dataset_length, batch_size):
        self.dataset_length = dataset_length
        self.batch_size = batch_size
        self._buffer = []

    def __iter__(self):
        return self

    def _buffer_check(self):
        if not self._buffer:
            self._buffer = list(np.random.choice(
                                self.dataset_length,
                                self.batch_size,
                                replace=False))

    def __next__(self):
        self._buffer_check()
        return self._buffer.pop()


def get_fixmatch_datasets():
    """
    Returns subsets of CIFAR

    Returns
    -------
    cifar_train : torch.utils.data.Dataset
        CIFAR 10 train dataset, (PIL image, label) pairs.
    cifar_test : torch.utils.data.Dataset
        CIFAR 10 test dataset, (PIL image, tensor) pairs.
    subset_labeled : torch.utils.data.Dataset
        Balanced subset of train with 250 examples, (PIL image, label) pairs.
    subset_crossval : torch.utils.data.Dataset
        Balanced subset of train with 2000 examples, (PIL image, label) pairs.
    subset_unlabeled : torch.utils.data.Dataset
        Balanced subset of train with 47750 examples, (PIL image, label) pairs.

    """
    cifar_train  = CIFAR10(
        root='./cifar_data/', train=True, transform=None, download=True)
    cifar_test = CIFAR10(
        root='./cifar_data/', train=False, transform=transforms.ToTensor(),
        download=True)
    
    # Split into balanced datasest
    ind_labeled = np.zeros((250,),dtype=int)
    ind_cross = np.zeros((2000,),dtype=int)
    ind_unlabeled = np.zeros((47750,),dtype=int)
    
    tgt = np.array(cifar_train.targets)
    for ii in range(10):
        indices, = np.where(tgt==ii)
        ind_labeled[ii*25:(ii+1)*25] = indices[:25]
        ind_cross[ii*200:(ii+1)*200] = indices[25:225]
        ind_unlabeled[ii*4775:(ii+1)*4775] = indices[225:]
        
    
    subset_labeled = Subset(cifar_train, ind_labeled)
    subset_crossval = Subset(cifar_train, ind_cross)
    subset_unlabeled = Subset(cifar_train, ind_unlabeled)
    
    return (cifar_train, cifar_test, subset_labeled,
        subset_crossval, subset_unlabeled)


    
# Adapted from:
# https://discuss.pytorch.org/t/apply-different-transform-data-augmentation-to-train-and-validation/63580/2
class MapDataset(Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """
    def __init__(self, dataset, map_func):
        self.dataset = dataset
        self.map_func = map_func
    
    def mapping(self,d):
        return self.map_func(d)
    
    def __getitem__(self, index):
        return self.mapping(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


class LabeledAugment(MapDataset):
    """
    Given a labeled dataset, applies the transform to the image only while
    leaving the label untouched
    """
    def mapping(self, d):
        img, y = d
        return self.map_func(img), y

class UnabeledAugment(MapDataset):
    """
    Given a labeled dataset, drops the label and applies a weak and strong
    transform to the image
    """
    def __init__(self, dataset, map_func_weak, map_func_strong):
        """
        dataset
        map_func_weak, map_func_strong: callables
        """
        super(UnabeledAugment, self).__init__(dataset,map_func_weak)
        self.map_func_strong = map_func_strong

    def __getitem__(self, index):
        img, y = self.dataset[index]
        return self.map_func(img), self.map_func_strong(img)
    
def get_random_rotate_label(dataset):
    """
    Returns a dataset where images are randomly rotated by 90n degrees
    along with the corrseponding labels (n)
    """
    return MapDataset(dataset, random_rotate)

def random_rotate(d):
    """
    Takes an (PIL image, label) tuple sampled from a dataset, rotate it and
    give a new label according to the rotation angle
    """
    img, _ = d[0]
    new_label = np.random.randint(4)
    new_img = img.copy().rotate(90*new_label)
    return new_img, new_label