#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 01:03:00 2020

@author: btayart
"""
import torch.nn as nn
import torch
import os
from layers import WideResnetBlock
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from utils import test_net
import numpy as np

def get_model_1():
    WRN_K = 4
    BASE_SIZE = 64
    INPUT_CHANNELS = 3
    N_CLASSES = 10
    CLASSIFIER_WIDTH = 512
    
    wrn_kw = {"k":WRN_K,"groups":4}
    
    model = nn.Sequential(
        nn.BatchNorm2d(INPUT_CHANNELS),
        nn.Conv2d(INPUT_CHANNELS, BASE_SIZE, (3, 3), bias=True, padding=1),
        # 32x32
        WideResnetBlock(BASE_SIZE,**wrn_kw),
    #    WideResnetBlock(BASE_SIZE,WRN_K),
        WideResnetBlock(BASE_SIZE,**wrn_kw,merge_mode="concatenate"),
        nn.MaxPool2d(2, stride=2),
        # 16x16
        WideResnetBlock(2*BASE_SIZE,**wrn_kw),
    #    WideResnetBlock(2*BASE_SIZE,WRN_K),
        WideResnetBlock(2*BASE_SIZE,**wrn_kw,merge_mode="concatenate"),
        nn.MaxPool2d(2, stride=2),
        # 8x8
        WideResnetBlock(4*BASE_SIZE,**wrn_kw),
    #    WideResnetBlock(4*BASE_SIZE,WRN_K),
        WideResnetBlock(4*BASE_SIZE,**wrn_kw,merge_mode="concatenate"),
        # Classifier
        nn.BatchNorm2d(8*BASE_SIZE),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        nn.Flatten(),
        nn.Dropout(.3),
        nn.Linear(8*BASE_SIZE,CLASSIFIER_WIDTH),
        nn.BatchNorm1d(CLASSIFIER_WIDTH),
        nn.ReLU(),
        nn.Linear(CLASSIFIER_WIDTH,N_CLASSES)
        )
    
    weight_file = os.path.join(
        os.path.dirname(__file__),
        "model_weights",
        "model_1.weights.P"
        )
    model.load_state_dict(torch.load(weight_file))
    return model


def get_model_2():
    WRN_K = 4
    BASE_SIZE = 64
    INPUT_CHANNELS = 3
    N_CLASSES = 10
    NORM = "batch_norm"
    CLASSIFIER_WIDTH = 512
    dropout=0.0
    kw = {"k": WRN_K, "normalization": NORM, "dropout_ratio":dropout}
    model = nn.Sequential(
        nn.BatchNorm2d(INPUT_CHANNELS),
        nn.Conv2d(INPUT_CHANNELS, BASE_SIZE, (3, 3), bias=True, padding=1),
        # 32x32
        WideResnetBlock(BASE_SIZE,**kw),
    #    WideResnetBlock(BASE_SIZE,WRN_K),
        WideResnetBlock(BASE_SIZE,**kw,merge_mode="add+concatenate"),
        nn.MaxPool2d(2, stride=2),
        # 16x16
        WideResnetBlock(2*BASE_SIZE,**kw),
    #    WideResnetBlock(2*BASE_SIZE,WRN_K),
        WideResnetBlock(2*BASE_SIZE,**kw,merge_mode="add+concatenate"),
        nn.MaxPool2d(2, stride=2),
        # 8x8
        WideResnetBlock(4*BASE_SIZE,**kw),
    #    WideResnetBlock(4*BASE_SIZE,WRN_K),
        WideResnetBlock(4*BASE_SIZE,**kw,merge_mode="add+concatenate"),
        # Classifier
        nn.BatchNorm2d(8*BASE_SIZE),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        nn.Flatten(),
        nn.Dropout(.3),
        nn.Linear(8*BASE_SIZE,CLASSIFIER_WIDTH),
        nn.BatchNorm1d(CLASSIFIER_WIDTH),
        nn.ReLU(),
        nn.Linear(CLASSIFIER_WIDTH,N_CLASSES)
        )
    
    weight_file = os.path.join(
        os.path.dirname(__file__),
        "model_weights",
        "model_2.weights.P"
        )
    model.load_state_dict(torch.load(weight_file))
    return model

if __name__=="__main__":
    model = get_model_1()
    model.train(False)
    
    cifar_test = CIFAR10(
            root='./cifar_data/',
            train=False,
            transform=ToTensor(),
            download=True)
    loader = DataLoader(
            dataset=cifar_test,
            batch_size= 100,
            drop_last= False,
            num_workers = 2)
    
    acc, loss, confusion_matrix = test_net(model, loader)
    print("confusion matrix:")
    print(np.array_str(confusion_matrix))