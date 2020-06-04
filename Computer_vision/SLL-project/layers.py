#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:48:56 2020

@author: btayart
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EvoNorm_S0(nn.Module):
    """ EvoNorm-S0 group normalization layer
    
    REFERENCE: Evolving Normalization-Activation Layers
    Hanxiao Liu, Andrew Brock, Karen Simonyan, Quoc V. Le
    arXiv:2004.02967 [cs.LG]
    """
    def __init__(self, num_groups, num_channels, eps=1e-5):
        """
        """
        super(EvoNorm_S0, self).__init__()
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.chn_per_group = self.num_channels//self.num_groups
        self.eps = float(eps)

        if self.num_groups*self.chn_per_group != self.num_channels:
            raise ValueError(
                "Number a channels should be a multiple of the number of groups")

        self.weight = nn.Parameter(torch.full((num_channels,), 2))
        self.bias = nn.Parameter(torch.zeros((num_channels,)))
        self.sigmoid_weight = nn.Parameter(torch.zeros((num_channels,)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, 2.)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.sigmoid_weight)

    def forward(self, input):
        gn_input = F.group_norm(
            input, self.num_groups, None, None, self.eps)
        sigmoid_weight = torch.sigmoid(
            input*self.sigmoid_weight.view(1, -1, 1, 1))

        return gn_input*sigmoid_weight*self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
    
def norm_block(channels, normalization):
    if normalization == "batch_norm":
        norm_fcn =  nn.Sequential(nn.BatchNorm2d(channels),
                                  nn.ReLU())
    elif normalization == "evo":
        norm_fcn = EvoNorm_S0(32, channels)
    else:
        raise ValueError(
            "normalization: expected 'batch_norm' or 'evo', got " + \
                repr(normalization))    
    return norm_fcn

class WideResnetBlock(torch.nn.Module):
    """
    (3,3) block for a Wide Resnet
    The block applies two successive 3x3 convolutions, the output of which is
    added to the input. The fist convolution expands the number of channels
    by a factor k, the second one shrinks it back to the number of channels in 
    the input.
    
    + variants:
        "concatenate" : the convolution output is concatenated rather than
        added, this doubles the number of output channels, half of them
        being equal to the input.
        
        "add+concatenate" : two convolutions are done on the intermediate
        result: one is added to the input and the second concatenated. This
        doubles the number of number of output channels.
    
    """
    ADD = 0
    CONCAT = 1
    ADD_CAT = 2
    def __init__(self,
                 channels,
                 k,
                 dropout_ratio=0.3,
                 merge_mode="add",
                 normalization="batch_norm",
                 groups=1):
        """
        

        Parameters
        ----------
        channels : int
            number of input channels.
        k : int
            widening factor for the convolution. The number of channels
            between the convolutions is multiplied by k.
        dropout_ratio : float, optional
            Proportion of channels dropped-out between the convolutions.
            Must be in  in [0,1] range. The default is 0.3.
        merge_mode : str, optional
            "add" to return addition of conv output to block input
            "concatenate" to return concatenation of conv output to block input
            "add+concatenate" to return addition of conv output to block input
                    concatenated with output of another conv
            . The default is "add".
        normalization : TYPE, optional
            "batch_norm" for BN+ReLU.
            "evo" for EvoNorms S0
            The default is "batch_norm".
        groups : int, optional
            number of groups for the first convolution

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        super(WideResnetBlock, self).__init__()

        if merge_mode == "add":
            self._output_mode = self.ADD
        elif merge_mode == "concatenate":
            self._output_mode = self.CONCAT
        elif merge_mode == "add+concatenate":
            self._output_mode = self.ADD_CAT
        else:
            raise ValueError("merge_mode: expected 'add', 'concatenate' or "+\
                             "'add+concatenate', got " + repr(merge_mode))

        self.norm1 = norm_block(channels,normalization)
        self.conv1 = nn.Conv2d(channels, k*channels,
                               (3, 3), bias=False, padding=1, groups=groups)
        self.drop = nn.Dropout2d(dropout_ratio)       
        self.norm2 = norm_block(k*channels,normalization)
        self.conv2 = nn.Conv2d(k*channels, channels,
                               (3, 3), bias=False, padding=1)

        if self._output_mode == self.ADD_CAT:
            self.conv_cat = nn.Conv2d(k*channels, channels,
                               (3, 3), bias=False, padding=1)
    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.norm2(x)
        
        if self._output_mode  == self.ADD_CAT:
            x_cat = self.conv_cat(x)
            
        x = self.conv2(x)
        if self._output_mode == self.ADD:
            return x + identity
        elif self._output_mode == self.CONCAT:
            return torch.cat((identity, x), dim=1)
        elif self._output_mode == self.ADD_CAT:
            return torch.cat((x+identity, x_cat), dim=1)
        else:
            raise RuntimeError('self._output_mode has been altered!')

class WideResnetExpand(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 k,
                 dropout_ratio=0.3,
                 normalization="batch_norm",
                 stride = 2):
        
        super(WideResnetExpand, self).__init__()
        channels = 2*in_channels
        
        self.norm1 = norm_block(in_channels,normalization)
        self.conv1 = nn.Conv2d(in_channels, k*channels,
                               (3, 3), stride=stride,
                               bias=False, padding=1)
        self.drop = nn.Dropout2d(dropout_ratio)
        self.norm2 = norm_block(k*channels,normalization)
        self.conv2 = nn.Conv2d(k*channels, channels,
                               (3, 3), bias=False, padding=1)

        self.conv_down = nn.Conv2d(in_channels, channels,
                                          (1, 1), stride=stride,
                                          bias=False)

    def forward(self, x):
        x = self.norm1(x)
        res = self.conv_down(x)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.norm2(x)
        x = self.conv2(x)
        
        return res + x
