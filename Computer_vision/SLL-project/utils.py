#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:39:38 2020

@author: btayart
"""
import torch
import numpy as np
from torch.nn.functional import cross_entropy

class TrainStats():
    """Utility class to log some training statistics"""
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.reset()
        
    def reset(self):
        self.n_batches = 0
        self.n_valid= 0
        self.loss = 0.0
        
    def log_stats(self,loss, valid=None):
        """Log batch statistics:
        loss: value of loss function (expected to be aggregated with a sum)
        valid: number of samples from which the loss was built
        """
        n = self.batch_size if (valid is None) else valid
        self.n_batches += 1
        self.n_valid += int(n)
        self.loss += loss.item()*n if (valid is None) else loss.item()
        
    def avg_valid(self):
        """Returns the proportion of valid samples"""
        if self.n_batches:
            return float(self.n_valid) / (self.n_batches*self.batch_size)
        else:
            return 0.
        
    def avg_loss(self, mode="valid"):
        """
        Returns average loss mong valid examples
        use mode = "all" if the loss function was aggregated with a mean
        """
        if mode=="valid":
            if self.n_valid:
                return self.loss / self.n_valid
        elif mode=="all":
            if self.n_batches:
                return self.loss / (self.n_batches * self.batch_size)
        else:
            raise ValueError("mode: expected 'valid' or 'all'")
        return 0.
        
def test_net(model, dataloader):
    """
    Check the performance of a model on a validation dataset

    Parameters
    ----------
    model : torch.nn.Module
        Model. Calculation is performed on the same device as the model
    dataloader : torch.utils.data.Dataloader
        Dataloader for the validation set

    Returns
    -------
    avg_loss : float
        Average of the loss function across the validation set
    accuracy : float
        Proportion of accurately predicted labels across the validation set
        (top-1 error)
    confusion_matrix : numpy.ndarray
        Confustion matrix, with predicted labels as rows and ground truth
        labels as columns
        confusion_matrix[ii][jj] contains the number of images of the validation
        set classified as *ii* while their actual label is *jj*

    """
    model.train(False)
    device = next(model.parameters()).device
    n_in_dataset = len(dataloader.dataset)
    cumul_loss = 0.0
    class_ok = 0
    
    all_gt = torch.zeros(n_in_dataset, dtype=torch.long)
    all_pred = torch.zeros(n_in_dataset, dtype=torch.long)
    with torch.no_grad():
        for batch_idx, (x, gt) in enumerate(dataloader):
            siz, = gt.size()
            i0 = batch_idx * dataloader.batch_size
            all_gt[i0:i0+siz] = gt
            
            x, gt = x.to(device), gt.to(device)
            output = model(x)
            cumul_loss += cross_entropy(output, gt, reduction="sum").item()
            predicted_class = output.argmax(dim=1)
            class_ok += (predicted_class == gt).sum()
            
            predicted_class = predicted_class.cpu()
            all_pred[i0:i0+siz] = predicted_class

    avg_loss = cumul_loss /len(dataloader.dataset)
    accuracy = float(class_ok) / n_in_dataset
    print("Well classified %5d / %5d, (%5.2f%% accuracy)" % (
        class_ok, n_in_dataset, 100*accuracy))
    print("Average loss : %f" % avg_loss)
    
    all_gt = all_gt.detach().numpy()
    all_pred = all_pred.detach().numpy()
    max_label = all_gt.max()
    
    cm_size = max_label+1 
    confusion_matrix = np.zeros((cm_size, cm_size), dtype=np.int64)
    for i_pred in range(cm_size):
        sel_gt = all_gt[all_pred==i_pred]
        for j_gt in range(cm_size):
            confusion_matrix[i_pred,j_gt] = (sel_gt==j_gt).sum()
            
    return avg_loss, accuracy, confusion_matrix