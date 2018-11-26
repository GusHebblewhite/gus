import gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2

def get_batch(data, batch_size, resize_dims, permute = True, normalise = '01'):
    '''
    A function which handles the processing of images for the network.
    It grabs <batch size> random imges from <data>, resizes them to <resize_dims>, 
    shuffles the order of the dimensions to comply with pytorch (channel, row, column)
    if <permute> is true and then normalises the data. Finlly it converts the tensor to 'float'
    '''
    batch_idx = np.random.choice(range(data.shape[0]), batch_size)
    #Rehape, change image format (int to float) and convert to a tensor:
    batch = [cv2.resize(data[x]/max(255*(normalise=='01'), 1), tuple(resize_dims[::-1])) for x in batch_idx] #Resize
    batch = torch.tensor(batch) #Convert to tensor
    if permute: batch = batch.permute(0,3,1,2)
    batch = batch.float()
    
    return batch


def rescale_AE(dims, rounding = "up"):
    '''
    This function takes some set of dimensions and finds the best (nearest) dimensions
    to scale these to such that we can use kernels to scale the image down to 4x4 or smaller.
    rounding: {'up', 'nearest'}
    '''
    assert rounding in ['up', 'nearest']
    outdims = []
    for each_dim in dims:
        #Create a list of possible sizes:
        sizes = np.outer([2**x for x in (np.arange(-1, 1) + int(np.log2(each_dim)))], [2,3]).flatten()
        
        if rounding == "up":
            outdims.append(min(sizes[sizes>each_dim]))

        elif rounding == "nearest":
            outdims.append(sizes[np.abs(sizes - each_dim).argmin()])

    #Find nearest size to each dimension:
    return outdims

