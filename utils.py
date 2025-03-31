import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai.transforms import MapTransform, InvertibleTransform, Resize
import nibabel as nib
import open3d as o3d
import nrrd

from math import floor, ceil
from monai.networks.nets import UNet
from rt_utils import RTStructBuilder
from monai.data import MetaTensor
from pathlib import Path


class ReadNrrdMaskd(MapTransform):
    """
        Read Nrrd file and return the mask
    """
    def __call__(self, data):
        d = dict(data)
        mask_path = str(d['mask'])
        mask = nrrd.read(Path(mask_path)) # returns a tuple with the first element as the mask and the second element as the header
        mask = mask[0]  # Extract the mask array from the tuple
        
        cortex = np.where(mask == 1, mask, 0)
        medulla = np.where(mask == 3, mask, 0)
        medulla[medulla == 3] = 1
        mass = np.where(mask == 2, mask, 0)
        mass[mass == 2] = 1
        background = np.ones_like(mask) - (cortex + medulla + mass)
        mask = np.stack((background, cortex, medulla, mass), axis=0)

        d['mask_meta_dict'] = d['image_meta_dict']
        meta = d['mask_meta_dict']
        affine = d['mask_meta_dict']['original_affine']
        d['mask'] = MetaTensor(mask.astype(np.float32), affine=affine, meta=meta)
        del d['image_path']
        return d
    

class WindowindCTBasedOnPercentiled(MapTransform):
    """
    Applies windowing to CT images based on percentiles.

    Args:
        data (dict): Input data dictionary containing 'image' key.

    Returns:
        dict: Modified data dictionary with windowed 'image' and additional 'image_shape' key.
    """

    def __call__(self, data):
        d = dict(data)

        im_copy = d['image'].clone()
        # windowing should be based on the energy level
        upper_value = np.percentile(im_copy, 99)
        lower_value = np.percentile(im_copy, 5)

        d['image'][ d['image'] < -200 ] = -200
        d['image'] = torch.clip(d['image'], lower_value, upper_value)
        d['image'] = (d['image'] - d['image'].min())/(d['image'].max() - d['image'].min())

        d['image_shape'] = d['image'][0].shape

        return d


class IndexTracker:
    def __init__(self, ax, X, vmin, vmax):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2
        self.vmin = vmin
        self.vmax = vmax

        self.im = ax.imshow(self.X[:, :, self.ind], vmax=self.vmax, vmin=self.vmin, cmap='gray') #cmap='gray',
        self.update()

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step)) # print step and direction
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

# # How to user IndexTracker
# fig, ax = plt.subplots(1, 1)
# X = np.transpose(ct, axes=(2,1,0)) # image to show
# tracker = IndexTracker(ax, X, vmin=np.amin(X), vmax=np.amax(X))
# fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
# plt.show()

def evaluate_true_false(inp):
    inp = str(inp).upper()
    if 'TRUE'.startswith(inp):
        return True
    elif 'FALSE'.startswith(inp):
        return False
    else:
        raise ValueError('Argument error. Expected bool type.')


def read_model(model_path):
    """
    Reads a saved model from the specified path and returns the model.

    Args:
        model_path (str): The path to the saved model file.

    Returns:
        torch.nn.Module: The loaded model.

    """
    saved_model = torch.load(model_path)

    # KiTS model
    model = UNet(spatial_dims=3, in_channels=1, out_channels=3, kernel_size=3, up_kernel_size=3, channels=[32, 64, 128, 256, 512],
                        strides=[2, 2, 2, 2], norm='instance', dropout=.3, num_res_units=3)
    
    model_dict = saved_model['model_state_dict']

    new_dict = {}
    for k,v in model_dict.items():
        if str(k).startswith('module'): # module will be there in case of training in multiple GPUs
            new_dict[k[7:]] = v
        else:
            new_dict[k] = v
    model.load_state_dict(new_dict)
    
    return model


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum
