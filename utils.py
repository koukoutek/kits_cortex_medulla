import numpy as np
import torch
from monai.transforms import MapTransform
import nrrd

from monai.data import MetaTensor
from pathlib import Path


class ReadNrrdMaskd(MapTransform):
    """A MONAI MapTransform for loading and processing NRRD segmentation masks.

    This transform:
    1. Loads a NRRD format mask file
    2. Separates multi-class mask into individual binary channels
    3. Creates a background channel
    4. Converts to MetaTensor with proper metadata

    Parameters
    ----------
    None (this is a concrete implementation of MapTransform)

    Input/Output
    ------------
    Operates on dictionary-style data with the following keys:

    Inputs:
        - 'mask': str or Path
            Path to the NRRD mask file
        - 'image_meta_dict': dict
            Metadata dictionary from corresponding image

    Outputs:
        - 'mask': MetaTensor
            4-channel binary mask tensor (background, cortex, medulla, mass)
        - 'mask_meta_dict': dict
            Copied image metadata dictionary
        - Removes 'image_path' key

    Processing Details
    -----------------
    Mask value mapping:
        - 1 → cortex (channel 1)
        - 3 → medulla (channel 2, converted to value 1)
        - 2 → mass (channel 3, converted to value 1)
        - Background (channel 0) is computed as 1 - (sum of other channels)

    Notes
    -----
    - Expects NRRD masks with specific label values (1, 2, 3)
    - Output mask will have shape (4, H, W, D) for 3D volumes
    - Preserves spatial metadata through MetaTensor
    - Assumes mask and image have same spatial characteristics
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
    """A MONAI MapTransform for CT image windowing based on intensity percentiles.

    This transform performs three key operations:
    1. Clips extreme low values (below -200 HU) to -200
    2. Applies percentile-based windowing (5th to 99th percentiles)
    3. Normalizes the intensities to [0, 1] range

    Parameters
    ----------
    None (this is a concrete implementation of MapTransform)

    Input/Output
    ------------
    Operates on dictionary-style data with the following keys:
    
    Inputs:
        - 'image': torch.Tensor 
            The input CT image in HU units (3D volume or 2D slice)
            
    Outputs:
        - 'image': torch.Tensor
            The windowed and normalized image (values in [0, 1])
        - 'image_shape': tuple
            The spatial dimensions of the image (excluding channels)

    Notes
    -----
    - Assumes input is in Hounsfield Units (HU)
    - Performs in-place modification of the input tensor
    - Preserves the original tensor's dtype
    - The 5th-99th percentile windowing helps exclude outliers
    - -200 HU clipping removes extreme low-density values (e.g., air)
    """

    def __call__(self, data):
        d = dict(data)

        im_copy = d['image'].clone()
        upper_value = np.percentile(im_copy, 99)
        lower_value = np.percentile(im_copy, 5)

        d['image'][ d['image'] < -200 ] = -200
        d['image'] = torch.clip(d['image'], lower_value, upper_value)
        d['image'] = (d['image'] - d['image'].min())/(d['image'].max() - d['image'].min())

        d['image_shape'] = d['image'][0].shape

        return d


class IndexTracker:
    """Interactive 3D volume slicer using mouse scroll wheel navigation.

    This class creates a matplotlib interactive plot that allows scrolling through
    slices of a 3D volume array using the mouse wheel. The display updates in real-time
    with the current slice number shown in the y-label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object where the image will be plotted.
    X : numpy.ndarray
        A 3D array (rows, columns, slices) containing the volume data to display.
        Must have at least 3 dimensions.
    vmin, vmax : float
        The minimum and maximum values for the image display range (contrast adjustment).

    Attributes
    ----------
    im : matplotlib.image.AxesImage
        The image object displayed on the axes.
    ind : int
        Current slice index being displayed.
    slices : int
        Total number of slices in the volume (3rd dimension of X).

    Notes
    -----
    - Requires matplotlib connection to scroll events
    - Default colormap is grayscale ('gray')
    - Wraps around when reaching first/last slice
    """
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

def evaluate_true_false(inp):
    """
    Converts a string or input into a boolean (`True` or `False`) based on partial matches with `"TRUE"` or `"FALSE"`.

    Parameters
    -----------
    inp : str, int, or any type convertible to string
        The input to evaluate. The function checks if the input (case-insensitive) partially matches 
        the start of `"TRUE"` or `"FALSE"`.

    Returns
    --------
    bool
        - `True` if `inp` (when converted to uppercase) is a prefix of `"TRUE"` (e.g., `"T"`, `"TR"`, `"TRUE"`).
        - `False` if `inp` is a prefix of `"FALSE"` (e.g., `"F"`, `"FA"`, `"FALSE"`).

    Raises
    -------
    ValueError
        If `inp` does not match the start of either `"TRUE"` or `"FALSE"`.

    Notes
    ------
    - The comparison is case-insensitive (e.g., `"t"`, `"FaL"` are valid).
    - Only the starting characters of the input are checked (e.g., `"TRU"` works, but `"RUE"` does not).
    """
    inp = str(inp).upper()
    if 'TRUE'.startswith(inp):
        return True
    elif 'FALSE'.startswith(inp):
        return False
    else:
        raise ValueError('Argument error. Expected bool type.')


def dice(im1, im2, empty_score=1.0):
    """Computes the Dice coefficient, a measure of set similarity.
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
