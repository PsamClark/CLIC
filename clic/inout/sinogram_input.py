"""
Sinogram Generation Pipeline.

This module loads particle projections from various formats (.mrc, .mrcs, .star),
applies preprocessing steps (noise, translation, masking, filtering), and generates
sinograms via Radon transform.

Inputs:
    - config: Configuration object from main.py
    - dataset: Particle images in .mrc, .mrcs, or .star format

Outputs:
    - all_sinos: Array of sinograms (N × nlines × line_length)
    - all_ims: Preprocessed images (N × ds_size × ds_size)
    - subsize: Number of particles processed
    - name_ids: List of image identifiers

Dependencies:
    - numpy
    - skimage
    - mrcfile
    - gemmi
    - cv2
    - spectral (custom module)
"""

import sys
import random
from pathlib import PurePath
from typing import Tuple, List, Any
from glob import glob

import numpy as np
from skimage.transform import radon, resize
import mrcfile
import gemmi
import cv2

from clic.utils.spectral import bandpass_image, tightmask


def load_mrc(path: str) -> np.ndarray:
    """
    Load an MRC file and return the first slice if 3D.

    Args:
        path: Path to the .mrc file.

    Returns:
        2D image array.
    """
    with mrcfile.mmap(path,'r') as f:
        image = f.data
    return image


def stand_image(image: np.ndarray) -> np.ndarray:
    """
    Standardize image to zero mean and unit variance.

    Args:
        image: Input image.

    Returns:
        Standardized image.
    """
    return (image - np.mean(image)) / np.std(image)


def add_noise(image: np.ndarray, rng, snr: float = 1) -> np.ndarray:
    """
    Add Gaussian noise to image based on signal-to-noise ratio.

    Args:
        image: Input image.
        snr: Desired signal-to-noise ratio.

    Returns:
        Noisy image.
    """
    sigma = np.sqrt(np.var(image) / snr)
    noise = rng.normal(0, sigma, image.shape)
    return image + noise


def add_trans(image: np.ndarray, trans: float = 0.05) -> np.ndarray:
    """
    Apply random translation to image.

    Args:
        image: Input image.
        trans: Fractional translation magnitude.

    Returns:
        Translated image.
    """
    shift = np.random.normal(0, trans * image.shape[0])
    direction = random.uniform(0, 2 * np.pi)
    mat = np.float32([[1, 0, shift * np.cos(direction)],
                    [0, 1, shift * np.sin(direction)]])
    return cv2.warpAffine(image, mat, image.shape)


def downscale(image: np.ndarray, ds: int) -> np.ndarray:
    """
    Resize image to ds × ds.

    Args:
        image: Input image.
        ds: Target dimension.

    Returns:
        Resized image.
    """
    return resize(image, (ds, ds), anti_aliasing=True)


def circular_mask(im: np.ndarray) -> np.ndarray:
    """
    Apply circular mask to image to reduce edge artifacts.

    Args:
        im: Input image.

    Returns:
        Masked image.
    """
    h, w = im.shape
    center = (w // 2, h // 2)
    radius = min(center[0], center[1], w - center[0], h - center[1])
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist <= radius
    return im * mask


def make_sinogram(image: np.ndarray, nlines: int = 120) -> np.ndarray:
    """
    Generate sinogram from image using Radon transform.

    Args:
        image: Input image.
        nlines: Number of projection angles.

    Returns:
        Transposed sinogram array.
    """
    theta = np.linspace(0., 360., nlines, endpoint=False)
    return radon(image, theta=theta, circle=True).T


def find_im_size(path: str) -> int:
    """
    Get image size from first .mrc file in dataset.

    Args:
        path: Path prefix to image.

    Returns:
        Image size (height).
    """
    im = load_mrc(path + '0.mrc')
    return im.shape[0]


def gblur(im: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur to image.

    Args:
        im: Input image.

    Returns:
        Blurred image.
    """
    return cv2.GaussianBlur(im, (5, 5), 0)


def preprocess(im: np.ndarray, config: Any, ds_size: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply full preprocessing pipeline to image.

    Args:
        im: Raw image.
        config: Configuration object.
        ds_size: Downscaled image size.

    Returns:
        Tuple of (sinogram, preprocessed image).
    """
    if config.snr is not None:
        im = add_noise(im, rng, config.snr)

    im, _ = bandpass_image(
        im, low=config.lowpass, high=config.highpass,
        pixel_size=config.pixel_size,
        method=config.filter_method
        )
    
    if config.tightmask:
        mask = tightmask(im)
        im = im*mask
        del mask   

    im = downscale(im, ds_size)
    im = stand_image(im)
    im = circular_mask(im)
    sino = make_sinogram(im, config.lines)
    return sino, im

def multi_mrcs(dset_path: str, ntot: int, rng) -> int:
    """
    Check if dataset path points to multiple .mrcs files.

    Args:
        dset_path: Dataset path.

    Returns:
        1 if multiple .mrcs files, else 0.
    """

    if 'mrc' in dset_path.suffix:
        print(str(dset_path))
        files = glob(str(dset_path)) 

    elif 'txt' in dset_path.suffix:

        files = [
            i.strip('\n') for  i in open(
                dset_path, 'r').readlines() if 'mrc' in PurePath(i).suffix]
        
    if len(files) == 0:
        print("No mrc files found!")

    nsub = ntot // len(files)

    for f,file in  enumerate(files):

        with mrcfile.mmap(file,'r') as mfile:
            
            if mfile.data.ndim==3: 
                if mfile.data.shape[0] > nsub:
                    choice = rng.choice(len(mfile.data),
                                        size = nsub,
                                        replace = False)
                    mdata = mfile.data[choice]
                

                else:
                    mdata = mfile.data[:]
                id_len = len(mdata)
            
            else:
                mdata = mfile.data[np.newaxis,:]
                id_len = 1
            
        
        ids = np.array([file]*id_len, dtype='S')
        ids = np.char.add(ids,choice.astype('S'))
           
        if f == 0: 
            mdata_out = mdata
            ids_out = ids

        else:

            mdata_out = np.concat((mdata_out,mdata))

            ids_out = np.concat((ids_out, ids))

    print(f"Only {mdata_out.shape[0]} particles available")
    
    return (mdata_out,ids_out), mdata_out.shape[0]

    
def get_part_locs(config: Any, rng) -> Tuple[Any, int]:
    """  
    Load particle locations from dataset path.

    Args:
        config: Configuration object with data_set and num attributes.

    Returns:
        Tuple of (particle locations, number to use).
    """
    dset_path = config.dataset

    if dset_path.suffix == '.star':
        starfile = gemmi.cif.read_file(dset_path)
        block = starfile.find_block('particles')
        part_locs = list(block.find_values('_rlnimagename'))
        n_max = len(part_locs)

    elif dset_path.suffix in ['.txt','.mrc','.mrcs']:
        part_locs, n_max = multi_mrcs(dset_path,config.num,rng)
    
    else:
        print(f"Error: Invalid path specification: {dset_path}")
        sys.exit()

    n = min(n_max, config.num)
    print(f"Will use {n} particles")
    return part_locs, n


def open_part(x: int, part_locs: Any, name_ids: List[str], dset_path: str,
              stacks: dict = None) -> Tuple[np.ndarray, List[str]]:
    """
    Load a single particle image from dataset.

    Args:
        x: Index of particle.
        part_locs: List or array of particle locations.
        name_ids: List to append image identifier.
        dset_path: Path to dataset.
        stacks: Optional cache of loaded stacks.

    Returns:
        Tuple of (image, updated name_ids).
    """
    if stacks is None:
        stacks = {}

    if dset_path.suffix == '.star':
        im_loc = part_locs[x]
        ind, stack_loc = im_loc.split('@')
        if stack_loc not in stacks:
            stacks[stack_loc] = load_mrc(stack_loc)
        stack = stacks[stack_loc]
        im = stack if stack.ndim == 2 else stack[int(ind) - 1]
        name_ids.append(im_loc)

    elif dset_path.suffix in ['.txt','.mrc','.mrcs']:
    
        im = part_locs[0][x]
        name_ids.append(part_locs[1][x])
    else:
        print(f"Error: Invalid path specification: {dset_path}")
        sys.exit()

    return im, name_ids


def sinogram_main(config: Any, part_locs: Any, subset: List[int], rng
                  ) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
    """
    Main function to generate sinograms from a subset of particles.

    Args:
        config: Configuration object.
        part_locs: Particle locations.
        subset: List of indices to process.

    Returns:
        Tuple of (sinograms, images, subset size, image identifiers).
    """
    name_ids: List[str] = []
    subsize = len(subset)
    all_sinos = None
    all_ims = None
    for x, x_sb in enumerate(subset):
        im, name_ids = open_part(x_sb, part_locs, name_ids, config.dataset)

        if x == 0:
            ds_size = int(im.shape[0] // config.downscale)
            all_sinos = np.zeros((subsize, config.lines, ds_size))
            all_ims = np.zeros((subsize, ds_size, ds_size))

        sino,imout = preprocess(im, config, ds_size, rng)
        all_sinos[x] = sino
        all_ims[x] = imout

    return all_sinos,all_ims, subsize, name_ids
