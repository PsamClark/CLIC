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
from typing import Optional
import random
from pathlib import PurePath
from typing import Tuple, List, Any
from glob import glob

import numpy as np
import pandas as pd
from skimage.transform import radon, resize
import mrcfile
import starfile
import cv2

from clic.utils.spectral import filter_image, tightmask, recentre_image


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


def stand_image(images: np.ndarray) -> np.ndarray:
    """
    Standardize image to zero mean and unit variance.

    Args:
        image: Input image.

    Returns:
        Standardized image.
    """
    return np.array([(x - np.mean(x)) / np.std(x) for x in images])


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
    return np.array([resize(x, (ds, ds), anti_aliasing=True) for x in image])


def circular_mask(image: np.ndarray) -> np.ndarray:
    """
    Apply circular mask to image to reduce edge artifacts.

    Args:
        im: Input image.

    Returns:
        Masked image.
    """
    n, h, w = image.shape
    center = (w // 2, h // 2)
    radius = min(center[0], center[1], w - center[0], h - center[1])
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist <= radius
    mask = mask.reshape(1,mask.shape[0], mask.shape[1]).repeat(n, 0)
    return image * mask


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

def populate_ctf_params(part_locs: pd.DataFrame, optics: pd.DataFrame) -> dict:

    ctf_params = [{

                            'defocusu': part_locs.loc[i,"rlnDefocusU"],
                            'defocusv': part_locs.loc[i,'rlnDefocusV'],
                            'defocus_angle': part_locs.loc[i,'rlnDefocusAngle'],
                            'pixel_size': optics['rlnImagePixelSize'][0],
                            'voltage': optics['rlnVoltage'][0],
                            'spherical_abberation': optics['rlnSphericalAberration'][0],
                            'amplitude_contrast': optics['rlnAmplitudeContrast'][0]

    } for i in range(len(part_locs))]

    return ctf_params

def preprocess(images: np.ndarray, config: Any,ds_size, 
               part_locs: Optional[pd.DataFrame]= None, optics: Optional[pd.DataFrame] = None, 
                rng = None) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Apply full preprocessing pipeline to image.

    Args:
        im: Raw image.
        config: Configuration object.
        ds_size: Downscaled image size.

    Returns:
        Tuple of (sinogram, preprocessed image).
    """
    image_dim = images.ndim
    if image_dim == 2:
        images = images[np.newaxis]

    if config.snr is not None:
        images = np.array([add_noise(x, rng, config.snr) for x in images])
    
    ctf_params = None

    if config.apply_ctf_correction:

        ctf_params = populate_ctf_params(part_locs, optics)

    images, _ = filter_image(
        images, 
        low=config.lowpass, high=config.highpass,
        pixel_size=config.pixel_size,
        ctf_params=ctf_params,
        method=config.filter_method,
        )
    
    if config.tightmask:
        mask = np.array([tightmask(x) for x in images])
        images *= mask
        del mask   
    if config.centre_particles:

      images = recentre_image(images, part_locs,optics)

    images = downscale(images, ds_size)
    images = stand_image(images)
    images = circular_mask(images)
    sinos = make_sinogram(images, config.lines)
    if image_dim == 2:
        sinos = sinos[0]
        images = images[0]
    return sinos, images

def preprocess_multi(images: np.ndarray, config: Any,ds_size, 
               part_locs: Optional[pd.DataFrame]= None, optics: Optional[pd.DataFrame] = None, 
                rng = None) -> Tuple[np.ndarray, np.ndarray]:
    
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
        images = add_noise(images, rng, config.snr)
    
    ctf_params = None

    if images.ndim == 2:
        images = images[np.newaxis]

    if config.apply_ctf_correction:

        ctf_params = populate_ctf_params(part_locs, optics)

    images, _ = filter_image(
        images, 
        low=config.lowpass, high=config.highpass,
        pixel_size=config.pixel_size,
        ctf_params=ctf_params,
        method=config.filter_method,
        )
    
    if config.tightmask:
        mask = tightmask(images)
        images = images*mask
        del mask   
    if config.centre_particles:

      images = recentre_image(images, part_locs, optics)


    images = downscale(images, ds_size)
    images = stand_image(images)
    images = circular_mask(images)
    sinos = make_sinogram(images, config.lines)
    return sinos, images

def multi_mrcs(dset_path: str, 
               ntot: int, 
               rng: np.random.RandomState) -> int:
    """
    Check if dataset path points to multiple .mrcs files.

    Args:
        dset_path: Dataset path.

    Returns:
        1 if multiple .mrcs files, else 0.
    """

    if 'mrc' in dset_path.suffix:
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
                    choice = np.arange(len(mfile.data))
                id_len = len(mdata)
            
            else:
                mdata = mfile.data[np.newaxis,:]
                id_len = 1
            
        
        fids = [str(file)]*id_len
        nids = list(choice.astype(str))
        ids = list(map('@'.join,zip(nids,fids)))
        
        ids = np.array(ids)
        if f == 0: 
            mdata_out = mdata
            ids_out = ids

        else:

            mdata_out = np.concat((mdata_out,mdata))

            ids_out = np.concat((ids_out, ids))
    id_shuffle=np.arange(mdata_out.shape[0])
    rng.shuffle(
        id_shuffle)
    
    print(f"Only {mdata_out.shape[0]} particles available")
    
    return (mdata_out[id_shuffle],ids_out[id_shuffle]), mdata_out.shape[0]

    
def get_part_locs(config: Any, rng) -> Tuple[Any, int]:
    """  
    Load particle locations from dataset path.

    Args:
        config: Configuration object with data_set and num attributes.

    Returns:
        Tuple of (particle locations, number to use).
    """
    dset_path = config.dataset
    optics=None

    if dset_path.suffix == '.star':
        metadata = starfile.read(dset_path)
        particles = metadata['particles']
        particles[["rlnStackID","rlnStackName"]] = particles.rlnImageName.str.split("@", expand=True)
        particles['rlnStackID'] = particles['rlnStackID'].astype(int)

        optics = metadata['optics']
        n_max = len(particles)

        choice = rng.choice(len(particles),
                                        size = config.num,
                                        replace = False)
        
        particles_sub = particles.iloc[choice].reset_index(drop=True)

    elif dset_path.suffix in ['.txt','.mrc','.mrcs']:
        particles_sub, n_max = multi_mrcs(dset_path,config.num,rng)
        print(particles_sub)
    else:
        print(f"Error: Invalid path specification: {dset_path}")
        sys.exit()

    n = min(n_max, config.num)
    print(f"Will use {n} particles")
    return particles_sub, optics, n


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
        parent = PurePath(dset_path).parent
        stack_loc = part_locs.loc[x,'rlnStackName']
        if stack_loc not in stacks:
            stacks[stack_loc] = load_mrc(str(PurePath(parent).joinpath(stack_loc)))
        stack = stacks[stack_loc]
        im = stack if stack.ndim == 2 else stack[part_locs.loc[x,'rlnStackID']-1]
        name_ids = part_locs

    elif dset_path.suffix in ['.txt','.mrc','.mrcs']:
    
        im = part_locs[0][x]
        name_ids.append(part_locs[1][x])
    else:
        print(f"Error: Invalid path specification: {dset_path}")
        sys.exit()
    return im, name_ids


def sinogram_main(config: Any, part_locs: Any, subset: List[int], optics: pd.DataFrame,
                  rng
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
    all_images = None


    for x in subset:
        image, name_ids = open_part(x, part_locs, name_ids, config.dataset)

        if all_images is None:
            ds_size = int(image.shape[0] // config.downscale)
            all_images = np.zeros((subsize, ds_size, ds_size))
            all_images[0] = image
        else:

            all_images[x] = image


    if config.dataset.suffix == '.star':
        part_info = part_locs[subset].reset_index(keep=False)
    else:
        part_info = None

    all_sinos,all_images = preprocess(all_images, config, ds_size, part_info, optics, rng)


    return all_sinos, all_images, subsize, name_ids
