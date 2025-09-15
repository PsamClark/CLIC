"""
Bandpass Filtering Utilities

This module provides functions for frequency-domain filtering of 2D images,
including Gaussian and Butterworth bandpass filters. It also includes tools
for generating tight masks, computing 1D/2D spectra, and binarizing images
based on outer region statistics.

Dependencies:
    - numpy
    - scipy.stats
    - torch
    - torchvision
    - skimage
    - cv2
"""
from typing import Tuple, Optional

import numpy as np
from skimage.transform import resize
import skimage.morphology as mph
import cv2

def binar_image(image: np.ndarray, sigma: float = 3) -> np.ndarray:
    """
    Binarize image based on outer region statistics.

    Args:
        image: Input image.
        sigma: Threshold in standard deviations.

    Returns:
        Binary image.
    """
    oimage = get_image_outer(image)
    omean = oimage.mean()
    ostd = oimage.std()
    return (image >= omean + ostd * sigma) | (image <= omean - ostd * sigma)


def get_image_outer(image: np.ndarray) -> np.ma.MaskedArray:
    """
    Mask central circular region to isolate outer image.

    Args:
        image: Input image.

    Returns:
        Masked array with central region excluded.
    """
    h, w = image.shape
    center = (w // 2, h // 2)
    radius = min(center[0], center[1], w - center[0], h - center[1])
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist <= radius
    return np.ma.array(image, mask=mask)


def tight_mask(image: np.ndarray, lpass: int = 8,
               dilate_radius: int = 5, gkern_size: int = 5) -> np.ndarray:
    """
    Generate tight mask around particle using bandpass filtering and dilation.

    Args:
        image: Input image.
        lpass: Low-pass cutoff for initial filtering.
        dilate_radius: Radius for morphological dilation.
        gkern_size: Kernel size for Gaussian blur.

    Returns:
        Smoothed mask image.
    """
    filt_image, _ = bandpass_image(image, low=lpass)
    bin_image = binar_image(filt_image)
    disc = mph.disk(dilate_radius)
    dilated = mph.binary_dilation(bin_image, disc).astype(float)
    return cv2.GaussianBlur(dilated, (gkern_size, gkern_size), 0)


def spectrum1d_sinogram(image: np.ndarray) -> np.ndarray:
    """
    Compute 1D power spectrum of sinogram.

    Args:
        image: Sinogram image.

    Returns:
        1D spectrum.
    """
    fourier_image = np.fft.fft(image)
    return np.sum(np.abs(fourier_image) ** 2, axis=-1)


def spectrum2d(image: np.ndarray) -> np.ndarray:
    """
    Compute 2D Fourier spectrum of image.

    Args:
        image: Input image.

    Returns:
        Shifted 2D Fourier spectrum.
    """
    if image.shape[0] | image.shape[1] > 1024:
        image = resize(image, (1024, 1024))
    return np.fft.fftshift(np.fft.fftn(image))


def bandpass_image(image: np.ndarray,
                   low: Optional[float] = None,
                   high: Optional[float] = None,
                   width: int = 5,
                   order: int = 20,
                   pixel_size: float = 1,
                   method: str = "butter") -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply bandpass filter to image in Fourier domain.

    Args:
        image: Input image.
        low: Low-pass cutoff (in spatial units).
        high: High-pass cutoff.
        width: Gaussian width (if method='gauss').
        order: Butterworth filter order.
        pixel_size: Pixel size scaling.
        method: 'gauss' or 'butter'.

    Returns:
        Tuple of (filtered image, filter mask).
    """
    if low is None and high is None:
        raise ValueError("Please specify at least one cutoff (low or high).")

    spec = spectrum2d(image)
    lpass = np.inf if low is None else spec.shape[0] * pixel_size / low
    hpass = 0 if high is None else spec.shape[0] * pixel_size / high

    bp_spec, mask = bpfilter(spec, lpass, hpass, width, order, method)
    filt_im = np.abs(np.fft.ifftn(np.fft.ifftshift(bp_spec)))
    filt_im = (filt_im - np.min(filt_im)) / np.ptp(filt_im) * 255
    filt_im = np.abs(filt_im - 255).astype(np.uint8)

    return filt_im, mask


def bpfilter(image: np.ndarray,
             low: float = np.inf,
             high: float = 0,
             width: int = 5,
             order: int = 20,
             method: str = "butter") -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply bandpass filter mask to Fourier image.

    Args:
        image: Fourier-transformed image.
        low: Low-pass cutoff.
        high: High-pass cutoff.
        width: Gaussian width.
        order: Butterworth order.
        method: 'gauss' or 'butter'.

    Returns:
        Tuple of (filtered spectrum, mask).
    """
    mask = np.ones(image.shape)
    if method == "gauss":
        mask = bp_gauss(mask, low, high, width)
    elif method == "butter":
        mask = bp_butter(mask, low, high, order)
    else:
        raise ValueError("Method must be 'gauss' or 'butter'.")

    return image * mask, mask


def bp_gauss(mask: np.ndarray, low: float, high: float, width: int) -> np.ndarray:
    """
    Create Gaussian bandpass filter mask.

    Args:
        mask: Initial mask array.
        low: Low-pass cutoff.
        high: High-pass cutoff.
        width: Gaussian width.

    Returns:
        Filter mask.
    """
    centre = [mask.shape[0] // 2, mask.shape[1] // 2]
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            relpos = np.sqrt((i - centre[0]) ** 2 + (j - centre[1]) ** 2)
            if relpos > low:
                mask[i, j] = np.exp(-(relpos - low) ** 2 / (2 * width ** 2))
            elif relpos < high:
                mask[i, j] = np.exp(-(relpos - high) ** 2 / (2 * width ** 2))
    return mask


def bp_butter(mask: np.ndarray, low: float, high: float, order: int) -> np.ndarray:
    """
    Create Butterworth bandpass filter mask.

    Args:
        mask: Initial mask array.
        low: Low-pass cutoff.
        high: High-pass cutoff.
        order: Filter order.

    Returns:
        Filter mask.
    """
    centre = [mask.shape[0]//2, mask.shape[1]//2]
    x_coord_vec = np.linspace(0, mask.shape[1], mask.shape[1])
    y_coord_vec = np.linspace(0, mask.shape[0], mask.shape[0])
    x_coord_mat, y_coord_mat = np.meshgrid(x_coord_vec, y_coord_vec, sparse=True)
    sq1 = (y_coord_mat - centre[0]) ** 2
    sq2 = (x_coord_mat - centre[1]) ** 2
    sqs = sq1 + sq2

    sq = np.sqrt(
            sqs
        )
    bwidth = 2*low
    bmid = 1
    if low == np.inf:

        bwidth =high*2

        bmid = mask.shape[0]//2

    elif high > 0:
        bwidth = low - high

        bmid = (low + high)/2
    mask = np.divide(1,np.sqrt(1 + ((sq - (bmid - 1))/(bwidth/2))**(2*order)))
    return mask
