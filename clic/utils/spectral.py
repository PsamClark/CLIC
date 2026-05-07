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


def tightmask(image: np.ndarray, lpass: int = 8,
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
    filt_image, _ = filter_image(image, low=lpass)
    bin_image = binar_image(filt_image)
    disc = mph.disk(dilate_radius)
    dilated = mph.binary_dilation(bin_image, disc).astype(float)
    gauss_tmask = cv2.GaussianBlur(dilated, (gkern_size, gkern_size), 0)
    return gauss_tmask.astype(np.float32)

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


def filter_image(image: np.ndarray,
                   low: Optional[float] = None,
                   high: Optional[float] = None,
                   width: int = 5,
                   order: int = 2,
                   pixel_size: float = 1,
                   ctf_params: Optional[dict] = None, 
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


    spec = spectrum2d(image)

    if low is None and high is None:
        mask=np.ones(spec.shape)

    else:
        lpass = np.inf if low is None else spec.shape[0] * pixel_size / low
        hpass = 0 if high is None else spec.shape[0] * pixel_size / high

        mask = bandpass_mask(spec, lpass, hpass, width, order, method)

    if ctf_params is not None:
        
        ctf=generate_ctf(image.shape[0],**ctf_params)

        mask *= ctf
    bp_spec = spec*mask
    filt_im = np.fft.ifftn(np.fft.ifftshift(bp_spec)).real
    filt_im = (filt_im - np.min(filt_im)) / np.ptp(filt_im) * 255

    return filt_im.astype(np.uint8), mask.astype(np.float32)


def bandpass_mask(image: np.ndarray,
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

    return mask


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

def correct_ctf(
        image,
        dfcsu,
        dfcsv,
        angle, 
        pixel_size=1,
        voltage=300,
        spherical_abb=2.7,
        amp_contrast=0.1,
        snr=20,
        method="flip"
        ):
    
    spec=spectrum2d(image)

    ctf=generate_ctf(
        spec,
        dfcsu,
        dfcsv,
        angle, 
        pixel_size,
        voltage,
        spherical_abb,
        amp_contrast, 
        )
    
    if method=="flip":

        filt_im = spec*np.sign(ctf)

    else:
        filt_im = (spec*ctf)/(ctf**2 + 1/snr)

    filt_im = np.fft.ifftn(np.fft.ifftshift(filt_im)).real
    filt_im2 = (filt_im - np.min(filt_im)) / np.ptp(filt_im) * 255

    return filt_im2,filt_im


def generate_ctf(
        spec_shape,
        defocusu,
        defocusv,
        defocus_angle, 
        pixel_size=1,
        voltage=300,
        spherical_abberation=2.7,
        amplitude_contrast=0.1, 
        ):


    freq_arr = np.arange(-spec_shape/2,spec_shape/2)/(pixel_size*spec_shape)
    x0, x1 = np.meshgrid(freq_arr,freq_arr)
    freq_arr = np.stack([x0.ravel(),x1.ravel()],axis=1)

    ctf = compute_ctf(freq_arr,
                    defocusu,
                    defocusv,
                    defocus_angle,
                    voltage,
                    spherical_abberation,
                    amplitude_contrast)



    ctf=ctf.reshape((spec_shape,spec_shape))
    return ctf

def compute_ctf(
    freqs: np.ndarray,
    dfu: float,
    dfv: float,
    dfang: float,
    volt: float,
    cs: float,
    w: float,
    phase_shift: Optional[float] = None,
    scalefactor: Optional[float] = None,
    bfactor: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the 2D CTF

    Input:
        freqs: Nx2 array of 2D spatial frequencies
        dfu: DefocusU (Angstrom)
        dfv: DefocusV (Angstrom)
        dfang: DefocusAngle (degrees)
        volt: accelerating voltage (kV)
        cs: spherical aberration (mm)
        w: amplitude contrast ratio
        phase_shift: degrees
        scalefactor : scale factor
        bfactor: envelope fcn B-factor (Angstrom^2)
    """
    # convert units
    volt = volt * 1000
    cs = cs * 10**7
    dfang = dfang * np.pi / 180
    if phase_shift is None:
        phase_shift = 0
    phase_shift = phase_shift * np.pi / 180

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / np.sqrt(volt + 0.97845e-6 * volt**2)
    x = freqs[..., 0]
    y = freqs[..., 1]
    ang = np.arctan2(y, x)
    s2 = x**2 + y**2
    df = 0.5 * (dfu + dfv + (dfu - dfv) * np.cos(2 * (ang - dfang)))
    gamma = (
        2 * np.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam**3 * s2**2)
        - phase_shift
    )
    ctf = np.sqrt(1 - w**2) * np.sin(gamma) - w * np.cos(gamma)
    if scalefactor is not None:
        ctf *= scalefactor
    if bfactor is not None:
        ctf *= np.exp(-bfactor / 4 * s2)

    return ctf

def calculate_snr(image):

    return np.mean(image)/np.std(get_image_outer(image))


def recentre_image(image,x_offset, y_offset):

    print(x_offset)
    print(y_offset)

    if x_offset > 0:
        im2 = np.pad(image, ((x_offset, 0), (0, 0)), mode='constant')
        im2 = im2[:image.shape[0]-x_offset, :]
    else:
        im2 = np.pad(image, ((0, -x_offset), (0, 0)), mode='constant')
        im2 = im2[-x_offset:, :]

    if y_offset > 0:
        im3 = np.pad(im2, ((0, 0), (y_offset, 0)), mode='constant')
        im3 = im3[:, :image.shape[0]-y_offset]

    else:
        im3 = np.pad(im2, ((0, 0), (0, -y_offset)), mode='constant')
        im3 = im3[:, -y_offset:]

    return im3