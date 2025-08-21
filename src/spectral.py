"""bandpass filtering functions
"""

import numpy as np
import scipy.stats as stats
import torch
from torchvision.transforms.functional import center_crop
from skimage.transform import resize
import skimage.morphology as mph
import cv2

def binar_image(image, sigma = 3):

    oimage = get_image_outer(image)

    omean = oimage.mean()
    ostd = oimage.std()

    bin_image = (image >= omean+ostd*sigma) | (image <= omean-ostd*sigma)

    return bin_image

def get_image_outer(image):
    '''
    get outer mask of image.
    '''
    h, w = image.shape
    center = (int(w/2), int(h/2))
    radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    
    masked_image = np.ma.array(image,mask=mask)

    return masked_image

def tight_mask(image, lpass = 8, dilate_radius = 5, gkern_size = 5):

    """
    Function to generate tight mask around particle. 
    """

    #low pass image 
    filt_image,_ = bandpass_image(image, low = lpass)
    
    #get binary image
    bin_image = binar_image(filt_image)

    # dilate image
    disc = mph.disk(dilate_radius)
    dilated_bi = mph.binary_dilation(bin_image, disc)

    #convert binary image to float
    dbi = dilated_bi.astype(float)
    
    #gaussian blur dilated_bi
    gauss_dbi = cv2.GaussianBlur(dbi, (gkern_size, gkern_size), 0)

    return gauss_dbi

def spectrum1d_sinogram(image):

    fourier_image = np.fft.fft(image)
    fourier_amplitudes = np.abs(fourier_image)**2

    return np.sum(fourier_amplitudes,axis=-1)

def spectrum1d(image):

    if image.shape[0] | image.shape[1] > 1024:
        image = resize(image, (1024,1024))
    npix = image.shape[0] if image.shape[0]<image.shape[1] else image.shape[1]

    image = center_crop(torch.from_numpy(image),(npix,npix))

    image=image.numpy()
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    Abins /= np.sum(Abins)

    return kvals,Abins

def spectrum2d(image):

    if image.shape[0] | image.shape[1] > 1024:
        image = resize(image, (1024,1024))
    fourier_image = np.fft.fftn(image)
    fourier_image = np.fft.fftshift(fourier_image)

    return fourier_image

def bandpass_image(image, low = None, high = None, width = 5, order = 20, pixel_size = 1,
                   method = "butter",):
    
    
    lpass = np.inf
    hpass = 0

    spec = spectrum2d(image)

    if ((low is None) and (high is None)):
        raise ValueError("please select atleast a high or a low filter cutoff.")

    if low is not None:
        lpass = spec.shape[0]*pixel_size/low

    if high is not None:
        hpass = spec.shape[0]*pixel_size/high

    bp_spec, mask = bpfilter(spec, lpass, hpass, width, order, method,)
    filt_im_complex = np.fft.ifftshift(bp_spec)
    filt_im_complex = np.fft.ifftn(filt_im_complex)
    filt_im = np.abs(filt_im_complex)
    filt_im -= np.min(filt_im)

    im_range = np.max(filt_im)
    filt_im /= im_range

    filt_im *= 255

    filt_im -= 255

    filt_im = np.abs(filt_im)

    filt_im = filt_im.astype(np.uint8)
    
    return filt_im, mask


def bpfilter(image, low = np.inf, high = 0, width = 5, order = 20, method = "butter",):

    
    mask = np.ones(image.shape)

    if method == "gauss":

        mask = bp_gauss(mask, low, high, width)

    elif method == "butter":

        mask = bp_butter(image, low, high, order)
    
    else:

        raise ValueError("method not found. Please specify either 'gauss' or 'butter'")

    image = np.multiply(image, mask)
    return image,mask


def bp_gauss(mask, low, high, width):
    centre = [mask.shape[0]//2, mask.shape[1]//2]

    for i in range(mask.shape[0]):

        for j in range(mask.shape[1]):

            relpos = np.sqrt((i-centre[0])**2+(j-centre[1])**2)

            if relpos > low:

                mask[i,j] = np.exp(-(relpos-low)**2/(2*width**2))

            elif relpos < high:

                mask[i,j] = np.exp(-(relpos-high)**2/(2*width**2))

    return mask


def bp_butter(mask, low, high, order):


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
