"""
Image Reconstruction via Dimensionality Reduction.

This module provides tools for:
    - Generating synthetic time‑series datasets.
    - Applying circular masking to images to minimise Radon transform artefacts.
    - Performing Radon transforms to create sinograms.
    - Using dimensionality reduction models (PCA or UMAP) to reconstruct sinograms.
    - Inverse Radon transforms (FBP and SART variants) to recover images.
    - Plotting learned components from models.
    - Saving all intermediate and final results as image files.

The workflow is:
    1. Generate or load an image.
    2. Apply a circular mask.
    3. Compute the sinogram via Radon transform.
    4. Reduce dimensions with PCA/UMAP and reconstruct the sinogram.
    5. Perform inverse transforms to reconstruct the image.
    6. Save components and reconstruction results to disk.

Dependencies:
    - numpy
    - scikit-image
    - matplotlib

Example:
    Run this module directly to execute the example pipeline::

        python reconstruction_module.py
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray
from skimage.transform import iradon, iradon_sart
import matplotlib.pyplot as plt


def save_image_figure(
    image: NDArray[np.floating],
    title: str,
    filename: str,
    cmap: str = "gray"
) -> None:
    """Display an image in a figure and save it to disk.

    Args:
        image (ndarray): 2D array to display.
        title (str): Figure title.
        filename (str): Output filename for saving.
        cmap (str, optional): Colormap for rendering. Defaults to "gray".
    """
    plt.figure(title)
    plt.imshow(image, cmap)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')


def plt_comps(mapper: Any) -> None:
    """Plot the components from a fitted dimensionality reduction model.

    This function visualizes the learned components (e.g., from PCA or UMAP)
    and saves the figure as 'components.png'.

    Args:
        mapper: A fitted model instance with a ``components_`` attribute
            (e.g., sklearn.decomposition.PCA).
    """
    comps: NDArray[np.floating] = mapper.components_

    plt.figure('Components')
    for i in range(1, comps.shape[0]):
        plt.subplot(comps.shape[0] // 10, 10, i)
        plt.plot(comps[i - 1])
        plt.xticks([])
        plt.yticks([])
    plt.savefig('components.png', bbox_inches='tight')


def recon_sino(
    im: NDArray[np.floating],
    mapper: Any,
    config: Any
) -> None:
    """Reconstruct an image from its sinogram using a dimensionality reduction model.

    Depending on the `config.model`, this function will transform the sinogram
    using PCA or UMAP, reconstruct it, and then perform inverse Radon transforms
    (FBP and SART variants). It saves the reconstruction images to disk.

    Args:
        im (ndarray): Input image or sinogram (depending on preprocessing).
        mapper: A fitted PCA or UMAP model with ``transform`` and
            ``inverse_transform`` (for UMAP) methods.
        config: An object with attributes:
            - nlines (int): Number of projection lines/angles.
            - model (str): Dimensionality reduction method ("PCA" or "UMAP").
            - snr (float or int): Signal-to-noise ratio used in file naming.
            - num_comps (int): Number of components used in the model.
    """
    theta: NDArray[np.floating] = np.linspace(0., 360., config.nlines, endpoint=False)
    sino: NDArray[np.floating] = im

    if config.model == "PCA":
        comps = mapper.components_
        sino_pca = mapper.transform(sino)
        sino_recon = np.dot(sino_pca[:, :], comps[:, :])
    elif config.model == "UMAP":
        sino_umap = mapper.transform(sino)
        sino_recon = mapper.inverse_transform(sino_umap)
    else:
        raise ValueError(f"Unsupported model: {config.model}")

    im_recon_fbp = iradon(sino_recon.T, theta=theta, filter_name='ramp')
    reconstruction_sart = iradon_sart(sino_recon.T, theta=theta)
    reconstruction_sart2 = iradon_sart(
        sino_recon.T, theta=theta, image=reconstruction_sart)
    reconstruction_sart3 = iradon_sart(
        sino_recon.T, theta=theta, image=reconstruction_sart2)

    # Save outputs
    save_image_figure(
        sino,
        "original",
        f'snr{config.snr}_sino.png'
    )
    save_image_figure(
        sino_recon,
        "recon",
        f'snr{config.snr}_model{config.model}_comp{config.num_comps}_recon_sino.png'
    )
    save_image_figure(
        im_recon_fbp,
        "recon fbp image",
        f'snr{config.snr}_model{config.model}_comp{config.num_comps}_recon_fbp.png'
    )
    save_image_figure(
        reconstruction_sart,
        "recon sart1 image",
        'recon_sart1.png'
    )
    save_image_figure(
        reconstruction_sart3,
        "recon sart3 image",
        f'snr{config.snr}_model{config.model}_comp{config.num_comps}_recon_sart3.png'
    )



