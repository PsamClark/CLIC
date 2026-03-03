"""
Dimensionality Reduction for Sinogram Lines.

This module reduces the dimensionality of sinogram line data using various
linear and nonlinear techniques such as PCA, Isomap, LLE, MDS, t-SNE, UMAP, and TRIMAP.

Input:
    - sinograms: NumPy array of shape (N, nlines, line)
    - config: model choice and number of components

Output:
    - sinograms_transformed: NumPy array of shape (N, nlines, num_comps)
    - model: trained dimensionality reduction model

Dependencies:
    - numpy
    - scikit-learn
    - umap-learn (optional)
    - trimap (optional)
"""
from typing import List, Tuple, Union
import numpy as np

from  sklearn.decomposition import PCA
from  sklearn.manifold import Isomap, LocallyLinearEmbedding as LLE
from  sklearn.manifold import MDS, TSNE
from umap import UMAP
from trimap import TRIMAP

def comp_var(sinos_trans: np.ndarray) -> List[float]:
    """
    Compute variance of each component in the transformed sinogram lines.

    Args:
        sinos_trans: Transformed sinogram lines of shape (N * nlines, num_comps)

    Returns:
        List of variances for each component.
    """
    return [np.var(x) for x in sinos_trans.T]


def split_sinos(sinos: np.ndarray) -> np.ndarray:
    """
    Reshape sinogram matrix into a flat list of lines.

    Args:
        sinos: Sinogram array of shape (N, nlines, line)

    Returns:
        Reshaped array of shape (N * nlines, line)
    """
    return np.reshape(sinos, (-1, sinos.shape[2]))


def fitmodel(sinos: np.ndarray, model_choice: str, num_comps: int
             ) -> Tuple[np.ndarray, Union[object, None]]:
    """
    Fit a dimensionality reduction model to sinogram lines.

    Args:
        sinos: Sinogram array of shape (N, nlines, line)
        model_choice: Name of the model to use ('PCA', 'ISOMAP', 'LLE', etc.)
        num_comps: Number of components to reduce to

    Returns:
        A tuple containing:
            - sinos_trans: Transformed sinogram lines of shape (N * nlines, num_comps)
            - model: Trained model instance
    """
    model = None

    # LINEAR
    if model_choice == 'PCA_skip':
        model = PCA(n_components=num_comps + 1)
    elif model_choice == 'PCA':
        model = PCA(n_components=num_comps)

    # MANIFOLDS
    elif model_choice == 'ISOMAP':
        model = Isomap(n_components=num_comps)
    elif model_choice == 'LLE':
        model = LLE(n_components=num_comps, n_neighbors=5)
    elif model_choice == 'MDS':
        model = MDS(n_components=num_comps)
    elif model_choice == 'TSNE':
        model = TSNE(n_components=num_comps)
    elif model_choice == 'UMAP':
       model = UMAP(n_neighbors=5, min_dist=0.3, n_components=num_comps,random_state=42)
    elif model_choice == 'TRIMAP':
        model = TRIMAP(n_iters=1000)

    lines = split_sinos(sinos)
    if model_choice == "TSNE":
        sinos_trans = model.fit_transform(lines)
        mod_fit = None
        
    else:
        mod_fit = model.fit(lines)
        sinos_trans = model.transform(lines)

    comp_var(sinos_trans)

    if model_choice == 'PCA_skip':
        sinos_trans = sinos_trans[:, 1:]

    return sinos_trans, mod_fit, model
