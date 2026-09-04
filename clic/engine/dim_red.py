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
import joblib
from  sklearn.decomposition import PCA
from  sklearn.manifold import Isomap, LocallyLinearEmbedding as LLE
from  sklearn.manifold import MDS, TSNE, trustworthiness
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


def fitmodel(sinos: np.ndarray, model_choice: str, num_comps: List[int],
             nn: int, min_dist: float, batch_dir:str, save_model:bool
             ) -> np.ndarray:
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
        model = [PCA(n_components=num_comps[0] + 1)]
    elif 'PCA' in model_choice:
        print("hello")
        model = [PCA(n_components=num_comps[0])]

    # MANIFOLDS
    elif model_choice == 'ISOMAP':
        model = [Isomap(n_components=num_comps[0])]
    elif model_choice == 'LLE':
        model = [LLE(n_components=num_comps[0], n_neighbors=5)]
    elif model_choice == 'MDS':
        model = [MDS(n_components=num_comps[0])]
    elif model_choice == 'TSNE':
        model = [TSNE(n_components=num_comps[0])]

    if 'UMAP' in model_choice:
       if 'PCA' in model_choice:
           print("I'm here")
           model.append(UMAP(n_neighbors=nn, 
                                 min_dist=min_dist, 
                                 n_components=num_comps[1],random_state=42))
           
       else:
           model = [UMAP(n_neighbors=nn, 
                                 min_dist=min_dist, 
                                 n_components=num_comps[0],random_state=42)]
    elif model_choice == 'TRIMAP':
        model = [TRIMAP(n_iters=1000)]
    print(model)
    lines = split_sinos(sinos)

    for i,mod in enumerate(model):
        if model_choice == "TSNE":
            lines = mod.fit_transform(lines)
            mod_fit = None
            
        else:
            mod_fit = mod.fit(lines)
            lines = mod.transform(lines)


        if model_choice == 'PCA_skip':
            lines = lines[:, 1:]

        if save_model:
            save_models(lines, mod_fit, mod, i, batch_dir)

    return lines


def save_models(lines, mod_fit, model,i: int,  batch_dir) -> None:

    np.save(f"{batch_dir}/mod_fit{i}.npy",mod_fit)
    np.save(f"{batch_dir}/cents_reddim{i}.npy",
                        lines)

    joblib.dump(model,f"{batch_dir}/dimred{i}.mod")