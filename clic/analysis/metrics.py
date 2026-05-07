"""
Metric Determination Script.

This module provides utilities to:
    - Align clustering outputs across batches
    - Compute clustering accuracy using permutation matching
    - Visualize clustering assignments as heatmaps

Inputs:
    - cluster_matrix.npy: Numpy array of clustering assignments
    - particle_ids.txt: List of particle identifiers

Outputs:
    - Accuracy score (float)
    - Cluster heatmap (via seaborn)

Dependencies:
    - numpy
    - pandas
    - seaborn
    - itertools
    - min_matrix.align_batches, min_matrix.make_line
"""
from typing import Union,Tuple,List
from itertools import permutations

import numpy as np
import pandas as pd
import seaborn as sns

from ..utils.min_matrix import align_batches, make_line


def cluster_accuracy(cc_piv: pd.DataFrame) -> float:
    """
    Determine clustering accuracy by finding the best permutation alignment.

    Args:
        cc_piv: Pivot table of cluster counts (classes × experiment IDs).

    Returns:
        Accuracy as a percentage.
    """
    if cc_piv is None:
        return None

    cc_vals = cc_piv.values
    total = np.sum(cc_vals)
    classes = cc_vals.shape[1]

    if cc_piv.index[0] == -1:
        cc_vals = cc_vals[1:]
    if cc_vals.shape[0] < cc_vals.shape[1]:
        cc_vals=cc_vals.T
    perms = permutations(np.arange(len(cc_vals)),cc_vals.shape[1])
    stored_count = 0
    for pp in perms:
        count = sum(cc_vals[pp[i],i] for i in range(len(pp)))
        stored_count = max(stored_count, count)

    correct = stored_count / float(total) * 100
    if correct < (100/classes):
        correct = (100/classes)
    return correct


def optimize_clustering(clic_exp: str, dir = None) -> pd.DataFrame:
    """
    Align clustering results and format them into a pivot table.

    Args:
        clic_exp: Path to experiment directory containing clustering results.

    Returns:
        Pivot table of class counts per experiment ID.
    """
    if dir is not None:
        mpath = f"{dir}/{clic_exp}"
    else:
        mpath = clic_exp
    try:
        data = np.load(f"{mpath}/cluster_matrix.npy")
    except:
        return None

    aligned_matrix = align_batches(data)
    all_classes = make_line(aligned_matrix)

    with open(f"{mpath}/particle_ids.txt", "r") as f:
        ids = [line.split('/')[-1][:4] for line in f.readlines()]
    ids = ids[:len(all_classes)]


    fmt_classes = pd.DataFrame({"id": ids, "class": all_classes})
    class_counts = fmt_classes.groupby(["id", "class"])["class"].agg("count")
    cc_df = class_counts.to_frame(name="count").reset_index()
    cc_piv = cc_df.pivot(index="class", columns="id", values="count").fillna(0)

    return cc_piv.astype(int)


def score_clustering(exp_id: Union[str, int], dir = None) -> float:
    """
    Calculate clustering score for a given experiment.

    Args:
        exp_id: Experiment identifier.

    Returns:
        Accuracy score as a float.
    """
    return cluster_accuracy(optimize_clustering(f"exp_{exp_id}",dir = dir))


def plot_clustering(exp_id: Union[str, int], dir = None) -> None:
    """
    Plot clustering heatmap for a given experiment.

    Args:
        exp_id: Experiment identifier.

    Returns:
        None
    """
    sns.heatmap(
        optimize_clustering(exp_id,dir = dir),
        annot=True,
        fmt="d",
        cbar=False
    )

def get_stats(group_lines: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean and standard deviation of Euclidean distances between all unique line pairs.

    Args:
        group_lines: A NumPy array of shape (N, D) representing N lines in D-dimensional space.

    Returns:
        A tuple containing:
            - mean: Mean Euclidean distance between line pairs.
            - std: Standard deviation of those distances.
    """
    num_lines = group_lines.shape[0]
    eucl_dists = np.array([])
    for x in range(num_lines):
        for y in range(x + 1, num_lines):
            dist = np.linalg.norm(group_lines[x] - group_lines[y])
            eucl_dists = np.append(eucl_dists, dist)
    mean = np.mean(eucl_dists)
    std = np.std(eucl_dists)

    print('mean: ', mean)
    print('std: ', std)
    return mean, std


def rand_stats(all_lines: np.ndarray, n_rand: int = 100) -> np.ndarray:
    """
    Randomly select a subset of lines from the dataset.

    Args:
        all_lines: A NumPy array of all available lines.
        n_rand: Number of lines to randomly select.

    Returns:
        A NumPy array of randomly selected lines.
    """
    n = all_lines.shape[0]
    rand = np.random.randint(n, size=n_rand)
    r_lines = all_lines[rand]

    return r_lines


def get_discrete_lines(lines: np.ndarray, th_lines: List[Tuple[int, float]],
                       r: float, theta: float) -> np.ndarray:
    """
    Extract discrete lines from sinograms within a radius around theoretical positions.

    Args:
        lines: A NumPy array of all lines.
        th_lines: List of tuples (sinogram index, line position).
        r: Radius around each theoretical line to consider.
        theta: Angular step size in the sinogram.

    Returns:
        A NumPy array of selected lines within the specified range.
    """
    group_lines = np.array([])
    for cl in th_lines:
        s, x = cl
        lower_x = x - r
        upper_x = x + r
        lower_discrete = np.ceil(lower_x / theta) * theta
        if lower_discrete < upper_x and lower_discrete <= 360:
            upper_discrete = np.floor(upper_x / theta) * theta
            for l in range(int(lower_discrete // theta), int(upper_discrete // theta)):
                per_sino = 360 // theta
                ind = int(s * per_sino + l)
                group_lines = np.append(group_lines, lines[ind])
    return group_lines
