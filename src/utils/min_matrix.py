"""
Batch Alignment and Consensus Labelling.

This module provides utilities to:
    - Align class labels across multiple clustering batches
    - Determine consensus class assignments
    - Score alignment quality using ground truth

Inputs:
    - matrix.npy: 3D array of shape (Batches × Particles × Classes)
    - gt_ids_bin.npy: Ground truth binary labels

Outputs:
    - Aligned matrix
    - Consensus class assignments
    - Accuracy score

Dependencies:
    - numpy
    - itertools
    - clustering.score_bins
"""
from itertools import permutations
from typing import List, Tuple
import numpy as np

def score_align(batch: np.ndarray, aligned_matrix: np.ndarray, p: Tuple[int]) -> int:
    """
    Score how well a batch aligns with previously aligned batches using a permutation.

    Args:
        batch: One-hot encoded batch matrix of shape (n × d).
        aligned_matrix: Previously aligned batches of shape (b × n × d).
        p: Permutation of class indices.

    Returns:
        Alignment score as an integer.
    """
    score = 0
    for g_id, one_hot in enumerate(batch):
        if 1 in one_hot:  # ignore unclassified
            unshift_c = np.argmax(one_hot)
            shift_c = p[unshift_c]
            for layer in aligned_matrix:
                score += layer[g_id, shift_c]
    return score


def make_slice(lin_batch: List[int], batch_ids: List[int], bnd: Tuple[int, int, int]) -> np.ndarray:
    """
    Convert linear class assignments into one-hot encoded batch matrix.

    Args:
        lin_batch: List of class assignments.
        batch_ids: List of batch indices per particle.
        BAC: Tuple of (Batches, Particles, Classes).

    Returns:
        One-hot encoded batch matrix of shape (n × d).
    """
    _, n, d = bnd
    batch = np.zeros((n, d), dtype=int)
    for i, i_class in enumerate(lin_batch):
        g_id = batch_ids[i]
        if 0 <= i_class < d:
            batch[g_id, i_class] = 1
    return batch


def make_line(al_matrix: np.ndarray) -> List[int]:
    """
    Determine consensus class for each particle across aligned batches.

    Args:
        al_matrix: Aligned matrix of shape (b × n × d).

    Returns:
        List of consensus class assignments per particle.
    """
    _, n, d = al_matrix.shape
    lin_matrix = []
    for g_id in range(n):
        scores = [np.sum(al_matrix[:, g_id, i]) for i in range(d)]
        lin_matrix.append(-1 if max(scores) == 0 else int(np.argmax(scores)))
    return lin_matrix


def align_batches(matrix: np.ndarray) -> np.ndarray:
    """
    Align class labels across multiple batches using permutation consensus.

    Args:
        matrix: Input matrix of shape (b × n × d).

    Returns:
        Aligned matrix of shape (b × n × d).
    """
    _, n, d = matrix.shape
    aligned_matrix = np.array([matrix[0]], dtype=int)
    perm_list = list(permutations(range(d)))

    for batch in matrix[1:]:
        scores = [score_align(batch, aligned_matrix, p) for p in perm_list]
        opt_perm = perm_list[np.argmax(scores)]

        opt_batch = np.zeros((n, d), dtype=int)
        for i, _ in enumerate(n):
            one_hot = batch[i]
            if 1 in one_hot:
                unshift_c = np.argmax(one_hot)
                shift_c = opt_perm[unshift_c]
                opt_batch[i, shift_c] = 1

        aligned_matrix = np.concatenate((aligned_matrix, [opt_batch]))

    return aligned_matrix
