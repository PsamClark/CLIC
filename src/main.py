#!/usr/bin/env python
"""
main.py

Main entry point for the CLIC 3D heterogeneity sorting algorithm.

This script orchestrates the pipeline for clustering 3D cryo-EM particle data using sinogram-based
dimensionality reduction and stepwise clustering. It supports batch processing, configuration via
command-line arguments, and outputs dendrograms and cluster assignments for each particle.

Key Components:
    - Argument parsing for configuration.
    - Batch processing of particles.
    - Sinogram generation and dimensionality reduction.
    - Stepwise clustering and dendrogram visualization.
    - Output of cluster assignments and results.

Usage:
    Run this script from the command line with the required arguments:
        python main.py -i <input_data> [other options]

Example:
    python main.py -i particles.star -n 1000 -b 750 -c 10 -m UMAP

Authors:
    Donovan Webb & Yuriy Chaban
"""

# Other dependencies
import argparse
import os
import sys
import random
import time

import numpy as np
import joblib

from inout.sinogram_input import sinogram_main
from inout.sinogram_input import get_part_locs
from engine.dim_red import fitmodel
from engine.clustering import clustering_main
from log import random_string
from log import store_config, store_images
from inout.star_writer import create
from utils.min_matrix import make_slice

# To silence deprecation warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()

TEXT = ''' Dataset to be considered for clustering. Input path to mrcs
stack, to individual mrc particles, or to particle starfile with "/PATH/TO/PARTS/*.mrc"
(Notes: 1. Don't forget "",
        2. if star file: run from relion home dir
        3. expects *.mrc or path/to/file.mrcs or path/to/file.star only.
'''
parser.add_argument("-i", "--data_set", help=TEXT, required=True, type=str)

TEXT = ''' Number of projections to consider total. Defaults to 1000 '''
parser.add_argument("-n", "--num", help=TEXT, default=1000, type=int)

TEXT = ''' Batchsize - runs overlapping batches of provided size. This
speeds up process and requires less memory. Recommended batch size is
750 < b < 2000'''
parser.add_argument("-b", "--batch_size", help=TEXT, default=-1, type=int)

TEXT = ''' Downscaling of image prior to making sinograms '''
parser.add_argument("-d", "--down_scale", help=TEXT, type=int, default = 1)

TEXT = ''' value of image filter highpass resolution '''
parser.add_argument("-hp", "--highpass", help = TEXT, type=int)

TEXT = ''' value of image filter lowpass resolution '''
parser.add_argument("-lp", "--lowpass", help = TEXT, default=5, type=int)

TEXT = '''apply tightmask before filtering'''
parser.add_argument("-tm", "--tightmask", help = TEXT, default=False, action='store_true')

TEXT = ''' image filter method '''
parser.add_argument("-fm", "--filter_method", help = TEXT, default="butter", type=str)

TEXT = ''' Number of components of dimensional reduction technique. This
requires some experimentation '''
parser.add_argument("-c", "--num_comps", help = TEXT, default=None, type=int)

TEXT = ''' Dimensional reduction technique.
options are: PCA, UMAP, TSNE, LLE, ISOMAP, MDS, TRIMAP.
Recommended: UMAP and PCA. '''
parser.add_argument("-m", "--model", help = TEXT, default='UMAP', type=str)

TEXT = ''' Save Model'''
parser.add_argument("-s", "--save_model", help = TEXT, default=False, action='store_true')

TEXT = ''' Number of lines in one sinogram (shouldn't need to change
recommended=120)'''
parser.add_argument("-l", "--nlines", help = TEXT, default=120, type=int)

TEXT = ''' Run on gpu with CUDA '''
parser.add_argument("-g", "--gpu", help = TEXT, default=False, action='store_true')

TEXT = ''' Number of clusters '''
parser.add_argument("-k", "--num_clusters", help = TEXT, default=2, type=int)

TEXT = ''' For testing: Signal to noise ratio to be applied to projection
before making sinograms '''
parser.add_argument("-r", "--snr", help = TEXT, default=-1, type=float)

args = parser.parse_args()


def batching(size, b_size):
    """
    Defines batching of data indices for processing.

    Args:
        size (int): Total number of items to batch.
        b_size (int): Desired batch size.

    Returns:
        list: List of numpy arrays or ranges, each representing a batch of indices.
    """
    all_n = range(size)
    if b_size >= size or b_size == -1:
        return [all_n]
    size_half = int(np.floor(b_size/2))
    batch_dist = np.array(
        [np.concatenate(
            (random.sample(range(0, x), size_half),
             np.array(range(x, x+size_half)))) for x in range(size_half*2, size, size_half)])
    batch_dist = np.concatenate(([range(0, size_half*2)], batch_dist))
    if size % (size_half*2) != 0:
        max_n_arg = int(np.argwhere(batch_dist[-1] == size))
        batch_dist[-1] = np.concatenate(
            (batch_dist[-1, :max_n_arg], random.sample(range(0, b_size),
                                                       size_half*2 - max_n_arg)))
    return batch_dist

def main(arguments):
    """ 
    Main function to run the CLIC clustering pipeline.
    Args:
        arguments: Parsed command-line arguments.
    """
    start = time.time()

    exp_id = random_string(6)
    exp_dir = f"exp_{exp_id}"
    os.makedirs(exp_dir, exist_ok = True)
    store_config(arguments,exp_id)
    part_locs, n = get_part_locs(arguments)
    batches = batching(n, arguments.batch_size)

    with open(f"{exp_dir}/particle_ids.txt", "w") as fl:
        for line in part_locs:
            fl.write(f"{line}\n")

    all_name_ids = []
    b = 0
    matrix = np.zeros((len(batches), n, arguments.num_clusters))
    for batch in batches:
        start_batch = time.time()
        batch_dir = f'{exp_dir}/batch_{b}'
        os.makedirs(batch_dir, exist_ok = True)
        print(f"### Running batch {b+1} of {len(batches)} with size {len(batch)} particles ###")

        all_sinos, all_ims, num, name_ids = sinogram_main(arguments, part_locs, batch)
        if b == 0:
            store_images(all_ims, all_sinos, name_ids, exp_dir)
        for name_id in name_ids:
            if name_id not in all_name_ids:
                all_name_ids.append(name_id)
        create(name_ids, batch_dir)

        arguments.num = num  # Update with lowest num

        if args.num_comps is not None:
            lines_reddim, mod_fit, model = fitmodel(all_sinos, arguments.model, arguments.num_comps)

            if args.save_model:
                np.save(f"{batch_dir}/mod_fit.npy",mod_fit)
                np.save(f"{batch_dir}/lines_reddim.npy",lines_reddim)
                joblib.dump(model,f"{batch_dir}/dimred.mod")
                

        else:
            lines_reddim = all_sinos
            arguments.num_comps = all_sinos.shape[-1]
        

        batch_classes = clustering_main(lines_reddim, arguments, batch_dir, name_ids)
        matrix[b] = make_slice(batch_classes, batch, matrix.shape)
        print(f"   Batch time: {time.time() - start_batch:.2f}s")
        b += 1

    np.save(f"{exp_dir}/cluster_matrix.npy", matrix)
    print(f"### Total time: {time.time() - start:.2f}s ###")


if __name__ == '__main__':
    main(args)
