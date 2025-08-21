#!/usr/bin/env python
'''
Main file for CLIC 3D heterogeneity sorting algorithm.  Config options
are passed by arguments and the job pipeline to be run can be seen at
the bottom of the file.

Results: A dendrogram of each batch is made.  Output of star files
with each particle assigned to a cluster.

Donovan Webb & Yuriy Chaban
'''
import argparse
import os
import sys
import random
import time

# Other dependencies
import matplotlib.pyplot as plt
import numpy as np

from sinogram_input import sinogram_main
from sinogram_input import get_part_locs
from dim_red import fitmodel
from clustering import clustering_main
from log import random_string
from log import store_config, store_images
import clustering
import star_writer
import min_matrix

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

TEXT = ''' image filter method '''
parser.add_argument("-fm", "--filter_method", help = TEXT, default="butter", type=str)

TEXT = ''' Number of components of dimensional reduction technique. This
requires some experimentation '''
parser.add_argument("-c", "--num_comps", help = TEXT, default=10, type=int)

TEXT = ''' Dimensional reduction technique.
options are: PCA, UMAP, TSNE, LLE, ISOMAP, MDS, TRIMAP.
Recommended: UMAP and PCA. '''
parser.add_argument("-m", "--model", help = TEXT, default='UMAP', type=str)

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

    """Define Batching 
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


if __name__ == '__main__':
    start = time.time()

    exp_id = random_string(6)
    EXP_DIR = f"exp_{exp_id}"
    os.makedirs(EXP_DIR, exist_ok = True)
    store_config(args,exp_id)
    part_locs, n = get_part_locs(args) 
    batches = batching(n, args.batch_size)

    with open(f"{EXP_DIR}/particle_ids.txt", "w") as fl:
        for line in part_locs:
            fl.write(f"{line}\n")

    all_name_ids = []
    B = 0
    matrix = np.zeros((len(batches), n, args.num_clusters))
    for batch in batches:
        start_batch = time.time()
        BATCH_DIR = f'{EXP_DIR}/batch_{B}'
        os.makedirs(BATCH_DIR, exist_ok = True)
        print(f"### Running batch {B+1} of {len(batches)} with size {len(batch)} particles ###")

        all_sinos, all_ims, num, name_ids = sinogram_main(args, part_locs, batch)
        if B == 0:
            store_images(all_ims, all_sinos, name_ids, EXP_DIR)
        for name_id in name_ids:
            if name_id not in all_name_ids:
                all_name_ids.append(name_id)
        star_file  = star_writer.create(name_ids, BATCH_DIR)

        args.num = num  # Update with lowest num
        lines_reddim, model = fitmodel(all_sinos, args.model, args.num_comps)


        batch_classes = clustering_main(lines_reddim, args, BATCH_DIR, name_ids)
        matrix[B] = min_matrix.make_slice(batch_classes, batch, matrix.shape)
        print(f"   Batch time: {time.time() - start_batch:.2f}s")
        B += 1

    np.save(f"{EXP_DIR}/cluster_matrix.npy", matrix)
    aligned_matrix = min_matrix.align_batches(matrix)
    all_classes = min_matrix.make_line(aligned_matrix)

    # Score classes in testing. input alternates between classes. i.e. classes  0101010101...
    BINARY_TEST = False
    if BINARY_TEST:
        ids_ints = clustering.ids_to_int(all_name_ids)
        gt_ids_bin = [x % args.num_clusters for x in ids_ints]
        np.save("gt_ids_bin.npy", gt_ids_bin)
        score = clustering.score_bins(gt_ids_bin, all_classes, args)
        print(f"Total_score: {score}")

    print(f"Total time: {time.time() - start:.2f}s")
    plt.show()
