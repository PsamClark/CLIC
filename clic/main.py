#!/usr/bin/env python
"""
main.py

Main entry point for the CLIC 3D heterogeneity sorting algorithm.

This script orchestrates the pipeline for clustering 3D cryo-EM particle data using sinogram-based
dimensionality reduction and stepwise clustering. It supports batch processing, configuration via
command-line config, and outputs dendrograms and cluster assignments for each particle.

Key Components:
    - Argument parsing for configuration.
    - Batch processing of particles.
    - Sinogram generation and dimensionality reduction.
    - Stepwise clustering and dendrogram visualization.
    - Output of cluster assignments and results.

Usage:
    Run this script from the command line with the required config:
        python main.py -i <input_data> [other options]

Example:
    python main.py -i particles.star -n 1000 -b 750 -c 10 -m UMAP

Authors:
    Donovan Webb & Yuriy Chaban
"""

# Other dependencies
import os
import sys
import time

import numpy as np
import pandas as pd
import joblib
import click
import collections

from .inout.sinogram_input import sinogram_main
from .inout.sinogram_input import get_part_locs
from .engine.dim_red import fitmodel
from .engine.clustering import clustering_main, get_centroids
from .log import random_string, ohk_to_label
from .log import Config, store_config, store_images
from .inout.star_writer import create
from .utils.min_matrix import make_slice
from .utils.customs import WildCardType 

# To silence deprecation warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

@click.command(name="CLIC")
@click.option("--dataset","-ds",type=WildCardType(exists=True),
              help="path to data")
@click.option("-n", "--num", help="number of particles", default=1000, type=int)

@click.option("-b", "--batch_size", help="batch size", default=None, type=int)
@click.option("-d", "--downscale",
              help="Downscaling of image prior to making sinograms",
              type=int, default = 1)
@click.option("-hp", "--highpass",
              help = "value of image filter highpass resolution in angstroms",
              type=int)
@click.option("-lp", "--lowpass",
              help = "value of image filter lowpass resolution in angstroms",
            type=int)
@click.option("-ps", "--pixel_size",
              help = "pixel size in angstroms",
              default=1, type=float)
@click.option("-tm", "--tightmask",
              help = "apply tightmask before filtering",
              is_flag=True)
@click.option("-cp", "--centre-particles",
              help = "center particles if star file with devaitions exists",
              is_flag=True)
@click.option("-ctf", "--apply-ctf-correction",
              help = "apply CTF correction to image if CTF data is available",
              is_flag=True)
@click.option("-fm", "--filter_method",
              help = "image filter method",
              default="butter", type=str)
@click.option("-c", "--comps",
              help = "number of components in dimensional reduction",
              default=None, type=int)
@click.option("-m", "--model",
               help = "Dimension reduction technique",
               default='UMAP', type=str)
@click.option("-cm", "--cluster-method",
               help = "method for clustering when number of clusters unknown",
               default='hdbscan', type=str)
@click.option("-s", "--save_model",
              help = "Save model",
              is_flag=True)
@click.option("-l", "--lines",
              help = "Number of sinogram lines",
              default=120, type=int)
@click.option("-g", "--gpu",
              help = "Run on GPU",
              is_flag=True)
@click.option("-k", "--clusters", help = "Number of clusters", type=int)
@click.option("-r", "--snr",
              help = "Signal to noise ratio to be applied to projection",
              default=None, type=float)
def run(dataset,
        num,
        batch_size,
        downscale,
        highpass,
        lowpass,
        tightmask,
        pixel_size,
        filter_method,
        comps,
        model,
        apply_ctf_correction,
        centre_particles, 
        cluster_method,
        save_model,
        lines,
        gpu,
        clusters,
        snr, 
        rng = None):
    """ 
    Main function to run the CLIC clustering pipeline.
    Args:
        config: Parsed command-line config.
    """
    start = time.time()
     
    if rng is None:
        rng = np.random.RandomState()

    config = Config(**locals().copy())
    exp_id = random_string(6)
    store_config(config,exp_id)    
    exp_dir = f"exp_{exp_id}"
    print(f"Experiment running under id: {exp_id}")
    os.makedirs(exp_dir, exist_ok = True)
    part_locs,optics, n = get_part_locs(config, rng)
    batches = batching(n, config.batch_size, rng)

    if config.dataset.suffix in [".txt",".mrcs"]:
        part_ids = part_locs[1]
    
    else: 
        part_ids = part_locs['rlnImageName']

    with open(f"{exp_dir}/particle_ids.txt", "w") as fl:

        for line in part_ids:
                
            fl.write(f"{line}\n")

    all_name_ids = []
    b = 0

    batch_classes=[]
    batch_numbers=[]
    batch_times=[]
    for batch in batches:
        start_batch = time.time()
        batch_dir = f'{exp_dir}/batch_{b}'
        os.makedirs(batch_dir, exist_ok = True)
        print(f"### Running batch {b+1} of {len(batches)} with size {len(batch)} particles ###")

        all_sinos, all_ims, num, name_ids = sinogram_main(config, part_locs,
                                                                  batch, optics, rng)
        if b == 0:
            store_images(all_ims, all_sinos, name_ids, exp_dir)
        if isinstance(name_ids, pd.DataFrame):

            if len(all_name_ids) == 0:
                all_name_ids = name_ids
            else:
                all_name_ids = pd.concat((all_name_ids,name_ids))
            all_name_ids.drop_duplicates()
        else:
            for name_id in name_ids:
                if name_id not in all_name_ids:
                    all_name_ids.append(name_id)
        create(name_ids, optics, batch_dir)

        config.num = num  # Update with lowest num

        if config.comps is not None:
            lines_reddim, mod_fit, model = fitmodel(all_sinos, config.model, config.comps)

            if config.save_model:
                np.save(f"{batch_dir}/mod_fit.npy",mod_fit)
                np.save(f"{batch_dir}/lines_reddim.npy",get_centroids(lines_reddim,config.lines))
                joblib.dump(model,f"{batch_dir}/dimred.mod")

            
        else:
            lines_reddim = all_sinos
            config.comps = all_sinos.shape[-1]
        
        bclass,bnum = clustering_main(lines_reddim, config, batch_dir, name_ids)
        batch_classes.append(bclass)
        batch_numbers.append(bnum)

        batch_time=time.time() - start_batch
        batch_times.append(batch_time)
        print(f"   Batch time: {batch_time:.2f}s")
        b += 1

    if config.clusters is not None:
        matrix = np.zeros((len(batches), n, config.clusters))
    else:
        matrix = np.zeros((len(batches), n, max(batch_numbers)))

    for b,batch in enumerate(batches):
        matrix[b] = make_slice(batch_classes[b], batch, matrix.shape)
    print (f"Number of clusters detected: {matrix.shape[-1]}")
    labels = ohk_to_label(matrix)

    batch_labels = [str(i) for i in range(matrix.shape[0])]

    batch_df = pd.DataFrame(labels.T, columns=batch_labels)


    batch_df.to_csv(f"{exp_dir}/labels.csv", index=False)

    np.save(f"{exp_dir}/cluster_matrix.npy", matrix)
    total_time = time.time() - start
    print(f"### Total time: {total_time:.2f}s ###")
    with open(f"{exp_dir}/times.txt","w") as timefile:
        timefile.write(f"{total_time:.2f},{np.mean(batch_times):.2f}\n")
    


def batching(size, b_size,rng):
    """
    Defines batching of data indices for processing.

    Args:
        size (int): Total number of items to batch.
        b_size (int): Desired batch size.

    Returns:
        list: List of numpy arrays or ranges, each representing a batch of indices.
    """
    all_n = range(size)
    if b_size is None:
        return [all_n]
    elif b_size >= size:
        print("batchsize given larger than or equal to sample size.")
        print("defaulting to no batching.")
        return [all_n]

    size_half = int(np.floor(b_size/2))
    batch_dist = np.array(
        [np.concatenate(
            (rng.choice(x, size = size_half, replace = False).astype(int),
             np.array(range(x, x+size_half)))) for x in range(size_half*2, size, size_half)])
    
    batch_dist = np.concatenate(([np.arange(0, size_half*2)], batch_dist))
    if (size-size_half*2) % (size_half) != 0:
        
        max_n_arg = np.argwhere(batch_dist[-1] == size)[0][0].astype(int)#


        batch_endfill = np.array([i for i in range(b_size) if i not in batch_dist[-1, :max_n_arg]])
        rng.shuffle(batch_endfill)
        batch_dist[-1] = np.concatenate(
            (batch_dist[-1, :max_n_arg], batch_endfill[:(size_half*2 - max_n_arg)].astype(int)))
    return batch_dist
