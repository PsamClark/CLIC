#!/usr/bin/env python
"""
Experiment Configuration and Evaluation Collation.

This module provides utilities for:
    - Generating unique experiment identifiers
    - Storing configuration metadata in JSON
    - Saving image and sinogram data to HDF5
    - Aggregating clustering scores across experiments
    - Flattening nested config dictionaries for tabular export

Outputs:
    - Configs stored in Configs/{exp_id}.json
    - Image data stored in {exp_id}/batch0_images.hdf5
    - Aggregated scores saved to {cwd.name}_cs.csv

Dependencies:
    - random
    - string
    - glob
    - pathlib
    - json
    - pandas
    - h5py
    - metrics.score_clustering
"""
from typing import Any, Dict, List, Tuple
import random
import string
from glob import glob
from pathlib import Path
import json
import panas as pd
import h5py
from pydantic import BaseModel
from analysis.metrics import score_clustering

from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    Str
)

class Config(BaseModel):

    dir: FilePath = Field(None,description="data_path")
    dsize: PositiveInt = Field(1000, description = "dataset size")
    bsize: PositiveInt = Field(None,"Batch size")
    dscale: PositiveFloat = Field(1,"downscaling")
    filter_method: str = Field("butter","bandpass filter method")
    lowpass: PositiveFloat = Field(5,"lowpass filter value in angstrom")
    highpass: PositiveFloat = Field(None,"highpass filter value in angstrom ")
    psize: PositiveFloat = Field(1, "pixel size")
    snr: PositiveFloat = Field(None, "snr ratio to add noise to the image")
    model: str = Field("UMAP","Model type")
    nlines: PositiveInt = Field(120, "number of sinogram lines")
    dims: PositiveInt = Field(3, "number of dimensins to reduce to")
    num_clusters: PositiveInt = Field(2,"number of clusters")

    
def random_string(length: int) -> str:
    """
    Generate a random alphanumeric string of specified length.

    Args:
        length: Desired length of the string.

    Returns:
        A random string containing letters and digits.
    """
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def store_config(args: Any, exp_id: str) -> None:
    """
    Store experiment configuration to a JSON file.

    Args:
        args: Parsed arguments or config object with required attributes.
        exp_id: Unique experiment identifier.

    Returns:
        None
    """
    Path("Configs").mkdir(exist_ok=True)
    config={}

    config['dir'] = args.data_set
    config['dsize'] = args.num
    config['data']['bsize'] = args.batch_size

    config['preprocess']['downscale'] = args.down_scale
    config['preprocess']['filter_method'] = args.filter_method
    config['preprocess']['lowpass'] = args.lowpass
    config['preprocess']['highpass'] = args.highpass
    config['preprocess']['snr'] = args.snr

    config['model']['model'] = args.model
    config['model']['nlines'] = args.nlines
    config['model']['dims'] = args.num_comps
    config['model']['clusters'] = args.num_clusters

    with open(f"Configs/{exp_id}.json", "w") as confile:
        json.dump(config, confile)

def load_config(fpath):

    try:
        with open(fpath, "r") as conffile:
            config = json.load(conffile)
    except: 
        raise ImportError("cofig_file not found")
    

def store_images(all_ims: Any, all_sinos: Any, all_ids: Any, exp_id: str) -> None:
    """
    Store image, sinogram, and ID data in an HDF5 file.

    Args:
        all_ims: Array of images.
        all_sinos: Array of sinograms.
        all_ids: Array of identifiers.
        exp_id: Unique experiment identifier.

    Returns:
        None
    """
    with h5py.File(f"{exp_id}/batch0_images.hdf5", "w") as imfile:
        imfile.create_dataset('batch0/images', data=all_ims)
        imfile.create_dataset('batch0/sinograms', data=all_sinos)
        imfile.create_dataset('batch0/ids', data=all_ids)


def collate_scores() -> None:
    """
    Aggregate clustering scores and configuration features across experiments.

    Returns:
        None
    """
    experiments = glob("Configs/*.json")
    out_list: List[List[Any]] = []

    for exp_conf in experiments:
        
        exp_id = Path(exp_conf).stem

        config = load_config(exp_conf)

        score = score_clustering(exp_id)
        if score is None:
            continue
        features, out_list = pop_features(config, score, out_list, exp_id)

    features.append("accuracy")
    features.insert(0, "exp_id")
    out_df = pd.DataFrame(data=out_list, columns=features)

    cwd = Path.cwd()
    out_df.to_csv(f"{cwd.name}_cs.csv", index=False)


def get_dict_entries(confdict: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
    """
    Recursively flatten a nested dictionary into key-value pairs.

    Args:
        confdict: Nested configuration dictionary.

    Returns:
        A tuple containing:
            - keys: Flattened keys with underscores.
            - values: Corresponding values.
    """
    keys: List[str] = []
    values: List[Any] = []

    for k, v in confdict.items():
        if isinstance(v, dict):
            keys_new, values_new = get_dict_entries(v)
            keys.extend([f"{k}_{i}" for i in keys_new])
            values.extend(values_new)
        else:
            keys.append(k)
            values.append(v)
    return keys, values


def pop_features(config: Dict[str, Any], score: float,
                 out_list: List[List[Any]], exp_id: str
                 ) -> Tuple[List[str], List[List[Any]]]:
    """
    Populate feature list with config values and clustering score.

    Args:
        config: Configuration dictionary.
        score: Clustering score for the experiment.
        out_list: Accumulated list of experiment results.
        exp_id: Unique experiment identifier.

    Returns:
        A tuple containing:
            - features: List of feature names.
            - out_list: Updated list of experiment results.
    """
    features, values = get_dict_entries(config)
    values.append(score)
    values.insert(0, exp_id)
    out_list.append(values)

    return features, out_list

