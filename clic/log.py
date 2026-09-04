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
from typing import Any, Dict, List, Tuple, Optional
import random
import string
from glob import glob
from pathlib import Path, PurePath
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from .analysis.metrics import score_clustering

from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    ValidationError
)

class Config(BaseModel):

    dataset: PurePath = Field('',description="data_path")
    num: PositiveInt = Field(1000, description = "dataset size")
    batch_size: Optional[int] = Field(None,description="Batch size")
    downscale: PositiveFloat = Field(1,description="downscaling")
    tightmask: Optional[Any] = Field(None, description= "apply tightmask")
    filter_method: str = Field("butter",description="bandpass filter method")
    lowpass: Optional[int] = Field(None,description="lowpass filter value in angstrom")
    highpass: Optional[int] = Field(None,description="highpass filter value in angstrom ")
    pixel_size: PositiveFloat = Field(1, description= "pixel size")
    apply_ctf_correction: bool = Field(False, description="perform ctf correction")
    centre_particles: bool = Field(False, description="centre particles based on Class2D")
    snr: Optional[float] = Field(None, description="snr ratio to add noise to the image")
    model: str = Field("UMAP",description="Model type")
    cluster_method: str = Field("hdbscan",description="method of clustering when unknown clusters")

    lines: PositiveInt = Field(120,description="number of sinogram lines")
    comps: List[PositiveInt] = Field([3,2], description="number of dimensions to reduce to")
    nearest_neighbours: PositiveInt = Field(15, description="nearest neighbours (for UMAP and LLE)")
    min_distance: PositiveFloat = Field(0.15, description="minimum distance (for UMAP)")

    clusters: Optional[PositiveInt] = Field(2,description="number of clusters")
    gpu: bool = Field(False, description="use GPUs")
    save_model: bool = Field(False, description="save model")


def random_string(length: int) -> str:
    """
    Generate a random alphanumeric string of specified length.

    Args:
        length: Desired length of the string.

    Returns:
        A random string containing letters and digits.
    """
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def store_config(config: Any, exp_id: str) -> None:
    """
    Store experiment configuration to a JSON file.

    Args:
        args: Parsed arguments or config object with required attributes.
        exp_id: Unique experiment identifier.

    Returns:
        None
    """
    config_dict = config.model_dump()
    for key,val in config_dict.items():

        if isinstance(val, PurePath):
            config_dict[key] = str(val)

        if isinstance(val,list):
            config_dict[key] = ','.join([str(i) for i in val])
    
    Path("Configs").mkdir(exist_ok=True)

    with open(f"Configs/{exp_id}.json","w") as confile:
        json.dump(config_dict, confile, indent=4)


def load_config(fpath):

    try:
        with open(fpath, "r") as conffile:

            conf = json.load(conffile)

            conf["comps"] = [int(i) for i in str(conf["comps"]).split(",")]
            config = Config(**conf)

        if  len(str(config.dataset)) ==0:
            raise ValueError("dataset path not provided!")

        return config
    
    except ValidationError as e:
        raise ImportError(f"config_file is invalid: {e}")
    

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
    plt.imshow(all_ims[0])
    plt.savefig(f"{exp_id}/sample_image.png")
    plt.close()

    plt.imshow(all_sinos[0])
    plt.savefig(f"{exp_id}/sample_sino.png")
    plt.close()


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
            print("no score")
            continue
        features, out_list = pop_features(config, score, out_list, exp_id)

    features.append("accuracy")
    features.insert(0, "exp_id")
    out_df = pd.DataFrame(data=out_list, columns=features)

    cwd = Path.cwd()
    out_df.to_csv(f"{cwd.name}_cs.csv", index=False)

def collate_configs() -> None:
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

        conf_dict = config.model_dump()
        features = list(conf_dict.keys())
        values = list(conf_dict.values())
        values.insert(0, exp_id)
        features.insert(0, "exp_id")
        out_list.append(values)

    out_df = pd.DataFrame(data=out_list, columns=features)

    cwd = Path.cwd()
    out_df.to_csv(f"{cwd.name}_config_log.csv", index=False)

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

    conf_dict = config.model_dump()
    features = list(conf_dict.keys())
    values = list(conf_dict.values())
    values.append(score)
    values.insert(0, exp_id)
    out_list.append(values)

    return features, out_list

def ohk_to_label(matrix):

    return(np.argmax(matrix,axis=-1))