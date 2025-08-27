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
import pandas as pd
import h5py

from analysis.metrics import score_clustering


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
    config: Dict[str, Dict[str, Any]] = {'model': {}, 'data': {}, 'preprocess': {}}

    config['data']['dir'] = args.data_set
    config['data']['dsize'] = args.num
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

        with open(exp_conf, "r") as conffile:
            config = json.load(conffile)

        score = score_clustering(exp_id)
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
    print(score)
    values.append(score)
    values.insert(0, exp_id)
    out_list.append(values)

    return features, out_list


if __name__ == "__main__":
    collate_scores()
