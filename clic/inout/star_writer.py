"""
STAR File Writer for CLIC

This module provides functions to create and update STAR files using the Gemmi CIF API.
It supports writing particle identifiers, iteration labels, and z-scores for classification tasks.

Functions:
    - create: Initialize a STAR file with particle IDs and image names.
    - update_data: Append iteration labels to an existing data table.
    - end_write: Finalize and write a STAR file with all iteration data and z-scores.
    - update: Legacy method for updating STAR files (slower, kept for compatibility).
"""

import time
from typing import List, Tuple 
import gemmi
import starfile
import pandas as pd
import numpy as np


def create(ids: List[str], optics, clic_dir: str) -> None:
    """
    Create initial STAR file with particle IDs and image names.

    Args:
        ids: List of image identifiers.
        clic_dir: Output directory for STAR file.

    Returns:
        Gemmi CIF Document object.
    """
    file_name = f"{clic_dir}/particles.star"
    if isinstance(ids, list):

        starfile.write({
            'particles':pd.DataFrame(
            {
                'rlnImageName': ids
            })}, file_name
        )
    else:
        starfile.write({'optics': optics,'particles':ids}, file_name)


def update_data(tags: List[str], labels: List[str], table: List[List[str]], it: int
                ) -> Tuple[List[str], List[List[str]]]:
    """
    Update tags and table with new iteration labels.

    Args:
        tags: List of STAR column tags.
        labels: Classification labels for current iteration.
        table: Data table to update.
        it: Iteration number.

    Returns:
        Updated (tags, table).
    """
    tags.append(f'_it{it}')
    for i, row in enumerate(table):
        row[it] = f'{labels[i]}'
    return tags, table


def end_write(tags: List[str], table: List[List[str]],
              z_score_list: List[str], clic_dir: str, ids: List[str]) -> None:
    """
    Write final STAR file with all iteration data and z-scores.

    Args:
        tags: Column tags.
        table: Data table with iteration labels.
        z_score_list: List of z-scores.
        clic_dir: Output directory.
        ids: List of image identifiers.
    """
    if isinstance(ids,pd.DataFrame):
        ids=ids['rlnImageName']


    table = np.insert(table,0,z_score_list,axis=0)
    cluster_table = pd.DataFrame(table , columns=tags)


    cluster_table['rlnID'] = np.insert(np.arange(len(ids)), 0, 0)

    cluster_table['rlnImageName'] = np.insert(ids,0,'z_score')
    starfile.write({'particles': cluster_table}, f'{clic_dir}/particles_CLIC.star')


