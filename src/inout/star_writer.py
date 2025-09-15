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


def create(ids: List[str], clic_dir: str) -> gemmi.cif.Document:
    """
    Create initial STAR file with particle IDs and image names.

    Args:
        ids: List of image identifiers.
        clic_dir: Output directory for STAR file.

    Returns:
        Gemmi CIF Document object.
    """
    out_doc = gemmi.cif.Document()
    block = out_doc.add_new_block('particles')
    tags = ['_id', '_rlnimagename']
    loop = block.init_loop('', tags)

    loop.add_row([f'-1\t', f'z_score\t'])  # z scores
    for x in range(1, len(ids)):
        loop.add_row([f'{x}\t', f'{ids[x]}\t'])

    out_doc.write_file(f'{clic_dir}/particles.star')


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
        row[it] = f'{labels[i]}\t'
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
    new_doc = gemmi.cif.Document()
    block = new_doc.add_new_block('particles')
    loop = block.init_loop('', tags)

    row1 = ['-', 'z_score']

    row1.extend(z_score_list)
    loop.add_row(row1)

    for x, ent in enumerate(ids):
        row = [f'{x}\t', f'{ent}\t'] 
        row.extend(table[x])       
        loop.add_row(row)

    new_doc.write_file(f'{clic_dir}/particles_CLIC.star')


def update(star_file: gemmi.cif.Document, labels: List[str], it: int,
           z_score: str, clic_dir: str) -> gemmi.cif.Document:
    """
    Legacy method to update STAR file with new iteration and z-score.

    Args:
        star_file: Existing STAR file as Gemmi Document.
        labels: Classification labels for current iteration.
        it: Iteration number.
        z_score: Z-score value.
        clic_dir: Output directory.

    Returns:
        Updated Gemmi Document.
    """
    block = star_file.find_block('particles')
    table = block.find_loop('_id').get_loop()
    tags = table.tags
    tags.append(f'_it{it}')

    new_doc = gemmi.cif.Document()
    block_temp = new_doc.add_new_block('particles')
    loop = block_temp.init_loop('', tags)

    # Add z-score row
    row = list(table[0])
    row.append(f'{z_score}')
    loop.add_row(row)

    tot_time = 0
    for i in range(1, len(table)):
        st_time = time.time()
        new_row = list(table[i])
        new_row.append(f'{labels[i]}\t')
        loop.add_row(new_row)
        tot_time += time.time() - st_time

    print(f"Update time: {tot_time:.4f} seconds")
    new_doc.write_file(f'{clic_dir}/particles_CLIC_old.star')
