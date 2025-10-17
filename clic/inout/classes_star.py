"""
classes_star.py

This script processes CLIC and Relion STAR files to assign particles to classes
based on z-score thresholds.
It reads classification results from a CLIC STAR file, matches particle image names
to those in a Relion STAR file,
and generates new STAR files for each class containing the corresponding particles.

Key Components:
    - from_z: Determines the class index based on a z-score cutoff.
    - new_star: Writes a new STAR file for a given class and its members.
    - Main execution: Parses arguments, reads input files,
    performs class assignment, and writes output STAR files.

Dependencies:
    - gemmi
    - argparse

Usage:
    Run this script from the command line with the required arguments:
        python classes_star.py -i <clic_input.star> -r <relion_input.star> -c <z_cut>

Example:
    python classes_star.py -i particles_CLIC.star -r particles_relion.star -c 2.5
"""
import argparse
import gemmi

from clic.utils.cutter import cut

def new_star(class_n, members, relstar, tbl ):
    """
    Writes a new STAR file for a given class and its members.

    Args:
        class_n (int): Class number.
        members (list of int): List of indices for particles in this class.
        relstar (gemmi.cif.Document): The Relion STAR file object.
        table (list): Table of particle data rows.

    Returns:
        None
    """
    blck = relstar.find_block('particles')
    lp = blck.find_loop_item('_rlnimagename').loop
    tgs = lp.tags
    lp = blck.init_loop('', tgs)
    for m in members:
        lp.add_row(tbl[m])

    relstar.write_file(f'particles_CLIC_{class_n}.star')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    t = ''' CLIC star file '''
    parser.add_argument("-i", "--clic_input", help=t, required=True, type=str)
    t = ''' Relion star file '''
    parser.add_argument("-r", "--relion_input", help=t, required=True, type=str)
    t = ''' z score to cut at '''
    parser.add_argument("-c", "--cut", help="level to cut at", required=True, type=float)

    args = parser.parse_args()

    clic_file = args.clic_input
    relion_file = args.relion_input
    z_cut = args.cut


    classes,im_names,vals = cut(clic_file, z_cut)
    # dict of classes and members (rln index)
    cl_dict = {i : [] for i in set(classes)}


    relion_star = gemmi.cif.read_file(relion_file)
    relion_block = relion_star.find_block('particles')
    all_parts = list(relion_block.find_values('_rlnimagename'))[0:]


    for x,im in enumerate(im_names):
        im_class = classes[x]
        rln_ind = all_parts.index(im)
        mems = cl_dict[im_class]
        mems.append(rln_ind)
        cl_dict[im_class] = mems

    block = relion_star.find_block('particles')
    loop = block.find_loop_item('_rlnimagename').loop
    tags = loop.tags
    table = [list(x) for x in block.find(tags)]
    for x in set(classes):
        mems = cl_dict[x]
        new_star(x, mems, relion_star, table)
