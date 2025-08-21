"""Metric determination script. 
"""
from itertools import permutations
import numpy as np
import pandas as pd
import seaborn as sns

from min_matrix import align_batches, make_line


def cluster_accuracy(cc_piv):

    """Determine cluster accuracy
    """
 
    cc_vals = cc_piv.values
    total= np.sum(cc_vals)

    if cc_piv.index[0]==-1:

        cc_vals = cc_vals[1:]
    perms = permutations(np.arange(len(cc_vals)))


    stored_count = 0

    for pp in perms:
        count = 0
        for i in range(len(pp)):
            count += cc_vals[i,pp[i]]
        if count > stored_count:

            stored_count = count
    
    

    correct = stored_count/float(total)*100

    return correct


def optimize_clustering(clic_exp):

    """Find optimal clustering  
    """
    data = np.load(f"{clic_exp}/cluster_matrix.npy")
    
    aligned_matrix = align_batches(data)
    all_classes = make_line(aligned_matrix)
    
    ids = [i.split('/')[-1][:4] for i in open(f"{clic_exp}/particle_ids.txt","r").readlines()]
    ids = ids[:len(all_classes)]
    fmt_classes = pd.DataFrame({"id":ids,"class":all_classes})
    
    class_counts = fmt_classes.groupby(["id", "class"])["class"].agg("count")
    
    cc_df = class_counts.to_frame(name="count")
    cc_df_flat = cc_df.reset_index()
    cc_piv = cc_df_flat.pivot(index = "class", columns = "id", values = "count").fillna(0)
    
    return cc_piv.astype(int)

def score_clustering(exp_id):

    """calculate clustering scores
    """
    return cluster_accuracy(
        optimize_clustering(f"exp_{exp_id}")
        )

def plot_clustering(exp_id):
    """plt output clusters against actual labels
    """

    sns.heatmap(
        optimize_clustering(exp_id),
        annot = True,
        fmt = "d",cbar = False
        )

