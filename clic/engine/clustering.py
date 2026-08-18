"""
input: dimensionally reduced lines
output: stepwise clustering of sinograms

Initially each sinogram is one cluster, so we have clusters C1, C2, C3 ... CN,
where N is num of projections.

A score is made for each pair of clusters by finding euclidian distance between
lines contained within each group.

Lowest scoring cluster pair is merged and the new larger cluster is named after
the smaller cluster number of the merging clusters. i.e. if C4 and C7 merge
they will now all be contained in C4.

This iterates until all points are contained within one cluster - C1.

At each merging we get an output of which cluster each sinogram belongs to.
Large merges are points of interest and so this is saved at the end of the
program.
The clusters prior to a large merge should be analysed further.
A dendrogram is made to display clustering.
Dendrogram is cut to produce clusters.
"""
import os
import math
import collections
from itertools import permutations

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as eucl_dist
import matplotlib.pyplot as plt
from scipy import stats 
from scipy.cluster.hierarchy import  fcluster
from numba import cuda
from sklearn.cluster import DBSCAN, HDBSCAN,OPTICS

from clic.inout.star_writer import end_write, update_data
from clic.log import Config


def initial_dict(lines, num):
    """
    Create a dictionary of initial clusters, assigning each sinogram to its own cluster.

    Args:
        lines (np.ndarray): Array of lines (sinograms).
        num (int): Number of projections (sinograms).

    Returns:
        dict: Dictionary mapping cluster indices to arrays of lines.
    """
    per_sino = lines.shape[0] // num  # How many single lines per sino
    # Put each line in an initial cluster
    cluster_dict = {i: lines[per_sino*i:per_sino*(i+1)] for i in range(num)}
    return cluster_dict


def plot_hist(dists, name):
    """
    Optionally plot a histogram of distances for cluster pairs.

    Args:
        dists (np.ndarray): Array of distances.
        name (tuple): Tuple of cluster indices.
    """
    if name[0] < 5:
        if name[1] < 5:
            # Get histogram
            hist, bins = np.histogram(dists, bins=100)
            hist = hist[1:]
            bins = bins[1:]

            # Plot
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, hist, align='center', width=width)


def cap_cluster(cl):
    """
    Cap the cluster size to N particles (with 120 lines each) for speed and memory efficiency.

    Args:
        cl (np.ndarray): Cluster array.

    Returns:
        np.ndarray: Possibly reduced cluster array.
    """
    n = 400
    len_cl = np.shape(cl)[0]

    if len_cl >= n*120:  # For speed up and save mem.
        rand_selection = np.random.randint(0, len_cl, n*120)
        cl = cl[rand_selection]
    return cl


def find_score(cl_x, cl_y):
    """
    Compute the score between two clusters using a distance-based metric.

    Args:
        cl_x (np.ndarray): First cluster array.
        cl_y (np.ndarray): Second cluster array.
        name (tuple): Tuple of cluster indices.

    Returns:
        float: Score between the two clusters.
    """
    cl_x = cap_cluster(cl_x)
    cl_y = cap_cluster(cl_y)

    paired_dists = eucl_dist(cl_x, cl_y)
    paired_dists = paired_dists.ravel()

    dists = (1/(1+paired_dists))**30

    score = np.mean(dists)
    return score


def dist_hist(cl_x):
    """
    Plot a histogram of distances within a single cluster.

    Args:
        cl_x (np.ndarray): Cluster array.
    """
    plt.figure(176)
    plt.title(f'{np.shape(cl_x)}')
    paired_dists = eucl_dist(cl_x, cl_x)
    plt.hist(paired_dists.ravel(), bins='auto')
    plt.show()


def find_scoretable(cluster_dict, cluster_labels):

    """
    Computes a score table for all pairs of clusters.
    For each pair of clusters, this function calculates a score
    using the `find_score` function and stores it in a matrix.
    The diagonal elements (where a cluster is compared with itself)
    are set to 0.
    Args:
        cluster_dict (dict): A dictionary mapping cluster labels to
        their corresponding data points or cluster members.
        cluster_labels (list): A list of cluster labels indicating
        the order of clusters in the score table.
    Returns:
        np.ndarray: A 2D NumPy array (score table) where the entry
        at [i, j] contains the score between cluster i and cluster j.

    """
    num_clusters = len(cluster_labels)
    scoretable = np.zeros((num_clusters, num_clusters))
    for x in range(num_clusters):
        for y in range(num_clusters):
            if x == y:
                scoretable[x, y] = 0
                continue
            else:
                cl_x = cluster_dict[cluster_labels[x]]
                cl_y = cluster_dict[cluster_labels[y]]
                scoretable[x, y] = find_score(cl_x, cl_y)
    return scoretable


@cuda.jit
def find_sctbl_cuda(a,b, d):
    """
    Compute the score table using CUDA for acceleration.

    Args:
        a (np.ndarray): Output grid for scores.
        d (np.ndarray): All sinogram data.
    """
    i, j = cuda.grid(2)
    if i == j:
        a[i, j] = 0
    elif (i < a.shape[0]) and (j < a.shape[1]):
        # find cluster arrays
        cl_x = d[i]
        cl_y = d[j]
        tot_score = 0
        for iter1 in range(cl_x.shape[0]):
            l1 = cl_x[iter1]
            for iter2 in range(cl_y.shape[0]):
                l2 = cl_y[iter2]
                # find eucl dist
                dist = 0
                for x,l in enumerate(l1):
                    dist += (l - l2[x])**2
                dist = math.sqrt(dist)

                if b.ndim>1:
                    b[i,j,iter1,iter2] = dist
                tot_score += 1/dist
        a[i, j] = tot_score


def update_scoretable(scoretable, a, b, num_clusters):
    """
    Update the score table after merging two clusters.

    Args:
        scoretable (np.ndarray): Current score table.
        a (int): Index of first cluster.
        b (int): Index of second cluster.
        num_clusters (int): Number of clusters after merge.

    Returns:
        np.ndarray: Updated score table.
    """
    # delete paired
    if a > b:
        x = b
        y = a
    else:
        x = a
        y = b
    # clear rows in order to update
    scoretable = np.delete(scoretable, x, 0)
    scoretable = np.delete(scoretable, y-1, 0)
    scoretable = np.delete(scoretable, x, 1)
    scoretable = np.delete(scoretable, y-1, 1)
    scoretable_e = np.zeros((num_clusters, num_clusters))
    # fill top left with original scoretable missing new entry
    scoretable_e[:-1, :-1] = scoretable
    scoretable = scoretable_e
    return scoretable


def update_clusterlabels(cluster_labels, p0, p1):
    """
    Update the cluster labels after merging two clusters.

    Args:
        cluster_labels (list): Current cluster labels.
        p0 (int): Label of first cluster.
        p1 (int): Label of second cluster.

    Returns:
        list: Updated cluster labels.
    """
    if p0 == p1:
        print("ERROR, p0=p1")
    cluster_labels.remove(p0)
    cluster_labels.remove(p1)
    cluster_labels.append(p0)
    return cluster_labels


def update_scores(cluster_dict, p0, num_clusters, cluster_labels, scoretable):
    """
    Fill in new empty score table slots after merging clusters.

    Args:
        cluster_dict (dict): Dictionary of clusters.
        p0 (int): Label of merged cluster.
        num_clusters (int): Number of clusters.
        cluster_labels (list): List of cluster labels.
        scoretable (np.ndarray): Current score table.

    Returns:
        np.ndarray: Updated score table.
    """
    cl_y = cluster_dict[p0]  # find score with new merged group
    for x in range(num_clusters):
        if cluster_labels[x] == p0:
            continue
        else:
            cl_x = cluster_dict[cluster_labels[x]]
            score = find_score(cl_x, cl_y)
            scoretable[x, -1] = score
            scoretable[-1, x] = score
    return scoretable


def update_scores_quick(cluster_dict, pointsab, pointsp, scoretable):
    """
    Quickly update the score table after merging clusters, without recomputing all distances.

    Args:
        cluster_dict (dict): Dictionary of clusters.
        pointsab (tuple): Indices of merged clusters in score table.
        pointsp (tuple): Labels of merged clusters.
        scoretable (np.ndarray): Current score table.

    Returns:
        np.ndarray: Updated score table.
    """
    # add empty row
    len_sc = np.shape(scoretable)[0] + 1
    scoretable_e = np.zeros((len_sc, len_sc))
    # fill top left with original scoretable missing new entry
    scoretable_e[:-1, :-1] = scoretable
    scoretable = scoretable_e

    a, b = pointsab  # ind of score table max
    p0, p1 = pointsp # cl labels for merging groups
    cl_x = cluster_dict[p0]
    len_x = np.shape(cl_x)[0]
    cl_y = cluster_dict[p1]
    len_y = np.shape(cl_y)[0]
    for z in range(np.shape(scoretable)[0]):
        if z in (a, b):  # dont update own?
            continue
        else:
            scores_z = scoretable[z]
            score_x = scores_z[a]
            score_y = scores_z[b]
            score = (len_x * score_x + len_y * score_y) / (len_x + len_y)
            scoretable[z, -1] = score
            scoretable[-1, z] = score
    # delete paired
    if a > b:
        x = b
        y = a
    else:
        x = a
        y = b
    # clear rows in order to update
    scoretable = np.delete(scoretable, x, 0)
    scoretable = np.delete(scoretable, y-1, 0)
    scoretable = np.delete(scoretable, x, 1)
    scoretable = np.delete(scoretable, y-1, 1)
    return scoretable


def update_cl(cluster_dict, scoretable, cluster_labels, z, z_corr):
    """
    Merge the two clusters with the highest score and update all relevant structures.

    Args:
        cluster_dict (dict): Dictionary of clusters.
        scoretable (np.ndarray): Current score table.
        cluster_labels (list): List of cluster labels.
        z (list): Linkage matrix for dendrogram.
        z_corr (list): Correlation matrix for dendrogram.

    Returns:
        tuple: Updated (cluster_dict, scoretable, cluster_labels, paired, z, z_corr, score_inv)
    """
    
    score = np.max(scoretable)
    a, b = np.where(scoretable == score)
    if len(a) > 1:  # only take one entry
        a = a[0]
        b = b[0]


    paired = (cluster_labels[int(a)], cluster_labels[int(b)])
    p0 = np.min(paired)  # By convention new group name is lowest of two
    p1 = np.max(paired)

    cluster_labels = update_clusterlabels(cluster_labels, p0, p1)
    scoretable = update_scores_quick(
        cluster_dict, (a, b), paired,
        scoretable
        )

    # update cluster dict
    new_group = np.concatenate((cluster_dict[p0],
                                cluster_dict[p1]))
    cluster_dict[p0] = new_group



    # Dendrogram update
    z.append([z_corr[p0], z_corr[p1], 1/score, 0])
    z_corr[p0] = np.max(z_corr) + 1

    return cluster_dict, scoretable, cluster_labels, paired, z, z_corr, 1/score


def print_clusters_all(num, all_paired):
    """
    Generate final cluster assignments from all merge pairs.

    Args:
        num (int): Number of initial clusters.
        all_paired (list): List of all merged pairs.

    Returns:
        np.ndarray: Array of final cluster assignments.
    """
    clusters = np.array(list(range(num)))

    for paired in all_paired:
        s1, s2 = paired
        c1 = clusters[s1]
        c2 = clusters[s2]

        # update clusters by merging two
        if c1 < c2:
            clusters[clusters == c2] = c1
        elif c2 < c1:
            clusters[clusters == c1] = c2

    return clusters


def print_clusters(clusters, count, large_merges, paired, config, z_score):
    """
    Update clusters and count after a merge, and record large merges.

    Args:
        clusters (np.ndarray): Current cluster assignments.
        count (dict): Dictionary of cluster sizes.
        large_merges (list): List of large merges.
        paired (tuple): Pair of merged clusters.
        config (object): Configuration object.
        z_score (float): Z-score of the merge.

    Returns:
        tuple: Updated (cl, clusters, count, large_merges)
    """
    # Takes one paired at a time
    s1, s2 = paired
    c1 = clusters[s1]
    c2 = clusters[s2]

    count1 = count[c1]
    count2 = count[c2]
    merge_size = min(count1, count2)
    if merge_size > 0.1*config.num:
        large_merges.append([c1, c2, count1, count2, z_score])

    # update clusters by merging two
    if c1 < c2:
        clusters[clusters == c2] = c1
        count[c1] += count[c2]
    elif c2 < c1:
        clusters[clusters == c1] = c2
        count[c2] += count[c1]

    # Display cluster
    cl = clusters
    return cl, clusters, count, large_merges

def center_sctble(scoretable):
    """
    Normalize the score table by its mean.

    Args:
        scoretable (np.ndarray): Score table.
        config (object): Configuration object.

    Returns:
        np.ndarray: Normalized score table.
    """
    mean = np.mean(scoretable)
    scoretable = scoretable/mean
    return scoretable

def clustering_unknown(lines: np.ndarray,config:Config, centroids: bool = True):
    
    if config.cluster_method == "optics":
        model = OPTICS(min_cluster_size=int(0.05*config.num))

    elif config.cluster_method == "hdbscan":
        model = HDBSCAN(copy=True, min_cluster_size=int(0.05*config.num))

    if centroids:
        sinos = np.reshape(
            np.ascontiguousarray(lines),
            (config.num, config.lines, config.comps))
        sinos = np.mean(sinos,axis=1)
        model.fit(sinos)

        clusters = model.labels_
    else:
        model.fit(lines)
        line_clusters=np.reshape(np.ascontiguousarray(model.labels_),
                                 (config.num, config.lines))
        clusters = stats.mode(line_clusters, axis=1).mode
    num_clusters = len(list(set(clusters)))

    return clusters, num_clusters

def get_centroids(line_data: np.ndarray,nlines:int) -> np.ndarray:
    print(line_data.shape)
    sinodata = np.reshape(
                np.ascontiguousarray(line_data),
                (int(line_data.shape[0]/nlines), nlines, line_data.shape[-1]))
    
    return np.mean(sinodata,axis=1)


def clustering_main(lines, config, clic_dir, ids):
    """
    Main function for stepwise clustering of sinograms.

    Args:
        lines (np.ndarray): Array of dimensionally reduced lines.
        config (object): Configuration object.
        clic_dir (str): Output directory.
        ids (list): List of sinogram identifiers.

    Returns:
        np.ndarray or None: Final cluster assignments if num_clusters is set,
        else None.
    """
    if config.clusters is None:

        return clustering_unknown(lines, config,centroids=False)
    
    cl_labels = list(range(config.num))
    cl_dict = initial_dict(lines, config.num)
    if config.gpu:
        sinos = np.reshape(
            np.ascontiguousarray(lines),
            (config.num, config.lines, config.comps))
        scoretable = np.zeros((config.num, config.num), dtype=np.float32)
        disttable = np.array([])
        
        save_dists = False

        if save_dists:

            disttable = np.zeros((config.num, config.num, sinos.shape[1],sinos.shape[1]), dtype=np.float32)

        # data to device
        d_sinos = cuda.to_device(sinos)
        d_scoretable = cuda.to_device(scoretable)
        d_disttable = cuda.to_device(disttable)
        
        # Set up enough threads for kernel
        threadsperblock = (32, 32)
        blockspergrid_x = (config.num + threadsperblock[0]) // threadsperblock[0]
        blockspergrid_y = (config.num + threadsperblock[1]) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        find_sctbl_cuda[blockspergrid, threadsperblock](d_scoretable, d_disttable, d_sinos)
        scoretable = d_scoretable.copy_to_host()
        disttable = d_disttable.copy_to_host()

    else:
        scoretable = find_scoretable(cl_dict, cl_labels)  # old method

    if disttable is not None:
        np.save(f"{clic_dir}/disttable.npy",disttable)
    # Normalize scoretable
    scoretable = center_sctble(scoretable)
    # print(f"   sctable time: {(time.time() - timer_sc_tbl)}")
    all_paired = []
    z = []  # Linkage matrix for drawing dendrogram
    z_corr = list(range(config.num))
    tags = []
    z_score_list = []  # tags for star file
    table = np.ndarray((config.num, config.num -1), dtype=object)


    for i in range(config.num - 1):
        cl_dict, scoretable, cl_labels, paired, z, z_corr, z_score = update_cl(cl_dict, scoretable,
                                                                      cl_labels, z, z_corr)
        all_paired.append(paired)

        if i == 0:  # First pass
            clusters = np.array(list(range(config.num)))
            count = {x: 1 for x in range(config.num)}
            cl, clusters, count, large_merges = print_clusters(
                clusters, count, [], paired, config, z_score)
        else:
            cl, clusters, count, large_merges = print_clusters(
                clusters, count, large_merges, paired, config, z_score)

        z_score_list.append(f'{z_score}')
        
        tags, table = update_data(tags, cl, table, i)

    end_write(tags, table, z_score_list, clic_dir, ids)

    np.save(f"{clic_dir}/large_merges", np.asarray(large_merges))


    if config.clusters is not None:
        current_cl = -1
        t = 1
        runs = 0
        num_clusters = config.clusters
        while current_cl != num_clusters:
            if runs > 100:
                print("Cannot exclude anomalies, Look at clustering of dendrogram")
                auto_cl = fcluster(z, t=num_clusters, criterion='maxclust') - 1
                current_cl = num_clusters
                continue
            if current_cl > num_clusters or current_cl == 0:
                t = 1.1 * t
            elif current_cl < num_clusters:
                t = 0.95 * t
            auto_cl = fcluster(z, t=t, criterion='distance') - 1
            cl_freq = collections.Counter(auto_cl)
            current_cl = sum(1 if cl_freq[cl] > int(0.05*config.num) else 0 for cl in cl_freq)
            runs += 1

        return auto_cl,current_cl
    

def score_bins(gt, exp, config):
    """
    Score the clustering assignments against ground truth using all permutations.

    Args:
        gt (list): Ground truth cluster assignments.
        exp (list): Experimental cluster assignments.
        config (object): Configuration object.

    Returns:
        tuple: (max score fraction, fraction of unassigned)
    """
    num_clusters = config.clusters
    perm = permutations(range(num_clusters))
    assert(len(gt) == len(exp))
    scores = []
    for p in perm:
        score = 0
        score_no_class = 0
        for i,_ in enumerate(gt):
            unshift_c = gt[i]
            shift_c = p[unshift_c]
            if shift_c == exp[i]:
                score += 1
            if exp[i] == -1:
                score_no_class += 1
        scores.append(score)
    return (max(scores)/len(gt), score_no_class/len(gt))


def ids_to_int(ids):
    """
    Convert a list of file paths to integer IDs.

    Args:
        ids (list): List of file paths.

    Returns:
        list: List of integer IDs.
    """
    ints = [os.path.basename(x) for x in ids]
    ints = [os.path.splitext(x)[0] for x in ints]
    ints = [int(x) for x in ints]
    return ints
