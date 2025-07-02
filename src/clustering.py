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
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as eucl_dist
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster
import star_writer
import numba
from numba import cuda
import math

import time
import collections
from itertools import permutations


def initial_dict(lines, num):
    ''' make dictionary of initial classes i.e. which sinogram each line
    belongs to '''
    per_sino = lines.shape[0] // num  # How many single lines per sino
    # Put each line in an initial cluster
    cluster_dict = {i: lines[per_sino*i:per_sino*(i+1)] for i in range(num)}
    return cluster_dict


def plot_hist(dists, name):
    # optional
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
    ''' For speed up and to save memory. Caps cluster size to
    N particles (with 120 lines each) '''
    N = 400
    len_cl = np.shape(cl)[0]

    if len_cl >= N*120:  # For speed up and save mem.
        rand_selection = np.random.randint(0, len_cl, N*120)
        cl = cl[rand_selection]
    return cl


def find_score(clX, clY, name):
    ''' find score between two clusters '''
    clX = cap_cluster(clX)
    clY = cap_cluster(clY)
            
    paired_dists = eucl_dist(clX, clY)
    paired_dists = paired_dists.ravel()
    """ options :
    negtive power of x,
    dists = 1/(paired_dists)
    negative power of x+1 (this caps to 1),
    dists = 1/(1+paired_dists)
    top i values,
    #dists[np.where(dists <= 0.05)] = 0
    log curve,
    dists = e**(-a*paired_dists)
    perhaps gaussian?
    dists = e**(-a*paired_dists**2)
    """
    dists = (1/(1+paired_dists))**30
    #dists = np.e**(-200*paired_dists**2)
    #dists = (1/(paired_dists))
    # plot_hist(dists, name)
    score = np.mean(dists)
    return score


def dist_hist(clX):
    ''' find histogram of dists in one cluster '''
    plt.figure(176)
    plt.title(f'{np.shape(clX)}')
    paired_dists = eucl_dist(clX, clX)
    plt.hist(paired_dists.ravel(), bins='auto')
    plt.show()


def find_scoretable(cluster_dict, cluster_labels):
    ''' All scores between all pairs of clusters, looks like:

       | C1 | C2 | C3 | C4 |
    C1 | XX | .. | .. | .. |
    C2 | .. | XX | .. | .. |
    C3 | .. | .. | XX | .. |
    C4 | .. | .. | .. | XX |

    ''' 
    num_clusters = len(cluster_labels)
    scoretable = np.zeros((num_clusters, num_clusters))
    for X in range(num_clusters):
        for Y in range(num_clusters):
            if X == Y:
                scoretable[X, Y] = 0
                continue
            else:
                clX = cluster_dict[cluster_labels[X]]
                clY = cluster_dict[cluster_labels[Y]]
                scoretable[X, Y] = find_score(clX, clY, (X, Y))
    return scoretable


@cuda.jit
def find_sctbl_cuda(a, d):
    ''' 
    finding sc table using cuda
    a is output grid, d is all sino data
    '''
    i, j = cuda.grid(2)
    if i == j:
        a[i, j] = 0
    elif (i < a.shape[0]) and (j < a.shape[1]):
        # find cluster arrays
        clX = d[i]
        clY = d[j]
        tot_score = 0
        for iter1 in range(clX.shape[0]):
            l1 = clX[iter1]
            for iter2 in range(clY.shape[0]):
                l2 = clY[iter2]
                # find eucl dist
                dist = 0
                for x in range(len(l1)):
                    dist += (l1[x] - l2[x])**2
                dist = math.sqrt(dist)
                #tot_score += 1/(1+dist)**30
                tot_score += 1/dist
        a[i, j] = tot_score


def update_scoretable(scoretable, a, b, num_clusters):
    ''' remove paired scored and add new empty row for new scores '''
    # delete paired
    if a > b:  x = b;  y = a
    else:  x = a;  y = b
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
    ''' remove paired and add lowest as new cl label '''
    if p0 == p1:
        print("ERROR, p0=p1")
    cluster_labels.remove(p0)
    cluster_labels.remove(p1)
    cluster_labels.append(p0)
    return cluster_labels


def update_scores(cluster_dict, p0, num_clusters, cluster_labels, scoretable):
    ''' fill in new empty scoretable slots  '''
    clY = cluster_dict[p0]  # find score with new merged group
    for X in range(num_clusters):
        if cluster_labels[X] == p0:
            continue
        else:
            clX = cluster_dict[cluster_labels[X]]
            score = find_score(clX, clY, (X, p0))
            scoretable[X, -1] = score
            scoretable[-1, X] = score
    return scoretable


def update_scores_quick(cluster_dict, pointsab, pointsp, num_clusters, cluster_labels, scoretable):
    ''' fill in new empty scoretable slots without finding eucl dist again'''
    # add empty row
    len_sc = np.shape(scoretable)[0] + 1
    scoretable_e = np.zeros((len_sc, len_sc))
    # fill top left with original scoretable missing new entry
    scoretable_e[:-1, :-1] = scoretable
    scoretable = scoretable_e

    a, b = pointsab  # ind of score table max
    p0, p1 = pointsp # cl labels for merging groups
    clX = cluster_dict[p0]
    lenX = np.shape(clX)[0]
    clY = cluster_dict[p1]
    lenY = np.shape(clY)[0]
    for Z in range(np.shape(scoretable)[0]):
        if Z == a or Z == b:  # dont update own?
            continue
        else:
            scoresZ = scoretable[Z]
            scoreX = scoresZ[a]
            scoreY = scoresZ[b]
            score = (lenX * scoreX + lenY * scoreY) / (lenX + lenY)
            scoretable[Z, -1] = score
            scoretable[-1, Z] = score
    # delete paired
    if a > b:  x = b;  y = a
    else:  x = a;  y = b
    # clear rows in order to update
    scoretable = np.delete(scoretable, x, 0)
    scoretable = np.delete(scoretable, y-1, 0)
    scoretable = np.delete(scoretable, x, 1)
    scoretable = np.delete(scoretable, y-1, 1)
    return scoretable


def update_cl(cluster_dict, scoretable, cluster_labels, Z, Z_corr):
    score = np.max(scoretable)
    a, b = np.where(scoretable == score)
    if len(a) > 1:  # only take one entry
        a = a[0]
        b = b[0]

    paired = (cluster_labels[int(a)], cluster_labels[int(b)])
    p0 = np.min(paired)  # By convention new group name is lowest of two
    p1 = np.max(paired)

    cluster_labels = update_clusterlabels(cluster_labels, p0, p1)
    num_clusters = len(cluster_labels)
    scoretable = update_scores_quick(cluster_dict, (a, b), paired, num_clusters,
                                         cluster_labels, scoretable)

    # update cluster dict
    new_group = np.concatenate((cluster_dict[p0],
                                cluster_dict[p1]))
    cluster_dict[p0] = new_group
    del cluster_dict[p1]  # is this needed? save memory, but slow down running

    # OLD Method ##
    # scoretable = update_scoretable(scoretable, a, b, num_clusters)
    # scoretable = update_scores(cluster_dict, p0, num_clusters,
    #                            cluster_labels, scoretable)
    # ## 


    # Dendrogram update
    Z.append([Z_corr[p0], Z_corr[p1], 1/score, 0])
    Z_corr[p0] = np.max(Z_corr) + 1

    return cluster_dict, scoretable, cluster_labels, paired, Z, Z_corr, 1/score


def print_clusters_all(num, all_paired):
    # takes all paired values to make final clustering
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

def center_sctble(scoretable, config):
    mean = np.mean(scoretable)
    std = np.std(scoretable)
    scoretable = scoretable/mean
    return scoretable


def clustering_main(lines, config, clic_dir, ids):
    cl_labels = list(range(config.num))
    cl_dict = initial_dict(lines, config.num)
    timer_sc_tbl = time.time()
    if config.gpu:
        sinos = np.reshape(np.ascontiguousarray(lines), (config.num, config.nlines, config.num_comps))
        scoretable = np.zeros((config.num, config.num), dtype=np.float32)
        # data to device
        d_sinos = cuda.to_device(sinos)
        d_scoretable = cuda.to_device(scoretable)
        # Set up enough threads for kernel
        threadsperblock = (32, 32)
        blockspergrid_x = (config.num + threadsperblock[0]) // threadsperblock[0]
        blockspergrid_y = (config.num + threadsperblock[1]) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        find_sctbl_cuda[blockspergrid, threadsperblock](d_scoretable, d_sinos)
        scoretable = d_scoretable.copy_to_host()
    else:
        scoretable = find_scoretable(cl_dict, cl_labels)  # old method
    # Normalize scoretable
    scoretable = center_sctble(scoretable, config)
    # print(f"   sctable time: {(time.time() - timer_sc_tbl)}")
    all_paired = []
    Z = []  # Linkage matrix for drawing dendrogram
    Z_corr = list(range(config.num))
    tags = ['_id', '_path']  # tags for star file
    z_score_list = []  # tags for star file
    table = np.ndarray((config.num, config.num -1), dtype=object)


    timer_cl = time.time()
    for i in range(config.num - 1):
        cl_dict, scoretable, cl_labels, paired, Z, Z_corr, z_score = update_cl(cl_dict, scoretable,
                                                                      cl_labels, Z, Z_corr)
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
        tags, table = star_writer.update_data(tags, cl, table, i)

        # ### dist hist
        # N = 10
        # for cl_i in cl_dict:
        #     cl_ar = cl_dict[cl_i]
        #     if np.shape(cl_ar)[0] >= N * 120:
        #         dist_hist(cl_ar)
        # ###
    #print(f"clustering/N: {(time.time() - timer_cl)/config.num}")

    star_writer.end_write(tags, table, z_score_list, clic_dir, ids)

    np.save(f"{clic_dir}/large_merges", np.asarray(large_merges))

    np.save(f"{clic_dir}/dendrogram", Z)
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z)

    ax = plt.gca()

    do_bin_test = True
    if do_bin_test:
        # Add color to dendro labels
        xlbls = ax.get_xmajorticklabels()
        ### Check for binary simualted data ###
        ids_ints = [i for i in range(len(ids))]
        gt_ids_bin = [x % config.num_clusters for x in ids_ints]
        for lbl in xlbls:
            if gt_ids_bin[int(lbl.get_text())] == 0:
                lbl.set_color('r')
            else:
                lbl.set_color('b')

        # Perform cut on binary
        # merge_score = [min(x[1]) for x in large_merges]  # size of merges
        # bin_merge_ind = merge_score.index(max(merge_score)) # largest merge
        # bin_merge = large_merges[bin_merge_ind]
        # z_cut = bin_merge[2]*0.999  # must go below last merge
        # import bin_test
        # exp_ids_bin = bin_test.cut(table, z_score_list, z_cut)
        # score = score_bins(gt_ids_bin, exp_ids_bin)
        #auto_cl = fcluster(Z, t=num_clusters, criterion='maxclust') - 1
    if config.num_clusters != -1:
        current_cl = -1
        t = 1
        runs = 0
        num_clusters = config.num_clusters
        while current_cl != num_clusters:
            if runs > 100:
                print("Cannot exclude anomalies, Look at clustering of dendrogram")
                auto_cl = fcluster(Z, t=num_clusters, criterion='maxclust') - 1
                current_cl = num_clusters
                continue
            elif current_cl > num_clusters or current_cl == 0:
                t = 1.1 * t
            elif current_cl < num_clusters:
                t = 0.95 * t
            auto_cl = fcluster(Z, t=t, criterion='distance') - 1
            cl_freq = collections.Counter(auto_cl)
            current_cl = sum([1 if cl_freq[cl] > int(0.05*config.num) else 0 for cl in cl_freq])
            runs += 1

    if do_bin_test:
        score = score_bins(gt_ids_bin, auto_cl, config)
        print(f"   Batch score: {score}")

    plt.savefig(f"{clic_dir}/dendrogram.pdf")
    if config.num_clusters != -1:
        return auto_cl




def score_bins(gt, exp, config):
    num_clusters = config.num_clusters
    perm = permutations(range(num_clusters))
    assert(len(gt) == len(exp))
    scores = []
    for p in perm:
        score = 0
        score_no_class = 0
        for i in range(len(gt)):
            unshift_c = gt[i]
            shift_c = p[unshift_c]
            if shift_c == exp[i]:
                score += 1
            if exp[i] == -1:
                score_no_class += 1
        scores.append(score)
    return (max(scores)/len(gt), score_no_class/len(gt))


def ids_to_int(ids):
    import os
    ints = [os.path.basename(x) for x in ids]
    ints = [os.path.splitext(x)[0] for x in ints]
    ints = [int(x) for x in ints]
    return ints
