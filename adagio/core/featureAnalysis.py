#!/usr/bin/python
# ADAGIO Android Application Graph-based Classification
# featureAnalysis.py >> Analysis of features from SVM linear model
# Copyright (c) 2016 Hugo Gascon <hgascon@mail.de>


import os
import numpy as np
import networkx as nx
from random import shuffle

from adagio.common import ml


"""
Example:

import featureAnalysis as fa
w_binary = clf.best_estimator_.coef_[0]
w_agg = fa.aggregate_binary_svm_weights(w, 13)

"""


def print_largest_weights(w_agg, n):
    """ Print the largest weights
    """
    idx = w_agg.argsort()[::-1][:n]
    w_agg_highest = w_agg[idx]
    labels = [np.binary_repr(i, 15) for i in idx]
    print(zip(w_agg_highest, labels))


def aggregate_binary_svm_weights(w_binary, expansion_bits):
    """ Return the aggregated version of the SVM weight vector considering
    the binary representation length of the original non-binary feature.

    Args:
        w_binary: an array of SVM weights related to binary features.
        expansion_bits: the number of bits used to represent each feature in
        the original feature vector.

    Returns:
        w: the aggregated version of the SVM weight vector
    """
    feature_idx = len(w_binary) / expansion_bits  # should be a int
    w = np.array([sum(w_binary[expansion_bits * i:expansion_bits * (i + 1)])
                  for i in range(feature_idx)])
    return w


def compute_neighborhoods_per_weights(d, w, n_weights, n_files=300):
    """ Write report with info about highed ranked neighborhoods in a samples
    according to the weights learnt by the linear SVM model.

    Args:
        d: directory of the files to be processed
        w: linear SVM weights
        n_weights: number of weights to analyze
        n_files: number of files to process from directory d

    Returns:
        Outputs the file feature_analysis.txt
    """

    files = read_files(d, "fcgnx", n_files)
    sorted_weights_idx = w.argsort()[::-1]

    f_out = "feature_analysis.txt".format(n_weights)
    print("[*] Writing file {0}...".format(f_out))
    fd = open(f_out, 'wb')
    # fd.write("Total number of weights in SVM model: {0}\n".format(len(w)))
    # fd.write("Selected number of highest weights per sample: {0}\n".format(n_weights))

    for f in files:
        fn = os.path.join(d, f)
        neighborhoods, n_nodes = get_high_ranked_neighborhoods(fn, w,
                                                               sorted_weights_idx,
                                                               n_weights)
        try:
            if neighborhoods:
                fd.write("\n\n#########################################\n\n")
                fd.write(os.path.basename(f)+"\n\n")
                fd.write("nodes: {0}\n\n".format(n_nodes))
                fd.write("\n".join(neighborhoods))
        except:
            pass
    fd.close()
    print("[*] File written.")


def get_high_ranked_neighborhoods(fcgnx_file, w, sorted_weights_idx,
                                  show_small=False, weights=1):
    # g = FCGextractor.build_cfgnx(fcgnx_file)
    g = nx.read_gpickle(fcgnx_file)
    g_hash = ml.neighborhood_hash(g)

    neighborhoods = []
    remaining_weights = weights

    for idx in sorted_weights_idx:
        if remaining_weights > 0:
            label_bin = np.binary_repr(idx, 15)
            label = np.array([int(i) for i in label_bin])
            matching_neighborhoods = []
            for m, nh in g_hash.node.iteritems():
                if np.array_equal(nh["label"], label):
                    neighbors_l = g_hash.neighbors(m)
                    if neighbors_l:
                        neighbors = '\n'.join([str(i) for i in neighbors_l])
                        matching_neighborhoods.append("{0}\n{1}\n{2}\n".format(w[idx],
                                                      m, neighbors))
                    else:
                        if show_small:
                            matching_neighborhoods.append("{0}\n{1}\n".format(w[idx], m))

            if matching_neighborhoods:
                remaining_weights -= 1
                neighborhoods += matching_neighborhoods
        else:
            n_nodes = g_hash.number_of_nodes()
            del g
            del g_hash
            return neighborhoods, n_nodes


def add_weights_to_nodes(g, w, show_labels=True):
    g_hash = ml.neighborhood_hash(g)

    # initialize the weight for every node in g_hash
    for n, nh in g_hash.node.iteritems():
        idx = int("".join([str(i) for i in nh["label"]]), 2)
        w_nh = w[idx]
        g_hash.node[n]["label"] = w_nh

    # create a copy of the weighted graph
    g_hash_weighted = g_hash.copy()

    # aggregate the weights of each node with the
    # original weight of its caller
    for n, nh in g_hash.node.iteritems():
        for neighbor in g_hash.neighbors(n):
            g_hash_weighted.node[neighbor]["label"] += g_hash.node[n]["label"]

    # create array of the node weigths
    g_weights = []
    for n, nh in g_hash_weighted.node.iteritems():
            g_weights.append(nh["label"])

    # normalize weight between 0.5 and 1 to plot
    g_weights = np.array(g_weights)
    g_weights.sort()
    g_weights_norm = normalize_weights(g_weights)
    g_weights_norm = g_weights_norm[::-1]
    d_w_norm = dict(zip(g_weights, g_weights_norm))

    # add normalized weight as color to each node
    for n, nh in g_hash_weighted.node.iteritems():
            w = g_hash_weighted.node[n]["label"]
            g_hash_weighted.node[n]["style"] = "filled"
            g_hash_weighted.node[n]["fillcolor"] = "0.000 0.000 {0}".format(d_w_norm[w])

    # write function name in the label of the node or remove label
    if show_labels:
        for n, nh in g_hash_weighted.node.iteritems():
            node_text = (n[0].split("/")[-1] + n[1] + "\n" +
                         str(g_hash_weighted.node[n]["label"]))
            g_hash_weighted.node[n]["label"] = node_text
    else:
        for n, nh in g_hash_weighted.node.iteritems():
            g_hash_weighted.node[n]["label"] = ""

    return g_hash_weighted


def normalize_weights(a, imin=0.0, imax=1.0):
    dmin = a.min()
    dmax = a.max()
    return imin + (imax - imin) * (a - dmin) / (dmax - dmin)


def read_files(d, file_extension, max_files=0):
    files = []
    for fn in os.listdir(d):
        if fn.lower().endswith(file_extension):
            files.append(os.path.join(d, fn))
    shuffle(files)

    # if max_files is 0, return all the files in dir
    if max_files == 0:
        max_files = len(files)
    files = files[:max_files]

    return files
