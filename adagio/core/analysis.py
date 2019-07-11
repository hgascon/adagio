#!/usr/bin/python
# ADAGIO Android Application Graph-based Classification
# analysis.py >> Load dataset, train, predict and evaluate  
# Copyright (c) 2015 Hugo Gascon <hgascon@mail.de>

from adagio.common import ml
from adagio.common import eval
import adagio.core.instructionSet as instructionSet

import os
import time
import collections
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import GridSearchCV


class Analysis:
    """ A class to run a classification experiment """

    def __init__(self, dirs, labels, split, max_files=0, max_node_size=0, 
                 precomputed_matrix="", y="", fnames=""):
        """ 
        The Analysis class allows to load sets of pickled graoh objects
        from different directories where the objects in each directory
        belong to different classes. It also provide the methods to run
        different types of classification experiments by training and 
        testing a linear classifier on the feature vectors generated
        from the different graph objects.

        :dirs: A list with directories including types of files for
            classification e.g. <[MALWARE_DIR, CLEAN_DIR]> or just 
            directories with samples from different malware families
        :labels: The labels assigned to samples in each directory.
            For example a number or a string.
        :split: The percentage of samples used for training (value
            between 0 and 1)
        :precomputed_matrix: name of file if a data or kernel matrix
            has already been computed.
        :y: If precomputed_matrix is True, a pickled and gzipped list
            of labels must be provided.
        :returns: an Analysis object with the dataset as a set of
            properties and several functions to train, test, evaluate
            or run a learning experiment iteratively
        """

        self.split = split
        self.X = []
        self.Y = np.array([])
        self.fnames = []
        self.X_train = [] 
        self.X_test = []
        self.Y_train = []
        self.Y_test = []
        self.clf = ""
        self.out = []
        # self.roc = 0
        self.rocs = []
        self.auc = 0
        self.b = 0
        self.feature_vector_times = []
        self.label_dist = np.zeros(2**15)
        self.sample_sizes = []
        self.neighborhood_sizes = []
        self.class_dist = np.zeros(15)
        self.predictions = []
        self.true_labels = []

        if precomputed_matrix:
            # Load the y labels and file names from zip pickle objects.
            print("Loading matrix...")
            self.X = np.savez_compressed(precomputed_matrix)
            print("[*] matrix loaded")
            self.Y = np.load(y)
            print("[*] labels loaded")
            self.fnames = np.load(fnames)
            print("[*] file names loaded")

        else:
            # loop over dirs
            for d in zip(dirs, labels):
                files = self.read_files(d[0], "fcg", max_files)
                print("Loading samples in dir {0} with label {1}".format(d[0],
                                                                         d[1]))

                # load labels and feature vectors
                for f in tqdm(files):
                    try: 
                        g = np.load(f)
                        size = g.number_of_nodes()
                        if size < max_node_size or max_node_size == 0:
                            if size > 0:
                                t0 = time.time()
                                x_i = self.compute_label_histogram(g)
                                # save feature vector computing time for
                                # performance evaluation
                                self.feature_vector_times.append(time.time() -
                                                                 t0)
                                # save distribution of generated labels
                                self.label_dist = np.sum([self.label_dist,
                                                          x_i], axis=0)
                                # save sizes of the sample for further analysis
                                # of the dataset properties
                                self.sample_sizes.append(size)
                                self.neighborhood_sizes += ml.neighborhood_sizes(g)
                                for n, l in g.node.iteritems():
                                    self.class_dist = np.sum([self.class_dist,
                                                              l["label"]], axis=0)
                                # delete nx object to free memory
                                del g
                                self.X.append(x_i)
                                self.Y = np.append(self.Y, [int(d[1])])
                                self.fnames.append(f)
                    except Exception as e:
                        print(e)
                        print("err: {0}".format(f))
                        pass

            # convert feature vectors to its binary representation
            # and make the data matrix sparse
            print("[*] Stacking feature vectors...")
            self.X = np.array(self.X, dtype=np.int16)
            print("[*] Converting features vectors to binary...")
            self.X, self.b = ml.make_binary(self.X) 

    ################################
    # Data Preprocessing functions #
    ################################

    def read_files(self, d, file_extension, max_files=0):
        """ Return a random list of N files with a certain extension in dir d
        
        Args:
            d: directory to read files from
            file_extension: consider files only with this extension
            max_files: max number of files to return. If 0, return all files.

        Returns:
            A list of max_files random files with extension file_extension
            from directory d.
        """

        files = []
        for f in os.listdir(d):
            if f.lower().endswith(file_extension):
                files.append(os.path.join(d, f))
        shuffle(files)

        # if max_files is 0, return all the files in dir
        if max_files == 0:
            max_files = len(files)
        files = files[:max_files]
        
        return files

    def compute_label_histogram(self, g):
        """ Compute the neighborhood hash of a graph g and return
            the histogram of the hashed labels.
        """

        g_hash = ml.neighborhood_hash(g)
        g_x = ml.label_histogram(g_hash)
        return g_x

    def remove_unpopulated_classes(self, min_family_size=20):
        """ Remove classes with less than a minimum number of samples.
        """
        c = collections.Counter(self.Y)
        unpopulated_families = [f for f, n in c.iteritems() if n < min_family_size]
        index_bool = np.in1d(self.Y.ravel(),
                             unpopulated_families).reshape(self.Y.shape)
        self.Y = self.Y[~index_bool]
        index_num = np.arange(self.X.shape[0])[~index_bool]
        self.X = self.X[index_num, :]

    def randomize_dataset(self):
        """ Randomly split the dataset in training and testing sets
        """

        n = len(self.Y)
        train_size = int(self.split * n)
        index = range(n)
        shuffle(index)
        train_index = sorted(index[:train_size])
        test_index = sorted(index[train_size:])

        self.X_train = self.X[train_index, :]
        self.X_test = self.X[test_index, :]
        self.Y_train = self.Y[train_index]
        self.Y_test = self.Y[test_index]

    def randomize_dataset_open_world(self):
        """ Split the dataset in training and testing sets where
            classes used for training are not present in testing set.
        """
        classes = list(set(self.Y))
        shuffle(classes)
        train_classes = classes[:int(self.split * len(classes))]
        test_classes = classes[int(self.split * len(classes)):]

        train_idx_bool = np.in1d(self.Y.ravel(),
                                 train_classes).reshape(self.Y.shape)
        test_idx_bool = np.in1d(self.Y.ravel(),
                                test_classes).reshape(self.Y.shape)

        self.Y_train = self.Y[train_idx_bool]
        self.Y_test = self.Y[test_idx_bool]

        idx_num = np.arange(self.X.shape[0])
        self.X_train = self.X[idx_num[train_idx_bool], :]
        self.X_test = self.X[idx_num[test_idx_bool], :]

    def randomize_dataset_closed_world(self):
        """ Randomly split the dataset in training and testing sets
        """
        n = len(self.Y)
        train_size = int(self.split * n)
        index = range(n)
        shuffle(index)
        train_index = sorted(index[:train_size])
        test_index = sorted(index[train_size:])

        self.X_train = self.X[train_index, :]
        self.X_test = self.X[test_index, :]
        self.Y_train = self.Y[train_index]
        self.Y_test = self.Y[test_index]

    def save_data(self):
        """ Store npz objects for the data matrix, the labels and
            the name of the original samples so that they can be used
            in a new experiment without the need to extract all
            features again
        """
        print("[*] Saving labels, data matrix and file names...")
        np.savez_compressed("X.npz", self.X)
        np.savez_compressed("Y.npz", self.Y)
        np.savez_compressed("fnames.npz", self.fnames)

    ###################################
    # Learning & Evaluation functions #
    ###################################

    def run_linear_experiment(self, rocs_filename, iterations=10):
        """
        Run a classification experiment by running several iterations.
        In each iteration data is randomized, a linear svm classifier
        is trained and evaluated using cross-validation over a the 
        cost parameter in the range np.logspace(-3, 3, 7). The best
        classifier is used for testing and a ROC curve is computed
        and saved as property and locally.

        :param rocs_filename: the file to save all rocs computed
        :param iterations: number of runs (training/testing)
        """
        for i in range(iterations):
            print("[*] Iteration {0}".format(i))
            print("[*] Randomizing dataset...")
            self.randomize_dataset()
            clf = GridSearchCV(svm.LinearSVC(), {'C': np.logspace(-3, 3, 7)})
            print("[*] Training...")
            clf.fit(self.X_train, self.Y_train)
            out = clf.best_estimator_.decision_function(self.X_test)
            print("[*] Testing...")
            roc = eval.compute_roc(np.float32(out.flatten()),
                                   np.float32(self.Y_test))
            self.rocs.append(roc)
            print("[*] ROC saved.")
        np.savez_compressed(rocs_filename, self.rocs)

    def run_linear_closed_experiment(self, iterations=10, save=False):
        """
        Train a classifier on test data, obtain the best combination of
        parameters through a grid search cross-validation and test the
        classifier using a closed-world split of the dataset. The results
        from the number of iterations are saved as npz files.

        :param iterations: number of runs (training/testing)
        :save: save predictions and labels if True
        """
        self.true_labels = np.array([])
        self.predictions = np.array([])
        for i in range(iterations):
            self.randomize_dataset_closed_world()
            clf = GridSearchCV(svm.LinearSVC(), {'C': np.logspace(-3, 3, 7)})
            clf.fit(self.X_train, self.Y_train)
            out = clf.best_estimator_.predict(self.X_test)
            self.predictions = np.append(self.predictions, out)
            self.true_labels = np.append(self.true_labels, self.Y_test)

        if save:
            np.savez_compressed("mca_predictions_closed.npz", self.predictions)
            np.savez_compressed("mca_true_labels_closed.npz", self.true_labels)

    def run_linear_open_experiment(self, iterations=10, save=False):
        """
        Train a classifier on test data, obtain the best combination of
        parameters through a grid search cross-validation and test the
        classifier using a open-world split of the dataset. The results
        from the number of iterations are saved as npz files.

        :param iterations: number of runs (training/testing)
        :save: save predictions and labels if True
        """
        self.true_labels = np.array([])
        self.predictions = np.array([])
        for i in range(iterations):
            self.randomize_dataset_open_world()
            clf = GridSearchCV(svm.LinearSVC(), {'C': np.logspace(-3, 3, 7)})
            clf.fit(self.X_train, self.Y_train)
            out = clf.best_estimator_.decision_function(self.X_test)
            classes = clf.best_estimator_.classes_
            for scores in out:
                m = np.max(scores)
                if (abs(m/scores[:][:]) < 0.5).any():
                    self.predictions = np.append(self.predictions, 99)
                else:
                    p = classes[np.where(scores==m)]
                    self.predictions = np.append(self.predictions, p)
            self.true_labels = np.append(self.true_labels, self.Y_test)

        if save:
            np.savez_compressed("mca_predictions_open.npz", self.predictions)
            np.savez_compressed("mca_true_labels_open.npz", self.true_labels)

    ########################
    # Plotting functions #
    ########################

    def plot_average_roc(self, filename, boundary=0.1):
        """
        Plot an average roc curve up to boundary using the rocs object
        of the the Analysis object. It can be called after
        run_linear_experiment. 

        :filename: name of the file to save the roc plot
        :boundary: upper False Positive limit for the roc plot
        :returns: None. It saves the roc plot in a png file with
            the specified filename
        """

        fps = np.linspace(0.0, 1.0, 10000)
        (avg_roc, std_roc) = eval.average_roc(self.rocs, fps)
        std0 = std_roc[1]
        std1 = avg_roc[0] + std_roc[0]
        std2 = avg_roc[0] - std_roc[0]
        plt.plot(avg_roc[1], avg_roc[0], 'k-',
                 std0, std1, 'k--', std0, std2, 'k--')
        plt.legend(('Average ROC', 'StdDev ROC'), 'lower right', shadow=True)
        plt.xlabel('False Positive Rate')
        plt.xlim((0.0, boundary))
        plt.ylabel('True Positive Rate')
        plt.ylim((0.0, 1.0))
        plt.title("Average ROC")
        plt.grid(True)
        plt.savefig(filename, format='png')

    def plot_average_rocs(self, filename, roc_pickles, boundary=0.1):
        """
        Average several ROC curves from different models and plot
        them together in the same figure.

        :filename: name of the file to save the plot
        :boundary: upper False Positive limit for the roc plot
        :roc_pickles: list of npz file names containing several rocs each
        :returns: None. It saves the roc plot in a png file with the
            specified filename
        """

        fps = np.linspace(0.0, 1.0, 10000)
        plt.figure(figsize=(18.5, 10.5))
        linestyles = ['k-', 'k--', 'k-.', 'k:', 'k.',
                      'k*', 'k^', 'ko', 'k+', 'kx']
        for f, style in zip(roc_pickles, linestyles[:len(roc_pickles)]):
            avg_roc, std_roc = eval.average_roc(np.load(f), fps)
            plt.plot(avg_roc[1], avg_roc[0], style)
        plt.legend(roc_pickles, 'lower right', shadow=True)
        plt.xlabel('False Positive Rate')
        plt.xlim((0.0, boundary))
        plt.ylabel('True Positive Rate')
        plt.ylim((0.0, 1.0))
        plt.title("Average ROCs")
        plt.grid(True)
        plt.savefig(filename, format='png')

    def get_high_ranked_neighborhoods(self, fcg_file,
                                      sorted_weights_idx, n_weights=3):
        """
        Retrieve the neighborhoods in a hashed graph with maximum weights.
        n_weights: 

        :param fcg_file: path of file containing a fcg
        :param sorted_weights_idx: index that sort the weights from the
                linear classifier
        :param n_weights: number of weights with maximum value to retrieve
                the associated neighborhoods
        :returns: a list of matching neighborhoods.
        """
        # g = FCG.build_fcg(fcg_file)
        g = np.load(fcg_file)
        g_hash = ml.neighborhood_hash(g)
        bits = len(instructionSet.INSTRUCTION_CLASS_COLOR)

        neighborhoods = []
        remaining_weights = n_weights

        for idx in sorted_weights_idx:
            if remaining_weights > 0:
                label_decimal = idx / self.b
                label_bin = np.binary_repr(label_decimal, bits)
                label = np.array([int(i) for i in label_bin])
                matching_neighborhoods = []
                for m, nh in g_hash.node.iteritems():
                    if np.array_equal(nh["label"], label):
                        neighborhood = "{0} {1}.{2}({3})".format(remaining_weights,
                                                                 m[0], m[1], m[2])
                        matching_neighborhoods.append(neighborhood)
                if matching_neighborhoods:
                    remaining_weights -= 1
                    neighborhoods += matching_neighborhoods
            else:
                del g
                del g_hash
                return neighborhoods