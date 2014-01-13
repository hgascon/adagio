#!/usr/bin/python
# ADAGIO Android Application Graph-based Classification
# analysis.py >> Load dataset, train, predict and evaluate  
# Copyright (c) 2013 Hugo Gascon <hgascon@uni-goettingen.de>

import sys
import os
import ml
import eval
import FCGextractor
import instructionSet
import random
import collections
import pz
import numpy as np
import scipy as sp
from random import shuffle
import time
from progressbar import *
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.grid_search import GridSearchCV



class MultiClassAnalysis:
    """ A class to run a multiclass classification experiment """

    def __init__(self, dataset_dir, families, split, precomputed_matrix="", y="", fnames=""):
     
        self.split = split
        self.families = families
        self.X = []
        self.Y = np.array([])
        self.fnames = []
        self.x_train = [] 
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.clf = ""
        self.out = []
        self.roc = 0
        self.auc = 0
        self.b = 0
        self.feature_vector_times = []
        self.label_dist = np.zeros(2**15)
        self.sample_sizes = []
        self.neighborhood_sizes = []
        self.class_dist = np.zeros(15)

        if precomputed_matrix:
            #Load the y labels and file names from zip pickle objects.
            print "Loading matrix..."
            self.X = pz.load(precomputed_matrix)
            print "[*] matrix loaded"
            self.Y = pz.load(y)
            print "[*] labels loaded"
            self.fnames = pz.load(fnames)
            print "[*] file names loaded"

        else:            
            files = self.read_files(dataset_dir, "fcgnx.pz")
            if len(files) > 0:
                print "Loading {0} samples".format(len(files))
                widgets = ['Unpickling... : ', Percentage(), ' ', Bar(marker='#',left='[',right=']'),
                            ' ', ETA(), ' ']
                pbar = ProgressBar(widgets=widgets, maxval=len(files))
                pbar.start()
                progress = 0

                # add each file name, label and object in self.dataset
                for f in files:
                    try:
                        sha = os.path.basename(f).split(".")[0]
                        label = families[sha]
                        g = pz.load(f)
                        size = g.number_of_nodes()
                        if size > 0:

                            t0 = time.time()
                            x_i = self.compute_feature_vector(g)
                            self.feature_vector_times.append( time.time() - t0 )
                            self.label_dist = np.sum([self.label_dist, x_i], axis=0)
                            self.sample_sizes.append(size)
                            self.neighborhood_sizes += ml.neighborhood_sizes(g)
                            for n, l in g.node.iteritems():
                                self.class_dist = np.sum([self.class_dist, l["label"]], axis=0)

                            # delete nx object to free memory
                            del g
                            self.X.append(x_i)
                            self.Y = np.append(self.Y, [int(label)])
                            self.fnames.append(f)

                    except KeyError:
                        pass
                    progress += 1
                    pbar.update(progress)
                pbar.finish()

            # convert feature vectors to its binary representation
            # and make the data matrix sparse
            print "[*] Stacking feature vectors..."
            self.X = np.vstack(self.X)
            print "[*] Converting features vectors to binary..."
            self.X, self.b = ml.make_binary(self.X)  

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
        for fn in  os.listdir(d):
            if fn.lower().endswith(file_extension):
                files.append(os.path.join(d, fn))
        shuffle(files)

        #if max_files is 0, return all the files in dir
        if max_files == 0:
            max_files = len(files)
        files = files[:max_files]
        
        return files

    def compute_feature_vector(self, g):
        """ Compute the neighboorhood hash of a graph g and return
            the histogram of the hashed labels.
        """
        g_hash = ml.neighborhood_hash(g)
        g_x = ml.label_histogram(g_hash)
        return g_x

    def remove_unpopulated_families(self, min_family_size=20):
        """ Remove classes with less than a minimum number of samples.
        """ 
        c = collections.Counter(self.Y)
        unpopulated_families = [f for f, n in c.iteritems() if n < min_family_size]
        index_bool = np.in1d(self.Y.ravel(), unpopulated_families).reshape(self.Y.shape)
        self.Y = self.Y[~index_bool]
        index_num = np.arange(self.X.shape[0])[~index_bool]
        self.X = self.X[index_num, :]

    def randomize_dataset_open_world(self):
        """ Split the dataset in training and testing sets where
            classes used for training are not present in testing set.
        """
        classes = list(set(self.Y))
        shuffle(classes)
        train_classes = classes[:int(self.split * len(classes))]
        test_classes = classes[int(self.split * len(classes)):]

        train_idx_bool = np.in1d(self.Y.ravel(), train_classes).reshape(self.Y.shape)
        test_idx_bool = np.in1d(self.Y.ravel(), test_classes).reshape(self.Y.shape)

        self.Y_train = self.Y[train_idx_bool]
        self.Y_test = self.Y[test_idx_bool]

        idx_num = np.arange(self.X.shape[0])
        self.X_train = self.X[ idx_num[train_idx_bool], :]
        self.X_test = self.X[ idx_num[test_idx_bool], :]

    def randomize_dataset_closed_world(self):
        """ Randomly split the dataset in training and testing sets
        """
        N = len(self.Y)
        train_size = int(self.split * N)
        index = range(N)
        shuffle(index)
        train_index = sorted(index[:train_size])
        test_index = sorted(index[train_size:])

        self.X_train = self.X[ train_index, : ]
        self.X_test = self.X[ test_index, : ]
        self.Y_train = self.Y[ train_index ]
        self.Y_test = self.Y[ test_index ]

    def run_closed_experiment(self, iterations=10):
        """ Train a classifier on test data, obtain the best combination of
        paramters through a grid search cross-validation and test the classifier
        using a closed-world split of the dataset. The results from the number
        of iterations are saved as pz files.
        """
        self.true_labels = np.array([])
        self.predictions = np.array([])
        for i in xrange(iterations):
            self.randomize_dataset_closed_world()
            clf = GridSearchCV(svm.LinearSVC(), {'C':np.logspace(-3,3,7)})
            clf.fit(self.X_train, self.Y_train)
            out = clf.best_estimator_.predict(self.X_test)
            self.predictions = np.append(self.predictions, out)
            self.true_labels = np.append(self.true_labels, self.Y_test)

        pz.save(self.predictions, "mca_predictions_closed.pz")
        pz.save(self.true_labels, "mca_true_labels_closed.pz")

    def run_open_experiment(self, iterations=10):
        """ Train a classifier on test data, obtain the best combination of
        paramters through a grid search cross-validation and test the classifier
        using a open-world split of the dataset. The results from the number
        of iterations are saved as pz files.
        """
        self.true_labels = np.array([])
        self.predictions = np.array([])
        for i in xrange(iterations):
            self.randomize_dataset_open_world()  
            clf = GridSearchCV(svm.LinearSVC(), {'C':np.logspace(-3,3,7)})
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
    
        pz.save(self.predictions, "mca_predictions_open.pz")
        pz.save(self.true_labels, "mca_true_labels_open.pz")



class Analysis:
    """ A class to run a binary classification experiment """

    def __init__(self, dirs, labels, split, max_files=0, max_node_size=0, 
                 precomputed_matrix="", y="", fnames=""):
        """ 

        Args:
            dirs: A list with two directories including both types of files for
                classification <[MALWARE_DIR, CLEAN_DIR]>
            labels: The labels assigned to samples in each dir <[1, 0]>
            split: The percentage of samples used for training (value between 0 and 1)
            precomputed_matrix: name of file if a data or kernel matrix has already
                been computed.
            y: If precomputed_matrix is True, a pickled and gzipped list of labels must
                be provided.

        Returns:
            An Analysis object with the dataset as a set of properties and several
            functions to train, test, evalute or run a learning experiment iteratively.

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
        self.roc = 0
        self.auc = 0
        self.b = 0
        
        if precomputed_matrix:
            #Load the y labels and file names from zip pickle objects.
            print "Loading matrix..."
            self.X = pz.load(precomputed_matrix)
            print "[*] matrix loaded"
            self.Y = pz.load(y)
            print "[*] labels loaded"
            self.fnames = pz.load(fnames)
            print "[*] file names loaded"

        else:
            # loop over dirs
            for d in zip(dirs, labels):
                files = self.read_files(d[0], "fcgnx.pz", max_files)
                print "Loading samples in dir {0} with label {1}".format(d[0], d[1])
                widgets = ['Unpickling... : ', Percentage(), ' ', Bar(marker='#',left='[',right=']'),
                                   ' ', ETA(), ' ']
                pbar = ProgressBar(widgets=widgets, maxval=len(files))
                pbar.start()
                progress = 0

                # load labels and feature vectors
                for f in files:
                    try: 
                        g = pz.load(f)
                        n = g.number_of_nodes()
                        if n < max_node_size or max_node_size == 0:
                            if n > 0:
                                x_i = self.compute_label_histogram(g)
                                # delete nx object to free memory
                                del g
                                self.X.append(x_i)
                                self.Y = np.append(self.Y, [int(d[1])])
                                self.fnames.append(f)
                    except Exception, e:
                        print e
                        print "err: {0}".format(f)
                        pass
                    progress += 1
                    pbar.update(progress)

                pbar.finish()

            # convert feature vectors to its binary representation
            # and make the data matrix sparse
            print "[*] Stacking feature vectors..."
            self.X = np.array(self.X, dtype=np.int16)
            print "[*] Converting features vectors to binary..."
            self.X, self.b = ml.make_binary(self.X) 
           
            # save labels, data matrix and file names
            #print "[*] Saving labels, data matrix and file names..."
            #pz.save(self.X, "X.pz")
            #pz.save(self.Y, "Y.pz")            
            #pz.save(self.fnames, "fnames.pz")


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
        for fn in  os.listdir(d):
            if fn.lower().endswith(file_extension):
                files.append(os.path.join(d, fn))
        shuffle(files)

        #if max_files is 0, return all the files in dir
        if max_files == 0:
            max_files = len(files)
        files = files[:max_files]
        
        return files

    def compute_label_histogram(self, g):
        """ Compute the neighboorhood hash of a graph g and return
            the histogram of the hashed labels.
        """

        g_hash = ml.neighborhood_hash(g)
        g_x = ml.label_histogram(g_hash)
        return g_x
    
    def randomize_dataset(self):
        """ Randomly split the dataset in training and testing sets
        """

        N = len(self.Y)
        train_size = int(self.split * N)
        index = range(N)
        shuffle(index)
        train_index = sorted(index[:train_size])
        test_index = sorted(index[train_size:])

        self.X_train = self.X[ train_index, : ]
        self.X_test = self.X[ test_index, : ]
        self.Y_train = self.Y[ train_index ]
        self.Y_test = self.Y[ test_index ]


    ###################################
    # Learning & Evaluation functions #
    ###################################

    def run_linear_experiment(self, rocs_filename, iterations=10):

        self.rocs = []
        for i in xrange(iterations):
            print "[*] Iteration {0}".format(i)
            print "[*] Randomizing dataset..."
            self.randomize_dataset()
            clf = GridSearchCV(svm.LinearSVC(), {'C':np.logspace(-3,3,7)})
            print "[*] Training..."
            clf.fit(self.X_train, self.Y_train)
            out = clf.best_estimator_.decision_function(self.X_test)
            print "[*] Testing..."
            roc = eval.compute_roc(np.float32(out.flatten()), np.float32(self.Y_test))
            self.rocs.append(roc)
            print "[*] ROC saved."
        pz.save(self.rocs, rocs_filename)

    ########################
    # Plotting functions #
    ########################

        
    def plot_average_roc(self, filename, boundary=0.1):
        """ Plot an average roc curve up to boundary using the rocs object of the the
        Analysis object. It can be called after run_linear_experiment. 

        Args:
            filename: name of the file to save the roc plot
            boundary: upper False Positive limit for the roc plot

        Returns:
            None. It saves the roc plot in a png file with the specified filename
        """

        fps = np.linspace( 0.0, 1.0, 10000 )
        (avg_roc, std_roc) = eval.average_roc( self.rocs, fps )
        std0 = std_roc[1]
        std1 = avg_roc[0] + std_roc[0]
        std2 = avg_roc[0] - std_roc[0]
        plt.plot(avg_roc[1], avg_roc[0], 'k-', std0, std1, 'k--', std0, std2, 'k--')
        plt.legend(('Average ROC', 'StdDev ROC'), 'lower right', shadow=True)
        plt.xlabel('False Positive Rate')
        plt.xlim((0.0, boundary))
        plt.ylabel('True Positive Rate')
        plt.ylim((0.0, 1.0))
        plt.title("Average ROC")
        plt.grid(True)
        plt.savefig(filename, format='png')

    def plot_average_rocs(self, filename, roc_pickles, boundary=0.1):
        """ Average several ROC curves from different models and plot them together in the
        same figure.
        
        Args:
            filename: name of the file to save the plot
            boundary: upper False Positive limit for the roc plot
            roc_pickles: list of pz file names containing several rocs each.  

        Returns:
            None. It saves the roc plot in a png file with the specified filename
        """

        fps = np.linspace( 0.0, 1.0, 10000 )
        plt.figure(figsize=(18.5,10.5))
        linestyles = ['k-','k--','k-.','k:','k.','k*','k^','ko','k+','kx']
        for f, style in zip(roc_pickles, linestyles[:len(roc_pickles)]):
            (avg_roc, std_roc) = eval.average_roc( pz.load(f), fps )
            plt.plot(avg_roc[1], avg_roc[0], style)
        plt.legend((roc_pickles), 'lower right', shadow=True)
        plt.xlabel('False Positive Rate')
        plt.xlim((0.0, boundary))
        plt.ylabel('True Positive Rate')
        plt.ylim((0.0, 1.0))
        plt.title("Average ROCs")
        plt.grid(True)
        plt.savefig( filename, format='png' )


    def get_high_ranked_neighborhoods(self, fcgnx_file, sorted_weights_idx, n_weights=3):
        """ Retrieve the neigborhoods in a hashed graph with maximum weights.
        
        Args:
            fcgnx_file: path of a fcgnx file containing a fcg.
            sorted_weights_idx: index that sort the weights from the linear classifer.
            n_weights: number of weights with maximum value to retrieve the
                associated neighborhoods.

        Returns:
            A list of matching neighborhoods.
        """
        # g = FCGextractor.build_cfgnx(fcgnx_file)
        g = pz.load(fcgnx_file)
        g_hash = ml.neighborhood_hash(g)
        bits = len(instructionSet.INSTRUCTION_CLASS_COLOR)
        
        neighborhoods = []
        remaining_weights = n_weights

        for idx in sorted_weights_idx:
            if remaining_weights > 0:
                label_decimal = idx / self.b
                label_bin = np.binary_repr( label_decimal, bits )
                label = np.array( [ int(i) for i in label_bin ] )
                matching_neighborhoods = []
                for m, nh in g_hash.node.iteritems():
                    if np.array_equal( nh["label"], label ):
                        matching_neighborhoods.append("{0} {1}.{2}({3})".format(remaining_weights,
                                                                                m[0], m[1], m[2]))
                if matching_neighborhoods:
                    remaining_weights -= 1
                    neighborhoods += matching_neighborhoods
            else:
                del g
                del g_hash
                return neighborhoods
