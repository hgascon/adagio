#!/usr/bin/python
# ADAGIO Android Application Graph-based Classification
# fcg_extractor.py >> Read all APKs in dir and save NX graphs as pickle objects
# Copyright (c) 2013 Hugo Gascon <hgascon@uni-goettingen.de>

""" A module to build NX objects from APKs call graphs. """

import sys
import os
import zipfile
import networkx as nx
import numpy as np
import adagio.common.pz as pz

from adagio.core.instructionSet import INSTRUCTION_SET_COLOR
from adagio.core.instructionSet import INSTRUCTION_CLASS_COLOR
from adagio.core.instructionSet import INSTRUCTION_CLASSES

from progressbar import *
from modules.androguard.androguard import *
from modules.androguard.androguard.core.analysis import *
from modules.androguard.androlyze import *
from adagio.common.utils import get_sha256


class FCGGenerator():

    def __init__(self, read_dir, out_dir):
        self.read_dir = read_dir
        self.out_dir = out_dir

    def process(self):
        """ Convert a series of APK into fcg objects. Load all
        APKs in a dir subtree and create FCG objects that are pickled
        for later processing and learning.
        """
        sys.setrecursionlimit(100000)
        files = []

        # check if fcg doesnt exist yet and mark the file to be processed
        for dirName, subdirList, fileList in os.walk(self.read_dir):
            for f in fileList:
                files.append(os.path.abspath(os.path.join(dirName, f)))

        # set up progress bar
        print "\nProcessing {} APK files in dir {}".format(len(files),
                                                           self.read_dir)
        widgets = ['Building FCGs: ',
                   Percentage(), ' ',
                   Bar(marker='#', left='[', right=']'),
                   ' ', ETA(), ' ']

        pbar = ProgressBar(widgets=widgets, maxval=len(files))
        pbar.start()
        progress = 0

        # loop through .apk files and save them in .fcg,pz format
        for f in files:

            # f = os.path.join(self.read_dir, fn)
            print "[] Loading {}".format(f)
            try:
                g = self.build_fcg(f)

            # if an exception happens, save the .apk in the corresponding dir
            except Exception as e:
                err = e.__class__.__name__
                err_dir = err + "/"
                d = os.path.join(self.read_dir, err_dir)
                if not os.path.exists(d):
                    os.makedirs(d)
                cmd = "cp {} {}".format(f, d)
                os.system(cmd)
                print "[*] {} error loading {}".format(err, f)
                continue

            h = get_sha256(f)
            if self.out_dir:
                out = self.out_dir
            else:
                out = self.read_dir
            fnx = os.path.join(out, "{}.fcg.pz".format(h))
            pz.save(g, fnx)
            print "[*] Saved {}\n".format(fnx)
            progress += 1
            pbar.update(progress)
        pbar.finish()
        print "Done."

    def build_fcg(self, filename):
        """ Using NX and Androguard, build a directed graph NX object so that:
            - node names are method names as: class name, method name and
              descriptor
            - each node has a label that encodes the method behavior
        """
        # nx graph for FCG extracted from APK: nodes = method_name,
        # labels = encoded instructions
        fcgnx = nx.DiGraph()
        try:
            a, d, dx = AnalyzeAPK(filename)
        except zipfile.BadZipfile:
            #if file is not an APK, may be a dex object
            d, dx = AnalyzeDex(filename)

        for method in d.get_methods():

            node_name = self.get_node_name(method)

            #find calls from this method
            children = []
            for cob in method.XREFto.items:
                remote_method = cob[0]
                children.append(self.get_node_name(remote_method))

            #find all instructions in method and encode using coloring
            instructions = []
            for i in method.get_instructions():
                instructions.append(i.get_name())
            encoded_label = self.color_instructions(instructions)

            #add node, children and label to nx graph
            fcgnx.add_node(node_name, label=encoded_label)
            fcgnx.add_edges_from([(node_name, child) for child in children])

        return fcgnx

    def get_node_name(self, method):
        """ Build unique identifier for a method """

        return (method.get_class_name(),
                method.get_name(),
                method.get_descriptor())

    def color_instructions(self, instructions):
        """ Node label based on coloring technique by Kruegel """

        h = [0] * len(INSTRUCTION_CLASS_COLOR)
        for i in instructions:
            h[INSTRUCTION_SET_COLOR[i]] = 1
        return np.array(h)

    def get_classes_from_label(self, label):

        classes = [INSTRUCTION_CLASSES[i] for i in xrange(len(label)) if label[i] == 1]
        return classes
