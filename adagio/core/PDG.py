#!/usr/bin/python

import sys
import os

import networkx as nx
import adagio.common.pz as pz
from progressbar import *
from modules.androguard.androlyze import *
from adagio.common.utils import get_sha256


class PDGGenerator():

    def __init__(self, read_dir, out_dir):
        self.read_dir = read_dir
        self.out_dir = out_dir

    def process(self):
        """ Convert a series of APK into pdg objects. Load all
        APKs in a dir subtree and create PDG objects that are pickled
        for later processing and learning.
        """
        sys.setrecursionlimit(100000)
        files = []

        # check if pdg doesnt exist yet and mark the file to be processed
        for dirName, subdirList, fileList in os.walk(self.read_dir):
            for f in fileList:
                files.append(os.path.join(dirName, f))

        # set up progress bar
        print "\nProcessing {} APK files in dir {}".format(len(files),
                                                             self.read_dir)
        widgets = ['Building PDGs: ',
                   Percentage(), ' ',
                   Bar(marker='#', left='[', right=']'),
                   ' ', ETA(), ' ']

        pbar = ProgressBar(widgets=widgets, maxval=len(files))
        pbar.start()
        progress = 0

        # loop through .apk files and save them in .pdg.pz format
        for f in files:

            f = os.path.join(self.read_dir, fn)
            print '[] Loading {0}'.format(f)
            try:
                g = self.build_pdg(f)

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
            fnx = os.path.join(out, "{}.pdg.pz".format(h))
            pz.save(g, fnx)
            print "[*] Saved {}\n".format(fnx)
            progress += 1
            pbar.update(progress)
        pbar.finish()
        print "Done."

    def build_pdg(self, filename):
        pdg = nx.DiGraph()
        print "Loading file {0}...".format(filename)
        try:
            a, d, dx = AnalyzeAPK(filename)
        except zipfile.BadZipfile:
            #if file is not an APK, may be a dex object
            d, dx = AnalyzeDex(filename)

        methods = d.get_methods()

        # set up progress bar
        widgets = ['Building PDG: ',
                   Percentage(), ' ',
                   Bar(marker='#', left='[', right=']'),
                   ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, maxval=len(methods))
        pbar.start()
        progress = 0

        for method in methods:
            for bb in dx.get_method(method).basic_blocks.get():
                children = []
                label = self.get_bb_label(bb)
                children = self.get_children(bb, dx)
                pdg.add_node(label)
                pdg.add_edges_from([(label, child) for child in children])
            progress += 1
            pbar.update(progress)
        pbar.finish()
        return pdg

    def build_icfg_nx(self, filename):
        """ Using NX and Androguard, build an interprocedural control flow (ICFG)
        graph NX object so that node names are basic blocks names: (class name,
        method name, descriptor, bb)
        """
        icgnx = nx.DiGraph()
        print "Loading file {0}...".format(filename)
        try:
            a, d, dx = AnalyzeAPK(filename)
        except zipfile.BadZipfile:
            #if file is not an APK, may be a dex object
            d, dx = AnalyzeDex(filename)

        methods = d.get_methods()

        # set up progress bar
        widgets = ['Building ICFG: ',
                   Percentage(), ' ',
                   Bar(marker='#', left='[', right=']'),
                   ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, maxval=len(methods))
        pbar.start()
        progress = 0

        for method in methods:
            for bb in dx.get_method(method).basic_blocks.get():
                children = []
                label = self.get_bb_label(bb)
                children = self.get_children(bb, dx)
                icgnx.add_node(label)
                icgnx.add_edges_from([(label, child) for child in children])
            progress += 1
            pbar.update(progress)
        pbar.finish()
        return icgnx

    def get_bb_label(self, bb):
        """ Return the descriptive name of a basic block
        """
        return self.get_method_label(bb.method) + (bb.name,)

    def get_method_label(self, method):
        """ Return the descriptive name of a method
        """
        return (method.get_class_name(), method.get_name(), method.get_descriptor())

    def get_children(self, bb, dx):
        """ Return the labels of the basic blocks that are children of the
        input basic block in and out of its method
        """
        return self.get_bb_intra_method_children(bb) + self.get_bb_extra_method_children(bb, dx)

    def get_bb_intra_method_children(self, bb):
        """ Return the labels of the basic blocks that are children of the
        input basic block within a method
        """
        child_labels = []
        for c_in_bb in bb.get_next():
            next_bb = c_in_bb[2]
            child_labels.append(self.get_bb_label(next_bb))
        return child_labels

    def get_bb_extra_method_children(self, bb, dx):
        """ Given a basic block, find the calls to external methods and
        return the label of the first basic block in these methods
        """

        call_labels = []
        #iterate over calls from bb method to external methods
        for cob in bb.method.XREFto.items:
            remote_method = cob[0]
            remote_method_analysis = dx.get_method(remote_method)
            #iterate over the offsets of the call instructions and check
            #if the offset is within the limits of the bb
            for path in cob[1]:
                if self.call_in_bb(bb, path.get_idx()):
                    try:
                        remote_bb = remote_method_analysis.basic_blocks.get().next()
                        call_labels.append(self.get_bb_label(remote_bb))
                    except StopIteration:
                        pass
        return call_labels

    def call_in_bb(self, bb, idx):
        return bb.get_start() <= idx <= bb.get_end()
