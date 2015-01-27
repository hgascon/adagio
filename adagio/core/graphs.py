#!/usr/bin/python
# ADAGIO Structural Analysis of Android Binaries
# Copyright (c) 2014 Hugo Gascon <hgascon@mail.de>

""" A module to build NX graph objects from APKs. """

import zipfile
import networkx as nx
import numpy as np

from adagio.core.instructionSet import INSTRUCTION_SET_COLOR
from adagio.core.instructionSet import INSTRUCTION_CLASS_COLOR
from adagio.core.instructionSet import INSTRUCTION_CLASSES

from progressbar import *
from modules.androguard.androlyze import *
from adagio.common.utils import get_sha256
import adagio.common.pz as pz


class FCG():

    def __init__(self, filename):
        self.filename = filename
        self.g = self.build_fcg()

    def build_fcg(self):
        """ Using NX and Androguard, build a directed graph NX object so that:
            - node names are method names as: class name, method name and
              descriptor
            - each node has a label that encodes the method behavior
        """
        # nx graph for FCG extracted from APK: nodes = method_name,
        # labels = encoded instructions
        fcg = nx.DiGraph()
        try:
            self.a = APK(self.filename)
            self.d = DalvikVMFormat(self.a.get_dex())
            self.d.create_python_export()
            self.dx = VMAnalysis(self.d)
            self.gx = GVMAnalysis(self.dx, self.a)
        except zipfile.BadZipfile:
            #if file is not an APK, may be a dex object
            self.d, self.dx = AnalyzeDex(self.filename)

        self.d.set_vmanalysis(self.dx)
        self.d.set_gvmanalysis(self.gx)
        self.d.create_xref()
        self.d.create_dref()

        methods = self.d.get_methods()
        for method in methods:
            node_name = self.get_method_label(method)

            #find calls from this method
            children = []
            for cob in method.XREFto.items:
                remote_method = cob[0]
                children.append(self.get_method_label(remote_method))

            # find all instructions in method and encode using coloring
            instructions = []
            for i in method.get_instructions():
                instructions.append(i.get_name())
            encoded_label = self.color_instructions(instructions)
            #add node, children and label to nx graph
            fcg.add_node(node_name, label=encoded_label)
            fcg.add_edges_from([(node_name, child) for child in children])

        return fcg

    def get_method_label(self, method):
        """ Return the descriptive name of a method
        """
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


class PDG():

    def __init__(self, filename):
        self.filename = filename
        self.g = self.build_icfg()
        self.nodes = {}

    def build_icfg(self):
        icfg = nx.DiGraph()
        try:
            self.a = APK(self.filename)
            self.d = DalvikVMFormat(self.a.get_dex())
            self.d.create_python_export()
            self.dx = VMAnalysis(self.d)
            self.gx = GVMAnalysis(self.dx, self.a)
        except zipfile.BadZipfile:
            #if file is not an APK, may be a dex object
            self.d, self.dx = AnalyzeDex(self.filename)

        self.d.set_vmanalysis(self.dx)
        self.d.set_gvmanalysis(self.gx)
        self.d.create_xref()
        self.d.create_dref()

        methods = self.d.get_methods()
        for method in methods:
            for bb in self.dx.get_method(method).basic_blocks.get():
                children = []
                label = self.get_bb_label(bb)
                children = self.get_children(bb, self.dx)
                icfg.add_node(label)
                icfg.add_edges_from([(label, child) for child in children])

        return icfg

    #def get_entry_points(self):
        #return self.gx.entry_nodes

    def get_bb_label(self, bb):
        """ Return the descriptive name of a basic block
        """
        return self.get_method_label(bb.method) + (bb.name,)

    def get_method_label(self, method):
        """ Return the descriptive name of a method
        """
        return (method.get_class_name(),
                method.get_name(),
                method.get_descriptor())

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


def process_dir(read_dir, out_dir, mode='FCG'):
    """ Convert a series of APK into graph objects. Load all
    APKs in a dir subtree and create graph objects that are pickled
    for later processing and learning.
    """
    sys.setrecursionlimit(100000)
    files = []

    # check if pdg doesnt exist yet and mark the file to be processed
    for dirName, subdirList, fileList in os.walk(read_dir):
        for f in fileList:
            files.append(os.path.join(dirName, f))

    # set up progress bar
    print "\nProcessing {} APK files in dir {}".format(len(files), read_dir)
    widgets = ['Building graphs: ',
               Percentage(), ' ',
               Bar(marker='#', left='[', right=']'),
               ' ', ETA(), ' ']

    pbar = ProgressBar(widgets=widgets, maxval=len(files))
    pbar.start()
    progress = 0

    # loop through .apk files and save them in .pdg.pz format
    for f in files:

        f = os.path.realpath(f)
        print '[] Loading {0}'.format(f)
        try:
            if mode is 'FCG':
                graph = FCG(f)
            elif mode is 'PDG':
                graph= PDG(f)

        # if an exception happens, save the .apk in the corresponding dir
        except Exception as e:
            err = e.__class__.__name__
            err_dir = err + "/"
            d = os.path.join(read_dir, err_dir)
            if not os.path.exists(d):
                os.makedirs(d)
            cmd = "cp {} {}".format(f, d)
            os.system(cmd)
            print "[*] {} error loading {}".format(err, f)
            continue

        h = get_sha256(f)
        if out_dir:
            out = out_dir
        else:
            out = read_dir
        fnx = os.path.join(out, "{}.pz".format(h))
        pz.save(graph.g, fnx)
        print "[*] Saved {}\n".format(fnx)
        progress += 1
        pbar.update(progress)
    pbar.finish()
    print "Done."
