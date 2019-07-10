# ADAGIO Structural Analysis of Android Binaries
# Copyright (c) 2014 Hugo Gascon <hgascon@mail.de>

""" A module to build NX graph objects from APKs. """

import zipfile
import networkx as nx
import numpy as np

from adagio.core.instructionSet import INSTRUCTION_SET_COLOR
from adagio.core.instructionSet import INSTRUCTION_CLASS_COLOR
from adagio.core.instructionSet import INSTRUCTION_CLASSES

from androguard.core.bytecodes.apk import APK
from androguard.core.analysis.analysis import Analysis
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.misc import AnalyzeDex
from adagio.common.utils import get_sha256
from tqdm import tqdm
import sys
import os


class FCG():

    def __init__(self, filename):
        self.filename = filename
        try:
            self.a = APK(filename)
            self.d = DalvikVMFormat(self.a.get_dex())
            self.d.create_python_export()
            self.dx = Analysis(self.d)
        except zipfile.BadZipfile:
            # if file is not an APK, may be a dex object
            _, self.d, self.dx = AnalyzeDex(self.filename)

        self.d.set_vmanalysis(self.dx)
        self.dx.create_xref()
        self.fcg = self.build_fcg()

    def get_graph(self):
        return self.fcg

    def build_fcg(self):
        """ Using NX and Androguard, build a directed graph NX object so that:
            - node names are analysis.MethodClassAnalysis objects
            - each node has a label that encodes the method behavior
        """
        fcg = self.dx.get_call_graph()
        for n in fcg.nodes:
            instructions = []
            try:
                ops = n.get_instructions()
                for i in ops:
                    instructions.append(i.get_name())
                encoded_label = self.color_instructions(instructions)
            except AttributeError:
                encoded_label = np.array2string(np.zeros(len(INSTRUCTION_CLASSES),
                                                             dtype=np.int8))
            fcg.node[n]["label"]  = encoded_label
        return fcg

    def color_instructions(self, instructions):
        """ Node label based on coloring technique by Kruegel """

        h = [0] * len(INSTRUCTION_CLASS_COLOR)
        for i in instructions:
            h[INSTRUCTION_SET_COLOR[i]] = 1
        return np.array(h)

    def get_classes_from_label(self, label):
        classes = [INSTRUCTION_CLASSES[i] for i in range(len(label)) if label[i] == 1]
        return classes


class PDG():

    def __init__(self, filename):
        """

        :type self: object
        """
        self.filename = filename
        try:
            self.a = APK(filename)
            self.d = DalvikVMFormat(self.a.get_dex())
            self.d.create_python_export()
            self.dx = Analysis(self.d)
        except zipfile.BadZipfile:
            # if file is not an APK, may be a dex object
            _, self.d, self.dx = AnalyzeDex(self.filename)

        self.d.set_vmanalysis(self.dx)
        self.dx.create_xref()
        self.fcg = self.dx.get_call_graph()
        self.icfg = self.build_icfg()

    def get_graph(self):
        return self.icfg
        
    def build_icfg(self):
        icfg = nx.DiGraph()
        methods = self.d.get_methods()
        for method in methods:
            for bb in self.dx.get_method(method).basic_blocks.get():
                children = []
                label = self.get_bb_label(bb)
                children = self.get_children(bb, self.dx)
                icfg.add_node(label)
                icfg.add_edges_from([(label, child) for child in children])
        return icfg

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
        # iterate over calls from bb method to external methods
        try:
            xrefs = dx.get_method_analysis(bb.method).get_xref_to()
        except AttributeError:
            return call_labels
        for xref in xrefs:
            remote_method_offset = xref[2]
            if self.call_in_bb(bb, remote_method_offset):
                try:
                    remote_method = dx.get_method(self.d.get_method_by_idx(remote_method_offset))
                    if remote_method:
                        remote_bb = next(remote_method.basic_blocks.get())
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

    # loop through .apk files and save them in .pdg.pz format
    print("\nProcessing {} APK files in dir {}".format(len(files), read_dir))
    for f in tqdm(files):

        f = os.path.realpath(f)
        print('[] Loading {0}'.format(f))
        try:
            if mode is 'FCG':
                graph = FCG(f).get_graph()
            elif mode is 'PDG':
                graph = PDG(f).get_graph()

        # if an exception happens, save the .apk in the corresponding dir
        except Exception as e:
            err = e.__class__.__name__
            err_dir = err + "/"
            d = os.path.join(read_dir, err_dir)
            if not os.path.exists(d):
                os.makedirs(d)
            cmd = "cp {} {}".format(f, d)
            os.system(cmd)
            print("[*] {} error loading {}".format(err, f))
            continue

        h = get_sha256(f)
        if not out_dir:
            out_dir = read_dir
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        fnx = os.path.join(out_dir, "{}".format(h))
        nx.write_gpickle(graph, fnx)
        print("[*] Saved {}\n".format(fnx))
    print("Done.")
