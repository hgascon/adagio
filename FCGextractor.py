#!/usr/bin/python
# ADAGIO Android Application Graph-based Classification
# fcg_extractor.py >> Read all APKs in dir and save NX graphs as pickle objects
# Copyright (c) 2013 Hugo Gascon <hgascon@uni-goettingen.de>

""" A module to build NX objects from APKs call graphs. """

import sys
import os
import random
import struct
import zipfile
import networkx as nx
import numpy as np
import pz
from hashlib import sha256
from progressbar import *
from optparse import OptionParser
from instructionSet import INSTRUCTION_SET_COLOR 
from instructionSet import INSTRUCTION_CLASS_COLOR
from instructionSet import INSTRUCTION_CLASSES
from androguard import *
from androguard.core.analysis import *
from androlyze import *



##################################################################################
#                          APK to NX encoding functions                          #    
##################################################################################

def process_apk_dir(dataset_dir):
    """ Convert a series of APK into FCGNX objects

    Load all APKs in a dir subtree and create FCG objects that are
    pickled for later processing and learning. 

    Args:
        dataset_dir: a directory containing a list of APK files.

    """
    sys.setrecursionlimit(100000)
    files = []

    # check if fcg doesnt exist yet and mark the file to be processed
    for dirName, subdirList, fileList in os.walk(dataset_dir):
        for f in fileList:
            files.append(os.path.join(dirName,f))

    # set up progress bar            
    print "\nProcessing {0} APK files in dir {1}".format(len(files), dataset_dir)
    widgets = ['Building CGs: ', Percentage(), ' ', Bar(marker='#',left='[',right=']'),
                       ' ', ETA(), ' ']
    pbar = ProgressBar(widgets=widgets, maxval=len(files))
    pbar.start()
    progress = 0

    # loop through .apk files and save them in .fcg format
    for f in files:
        # f = os.path.join(dataset_dir, fn)
        print "[] Loading {0}".format(f)
        try:
            g = build_fcg_nx(f)
        # if an exception happens, save the .apk in the corresponding dir
        except Exception as e:
            err = e.__class__.__name__
            err_dir = err + "/"
            d = os.path.join(dataset_dir, err_dir)
            if not os.path.exists(d):
                os.makedirs(d)
            cmd = "cp {0} {1}".format(f, d)
            os.system(cmd)
            print "[*] {0} error loading {1}".format(err, f)
            continue

        h = get_sha256(f)
        fnx = os.path.join(os.path.split(f)[0], "{0}.fcg.pz".format(h))
        pz.save(g, fnx)
        print "[*] Saved {0}\n".format(fnx)
        progress += 1
        pbar.update(progress)
    pbar.finish()
    print "Done."

def get_sha256(filename):
    """ Return sha256 of the file in the input path. """
    f = open(filename)
    s = sha256()
    s.update(f.read())
    digest = s.hexdigest()
    f.close()
    return digest

def build_fcg_nx(file):
    """ Using NX and Androguard, build a directed graph NX object so that:
        - node names are method names as: class name, method name and descriptor
        - each node has a label that encodes the method behavior
    """ 
    #nx graph for FCG extracted from APK: nodes = method_name, labels = encoded instructions 
    fcgnx = nx.DiGraph()
    try:
        a, d, dx = AnalyzeAPK(file)
    except zipfile.BadZipfile:
        #if file is not an APK, may be a dex object
        d, dx = AnalyzeDex(file)

    for method in d.get_methods():
        
        node_name = get_node_name(method) 
        
        #find calls from this method
        children = []
        for cob in method.XREFto.items:
            remote_method = cob[0]
            children.append(get_node_name(remote_method))

        #find all instructions in method and encode using coloring
        instructions = []
        for i in method.get_instructions():
            instructions.append(i.get_name())
        encoded_label = color_instructions(instructions)
        
        #add node, children and label to nx graph
        fcgnx.add_node(node_name, label = encoded_label)
        fcgnx.add_edges_from([(node_name, child) for child in children])

    return fcgnx

def get_node_name(method):
    """ Build unique identifier for a method """
    return (method.get_class_name(), method.get_name(), method.get_descriptor()) 

def color_instructions(instructions):
    """ Node label based on coloring technique by Kruegel """
    h = [0] * len(INSTRUCTION_CLASS_COLOR)
    for i in instructions:
        h[INSTRUCTION_SET_COLOR[i]] = 1
    return np.array(h) 

def get_classes_from_label(label):
    idx = np.where(label==1)[0]
    classes = [INSTRUCTION_CLASSES[i] for i in xrange(len(label)) if label[i]==1]
    return classes

#####################################################################
#                       ICFG related functions                      #    
#####################################################################

def build_icfg_nx(file):
    """ Using NX and Androguard, build a directed graph NX object so that
        node names are basic blocks names: (class name, method name, descriptor, bb)
    """ 
    icgnx = nx.DiGraph()
    print "Loading file {0}...".format(file)
    a, d, dx = AnalyzeAPK(file)
    methods = d.get_methods()

    # set up progress bar            
    widgets = ['Building ICFG: ', Percentage(), ' ', Bar(marker='#',left='[',right=']'),
                       ' ', ETA(), ' ']
    pbar = ProgressBar(widgets=widgets, maxval=len(methods))
    pbar.start()
    progress = 0   
    
    for method in methods:
        for bb in dx.get_method(method).basic_blocks.get():
            children = []
            label = get_bb_label(bb)
            children = get_children(bb, dx)
            icgnx.add_node(label)
            icgnx.add_edges_from([(label, child) for child in children])
        progress += 1
        pbar.update(progress)
    pbar.finish()
    return icgnx
    
def get_bb_label(bb):
    """ Return the descriptive name of a basic block
    """
    return get_method_label(bb.method) + (bb.name,)

def get_method_label(method):
    """ Return the descriptive name of a method
    """
    return (method.get_class_name(), method.get_name(), method.get_descriptor())

def get_children(bb, dx):
    """ Return the labels of the basic blocks that are children of the input 
        basic block in and out of its method
    """
    return get_bb_intra_method_children(bb) + get_bb_extra_method_children(bb, dx)

def get_bb_intra_method_children(bb):
    """ Return the labels of the basic blocks that are children of the input 
    basic block within a method
    """
    child_labels = []
    for call_in_bb in bb.get_next():
        next_bb = call_in_bb[2]
        child_labels.append(get_bb_label(next_bb))
    return child_labels

def get_bb_extra_method_children(bb, dx): 
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
            if call_in_bb(bb, path.get_idx()):
                try:
                    remote_bb = remote_method_analysis.basic_blocks.get().next()
                    call_labels.append(get_bb_label(remote_bb))
                except StopIteration:
                    pass
    return call_labels

def call_in_bb(bb, idx):
    return bb.get_start() <= idx <= bb.get_end()

def list_XREF(file):
    try:
        a, d, dx = AnalyzeAPK(file)
    except zipfile.BadZipfile:
        #if file is not an APK, may be a dex object
        d, dx = AnalyzeDex(file)

    for method in d.get_methods():
        print get_node_name(method)
        print "XREFfrom:", [get_node_name(m[0]) for m in method.XREFfrom.items]
        print "XREFto:", [get_node_name(m[0]) for m in method.XREFto.items]

##################################################################################
#                       Android API Calls related functions                      #    
##################################################################################

def list_calls_apks_in_dir(dir, l):
    """ Return a list with all API calls found in first l APK files in dir """
    calls = []
    for f in os.listdir(dir):
        if f.lower().endswith("apk") and l>0:
            calls.append(list_calls(os.path.join(dir, f)))
            l -= 1
    return calls

def list_calls(file):
    """ Return a list with all API calls found in file (APK). Calls definition
        are reformatted as in java declarations.
    """
    apicalls = []
    a, d, dx = AnalyzeAPK(file)
    for method in d.get_methods():
        for i in method.get_instructions():
            if i.get_name()[:6] == "invoke":
                call = i.get_output(0).split(',')[-1].strip() #get method desc
                call = call[:call.index(')')+1]  #remove return value
                call = call.split('->') #split in class and method
                method_class = get_type(call[0])
                ins_method, params = call[1].split('(')
                params = ','.join(parse_parameters(params.replace(')','')))
                apicall = "{0}.{1}({2})".format(method_class, ins_method, params)
                apicalls.append(apicall)
    return apicalls

def list_calls_with_permissions(file, permission_map_file):
    """ List all API calls which require a permissions in file (according the
        mapping from Felt et al. CSS 2011 in APICalls.txt).
    """

    df = DataFrame.from_csv(permission_map_file, sep='\t')
    a, d, dx = AnalyzeAPK(file)
    for method in d.get_methods():
        for i in method.get_instructions():
            if i.get_name()[:6] == "invoke":
                #get method desc
                call = i.get_output(0).split(',')[-1].strip() 
                #remove return value
                call = call[:call.index(')')+1]  
                #split in class and method
                call = call.split('->') 
                method_class = get_type(call[0])
                ins_method, params = call[1].split('(')
                params = ','.join(parse_parameters(params.replace(')','')))
                apicall = "{0}.{1}({2})".format(method_class, ins_method, params)
                try:
                    print df.ix[apicall]["Permission(s)"]
                    print apicall
                except:
                    pass

def parse_parameters(p):
    """ Parse and format parameters extracted from API
        calls found in smali code
    """
    types = ['S', 'B', 'D', 'F', 'I', 'J', 'Z', 'C']
    parameters = []
    buff = []
    i = 0
    while i < len(p):
        if p[i] == '[':
            buff.append(p[i])
        if p[i] in types:
            buff.append(p[i])
            parameters.append(''.join(buff))
            buff = []
        if p[i] == 'L':
            buff.append(p[i:][:p[i:].index(';')+1])
            parameters.append(''.join(buff))
            i += len(buff[0])-1
            buff = []
        i += 1

    return [ get_type(param) for param in parameters ]


##################################################################################
#                Main function to start encoding of APKs in directory            #    
##################################################################################

if __name__ == "__main__":
    usage = "usage: %prog [<dataset dir>] "
    parser = OptionParser(usage)
    (options, args) = parser.parse_args()

    if len(args) < 1:
        parser.print_help()
        sys.exit(1)

    else:
        sys.exit(process_apk_dir(args[0]))
