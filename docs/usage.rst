.. Adagio documentation master file, created by
   sphinx-quickstart on Thu Jul 10 17:23:44 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _usage:

=====
Usage
=====

Start by creating a couple of directories, one for malware and another for clean apps. Copy your samples to these directories and then use FCGextractor to generate the function call graphs:
 
     $> ./FCGextractor.py <malware_dir>
     $> ./FCGextractor.py <clean_dir>

For each one of the samples, a file named with the **SHA256** of the binary and extension **fcg.pz** will be generated. The extension **pz** indicates that the file is a *pickled* and *gziped* object that can be loaded using the pz.py module. The **fcg** extension indicates that the object is a function call graph object. Each **fcg** object is basically a Networkx_ directed graph where each node corresponds to a method in the decompiled DEX code. Additionally, a 15 bit label property is assigned to each node. You can read the paper to know how this binary vector is generated according to the decompiled instructions found in each method.

.. _Networkx: http://networkx.github.io

In order to use the analysis modules in a interactive way, the best option is iPython_.

.. _iPython: http://ipython.org/

