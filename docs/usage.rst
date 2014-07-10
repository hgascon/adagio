.. Adagio documentation master file, created by
   sphinx-quickstart on Thu Jul 10 17:23:44 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _usage:

=====
Usage
=====

Start by creating a couple of directories, one for malware and another for clean apps. Copy your samples to these directories and then use FCGextractor to generate the function call graphs::

     $ ./FCGextractor.py malware_dir
     $ ./FCGextractor.py clean_dir

For each one of the samples, a file named with the **SHA256** of the binary and extension **fcg.pz** will be generated::

    $ ls malware_dir
    ec0e0d25aa1de4f38894fb1999d6f21535610ffba15423a02ec993fea1561c66.fcg.pz
    eeea9fb531c7d24fe5edcbbb039bf4f19dff285447cdc13f15260145c40b89c8.fcg.pz
    f01d9a2d49f49d28aaf222d7c5ba855648f46768a403808e652ee1d856c46076.fcg.pz
    f18891b20623ad35713e7f44feade51a1fd16030af55056a45cefa3f5f38e983.fcg.pz
    f391cc4ea5961d649bc62a0466560dc76eaebcf26f0c8452c671c2d2b34361b8.fcg.pz
    f6239ba0487ffcf4d09255dba781440d2600d3c509e66018e6a5724912df34a9.fcg.pz
    f7c36355c706fc9dd8954c096825e0613807e0da4bd7f3de97de0aec0be23b79.fcg.pz
    f9a57ca11d800dbcecc9042f056e202f697da89d7fd117779ef73af35a8580b5.fcg.pz
    fca4e19344883ab94c6ac8a45a1d93f3a06920028d2aa660bfb217b5aabb3874.fcg.pz
    fd0eccda9d8ee948c8f72c68e3d72f7e6bf2043e7bdb7213e6c6a6c29110621e.fcg.pz
    fe1b0a59d0039683d007de6a1851b08c34a8dfd76765ec9d9388cbb670d6d6ab.fcg.pz

The extension **pz** indicates that the file is a *pickled* and *gzipped* object that can be loaded using the pz.py module. The **fcg** extension indicates that the object is a function call graph object. Each **fcg** object is basically a Networkx_ directed graph where each node corresponds to a method in the decompiled DEX code. Additionally, a 15 bit label property is assigned to each node. You can read the paper to know how this binary vector is generated according to the decompiled instructions found in each method.

.. _Networkx: http://networkx.github.io

In order to use the analysis modules in a interactive way, the best option is iPython_.

.. _iPython: http://ipython.org/

