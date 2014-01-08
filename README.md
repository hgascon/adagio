adagio
======

### Structural Analysis and Detection of Android Malware

Adagio contains several modules that implement the method described in the
paper:

**[Structural Detection of Android Malware using Embedded Call Graphs](http://user.informatik.uni-goettingen.de/~hgascon/docs/2013b-aisec.pdf)**
Hugo Gascon, Fabian Yamaguchi, Daniel Arp, Konrad Rieck
*ACM Workshop on Security and Artificial Intelligence (AISEC) November 2013*

These modules allow to extract and label the call graphs from a series of
Android APKs or DEX files and apply an explicit feature map that captures
their structural relationships. The analysis module provides classes to desing a binary
or multiclass classification experiment using the vectorial representation and
support vector machines.

In order to use the code, you will need to install the following dependencies:

* [Androguard](https://code.google.com/p/androguard/), the reverse engineering toolkit for Android.
* [scikit-learn](http://scikit-learn.org/stable/), the awesome python toolbox for machine learning.
* [networkx](http://networkx.github.io), a python package for manipulation of complex networks.

