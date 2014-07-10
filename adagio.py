#!/usr/bin/python
# ADAGIO Android Application Graph-based Classification
# adagio.py >> Main wrapper for other modules
# Copyright (c) 2013 Hugo Gascon <hgascon@uni-goettingen.de>


import sys
import FCGextractor as fcge
from optparse import OptionParser


if __name__ == "__main__":
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    (options, args) = parser.parse_args()
    parser.add_option("-e", "--extract", type="str",
                      dest="apk_dir", default="",
                      help="dir with apk/dex samples to generate call graphs"
                      )

    if len(args) < 1:
        parser.print_help()
        sys.exit(1)

    else:
        if options.apk_dir:
            fcge.process_apk_dir(options.apk_dir)
