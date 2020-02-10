#!/usr/bin/env python
"""
Script to run through EPILo files and average down H_ChanR|P|T
variables along some axis.
"""

# There are ie ic pe pc files
# Looks like I should use 'ic' files

import argparse
import datetime
import numpy
import os
import pickle
import sys
import warnings

from common import uncert_prop
import isois
import spacepy.pycdf



def main():
    files = isois.get_latest('psp_isois-epilo_l2-ic')
    # Some test cases:
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:1]
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:10] # just first 10
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:100]

    # It's faster (at least to program) to initialize these as the python
    # objects and then go to numpy...at least this way I can't see the
    # copying in memory...
    epoch = []
    R = []

    for ff in files:
        print('Starting file {}...'.format(os.path.basename(ff)))

        # Open each file:
        with spacepy.pycdf.CDF(ff) as f:
            # Get the file's epoch:
            fEpoch = f['Epoch_ChanD'][...]
            # Get the actual radial values:
            fR = f['HGC_R_ChanD'][...]

            if len(fEpoch) > 0:
                epoch.extend(fEpoch.tolist())
                R.extend(fR.tolist())

    # Finally cast everything back to numpy arrays:
    epoch = numpy.array(epoch)
    R = numpy.array(R)

    # This will only work with the same version of python as when used with this script:
    with open('../data/R.pickle{}'.format(sys.version_info[0]),
              'wb') as fp:
        pickle.dump({'R':R,
                     'epoch':epoch},
                    fp)

if __name__ == "__main__":
    main()
