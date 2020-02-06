#!/usr/bin/env python
"""
Script to run through FIELDS magnetic field files and concat all the data into
one big file.  (This makes my life easier and faster later.)
"""

import datetime
import glob
import numpy
import os
import pickle
import sys

import spacepy.pycdf



MAGGLB = 'psp_fld_l2_mag_SC_1min_*'
""" Glob expression to hand to glob to find the mag files. """
MAGDIR = '/home/share/data/FIELDS_shared/level2'
""" Dir to append to front of glob expression to find the mag files. """
MAGVNM = 'psp_fld_l2_mag_SC'
""" Name of the magnetic field variable. """
MAGEPC = 'epoch_mag_SC'
""" Name of the magnetic field epoch variable. """
MAGNUM = 1440
""" **MAX** number of magnetic field entries per day.  (Ie., 1 per minute,
or 1440/day). """

def main():
    # Get the files from glob:
    files = glob.glob(os.path.join(MAGDIR, MAGGLB))
    # Some test cases:
    # files = glob.glob(os.path.join(MAGDIR, MAGGLB)[:1]
    # files = glob.glob(os.path.join(MAGDIR, MAGGLB)[:10] # just first 10
    # files = glob.glob(os.path.join(MAGDIR, MAGGLB)[:100]

    # Sort the files because glob doesn't sort them:
    files.sort()

    with spacepy.pycdf.CDF(files[0]) as f:
        lenn = f[MAGVNM][...].shape[1]
        # This should always be 3, but check anyways:
        assert lenn == 3

    # Initialize these as the right size to avoid copying the array every time:
    maxn = len(files) * MAGNUM
    cmag = numpy.empty(shape=(maxn, lenn), dtype=numpy.float)
    cepoch = numpy.empty(shape=maxn, dtype=datetime.datetime)
    
    # Counter, so we know what hole in the arrays we're at:
    j = 0

    import pdb; pdb.set_trace()

    for ff in files:
        print('Starting file {}...'.format(os.path.basename(ff)))

        # Open each file:
        with spacepy.pycdf.CDF(ff) as f:
            # Get how many entries we have in this file:
            flenn = f[MAGVNM].shape[0]
            # Now get the actual data:
            mag = f[MAGVNM][...]
            epoch = f[MAGEPC][...]
            
            # Fill in the output array with what we have:
            cmag[j:j+flenn] = mag
            cepoch[j:j+flenn] = epoch
        
        j += flenn

    # This will only work with the same version of python as when used with this script:
    with open('../data/magfield.pickle{}'.format(sys.version_info[0]),
              'wb') as fp:
        pickle.dump({'mag':cmag[:j],
                     'epoch':cepoch[:j]},
                    fp)

if __name__ == "__main__":
    main()
