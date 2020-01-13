#!/usr/bin/env python
"""
Script to run through EPILo files and average down H_ChanR|P|T
variables along some axis.
"""

# There are ie ic pe pc files
# Looks like I should use 'ic' files

import datetime
import json
import numpy
import os
import pickle
import sys
import warnings

import isois
import spacepy.pycdf

# How many intervals over which to average per day,
# WE ASSUME EVERY FILE IS ONLY 24 HOURS!
INT_PER_DAY = 24
# For minutes, for example, this would be 1440.

files = isois.get_latest('psp_isois-epilo_l2-ic')
# Some test cases:
# files = isois.get_latest('psp_isois-epilo_l2-ic')[:1]
# files = isois.get_latest('psp_isois-epilo_l2-ic')[:10] # just first 10

elevation = numpy.choose(numpy.arange(80) % 10, (0, 1, 1, 0, 2, 2, 3, 3, 4, 5))
# Where look directions have bad data (spurious TOF signal):
# e0: bad at 460:3000 keV
# e1: bad at 680:2600 keV
# e2: bad at 700:6300 keV
# e3: bad at 450:2400 keV
# e4: bad at 265:950 keV
# e5: bad at 180:820 keV
mincut = [9, 10, 10, 9, 8, 7]
maxcut = [13, 13, 14, 12, 11, 11]

davg = numpy.empty(shape=(0, 15), dtype=numpy.float)
davg_epoch = numpy.empty(0, dtype=datetime.datetime)

# This is so nanmean doesn't give us "RuntimeWarning: Mean of empty slice"
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for f in files:
        print('Starting file {}...'.format(os.path.basename(f)))
        f = spacepy.pycdf.CDF(f)
        flux = f['H_Flux_ChanR'][:, :, 0:15]  # ChanR only includes 15 values
        flux[flux < 0] = numpy.nan  # Cut out fill...
        for e in range(6):
            flux[:, elevation == e, mincut[e]:maxcut[e]] = numpy.nan
        epoch = f['Epoch_ChanR'][...]
        if len(epoch) > 0:  # Apparently there are some files where we have nothing...
            for i in range(INT_PER_DAY):
                starttime = epoch[0].replace(hour=0, minute=0, second=0, microsecond=0) + i * datetime.timedelta(days=1) / INT_PER_DAY
                stoptime = starttime + datetime.timedelta(days=1) / INT_PER_DAY
                startidx = numpy.searchsorted(epoch, starttime)
                stopidx = numpy.searchsorted(epoch, stoptime)
                # Here we average over look direction (axis 1) and time (axis 0):
                davg = numpy.concatenate([davg, numpy.reshape(numpy.nanmean(numpy.nanmean(flux[startidx:stopidx], axis=1),axis=0),
                                                                   (1, 15))])
                davg_epoch = numpy.concatenate([davg_epoch, [starttime + datetime.timedelta(days=1) / INT_PER_DAY / 2]])

# This will only work with the same version of python as when used with this script:
with open('datetime_and_flux.pickle{}'.format(sys.version_info[0]), 'wb') as fp:
    pickle.dump({'flux':davg, 'epoch':davg_epoch}, fp)
