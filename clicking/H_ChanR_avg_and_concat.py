#!/usr/bin/env python
"""
Script to run through EPILo files and average down H_ChanR|P|T
variables along some axis.
"""

# There are ie ic pe pc files
# Looks like I should use 'ic' files

import datetime
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

flux_mean = numpy.empty(shape=(0, 15), dtype=numpy.float)
dflux_mean = numpy.empty(shape=(0, 15), dtype=numpy.float)
epoch_mean = numpy.empty(0, dtype=datetime.datetime)

def uncert_prop(inarr, axis):
    """ Propagate the uncertainty of numbers on some axis when averaging down
    along that axis. """
    # Uncertainty carries like addition of quadrature divided by number of
    # things.
    return numpy.sqrt(numpy.sum(inarr**2, axis=axis)) / \
        numpy.sum(numpy.invert(numpy.isnan(inarr)), axis=axis)
    # Alternatively, we may need to use:
    # return numpy.sqrt(numpy.sum(inarr**2, axis=axis))

# This is so nanmean doesn't give us "RuntimeWarning: Mean of empty slice"
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

    for f in files:
        print('Starting file {}...'.format(os.path.basename(f)))

        # Open each file:
        f = spacepy.pycdf.CDF(f)
        flux = f['H_Flux_ChanR'][:, :, :15]  # ChanR only includes 15 values
        flux[flux < 0] = numpy.nan  # Cut out fill...
        dflux = f['H_Flux_ChanR_DELTA'][:, :, :15]
        dflux[dflux < 0] = numpy.nan
        # Not sure if we need this:
        # dflux[flux < 0] = numpy.nan

        # But replace bad bins (at particular elevations) with nan's:
        for e in range(6):
            flux[:, elevation == e, mincut[e]:maxcut[e]] = numpy.nan
            dflux[:, elevation == e, mincut[e]:maxcut[e]] = numpy.nan

        # Get the epoch:
        epoch = f['Epoch_ChanR'][...]

        if len(epoch) > 0:  # Apparently there are some files where we have nothing...
            for i in range(INT_PER_DAY):
                # Get slicing indices for the time we're looking at:
                starttime = epoch[0].replace(hour=0, minute=0, second=0, microsecond=0) + i * datetime.timedelta(days=1) / INT_PER_DAY
                stoptime = starttime + datetime.timedelta(days=1) / INT_PER_DAY
                startidx = numpy.searchsorted(epoch, starttime)
                stopidx = numpy.searchsorted(epoch, stoptime)

                # Here we average over look direction (axis 1) and time (axis 0):
                flux_mean = numpy.concatenate([flux_mean, numpy.reshape(numpy.nanmean(numpy.nanmean(flux[startidx:stopidx], axis=1), axis=0),
                                                                        (1, 15))])
                dflux_mean = numpy.concatenate([dflux_mean, numpy.reshape(uncert_prop(uncert_prop(dflux[startidx:stopidx], 1), 0),
                                                                          (1, 15))])
                epoch_mean = numpy.concatenate([epoch_mean, [starttime + datetime.timedelta(days=1) / INT_PER_DAY / 2]])

# This will only work with the same version of python as when used with this script:
with open('datetime_and_flux.pickle{}'.format(sys.version_info[0]), 'wb') as fp:
    pickle.dump({'flux':flux_mean, 'dflux':dflux_mean, 'epoch':epoch_mean}, fp)
