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

from common import uncert_prop
import isois
import spacepy.pycdf

# How many intervals over which to average per day,
# WE ASSUME EVERY FILE IS ONLY 24 HOURS!
INT_PER_DAY = 24
# For minutes, for example, this would be 1440.



def main():
    files = isois.get_latest('psp_isois-epilo_l2-ic')
    # Some test cases:
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:1]
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:10] # just first 10
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:100]

    # Initialize these as the right size to avoid copying the array every time:
    maxn = len(files) * INT_PER_DAY
    flux_mean = numpy.empty(shape=(maxn, 15), dtype=numpy.float)
    dflux_mean = numpy.empty(shape=(maxn, 15), dtype=numpy.float)
    epoch_mean = numpy.empty(maxn, dtype=datetime.datetime)

    # This is so nanmean doesn't give us "RuntimeWarning: Mean of empty slice"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Counter, so we know what hole in the arrays we're at:
        j = 0

        for f in files:
            print('Starting file {}...'.format(os.path.basename(f)))

            # Open each file:
            f = spacepy.pycdf.CDF(f)
            flux = f['H_Flux_ChanR'][:, :, :15]  # ChanR only includes 15 values
            flux[flux < 0] = numpy.nan  # Cut out fill...
            dflux = f['H_Flux_ChanR_DELTA'][:, :, :15]
            dflux[dflux < 0] = numpy.nan

            # Get the epoch:
            epoch = f['Epoch_ChanR'][...]

            if len(epoch) > 0:  # Apparently there are some files where we have nothing...
                for i in range(INT_PER_DAY):
                    # Get slicing indices for the time we're looking at:
                    starttime = epoch[0].replace(hour=0, minute=0, second=0, microsecond=0) \
                        + i * datetime.timedelta(days=1) / INT_PER_DAY
                    stoptime = starttime + datetime.timedelta(days=1) / INT_PER_DAY
                    startidx = numpy.searchsorted(epoch, starttime)
                    stopidx = numpy.searchsorted(epoch, stoptime)

                    # Here we average over look direction (axis 1) and time (axis 0):
                    flux_mean[j] = numpy.reshape(numpy.nanmean(numpy.nanmean(flux[startidx:stopidx], axis=1),
                                                               axis=0), (1, 15))
                    dflux_mean[j] = numpy.reshape(uncert_prop(uncert_prop(dflux[startidx:stopidx], 1), 0),
                                                  (1, 15))
                    epoch_mean[j] = starttime + datetime.timedelta(days=1) / INT_PER_DAY / 2

                    # Step the counter:
                    j += 1

    # This will only work with the same version of python as when used with this script:
    with open('../data/ic_event_datetime_flux.pickle{}'.format(sys.version_info[0]), 'wb') as fp:
        pickle.dump({'flux':flux_mean[:j], 'dflux':dflux_mean[:j], 'epoch':epoch_mean[:j]}, fp)

if __name__ == "__main__":
    main()
