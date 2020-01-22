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

# This can be ChanR, ChanT, or ChanP
VAR = "ChanR"


def main():
    files = isois.get_latest('psp_isois-epilo_l2-ic')
    # Some test cases:
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:1]
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:10] # just first 10
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:100]

    # Need to maximum number of possible bins so we can set the array size
    # correctly
    # NOTE: Due to inconsistencies (see first/last file binning inconsistency
    # issue), leave this the maximum size possible AND DON'T exclude invalid
    # numbers.  Disciminate in plotting.
    with spacepy.pycdf.CDF(files[0]) as f:
        lenn = f['H_{}_Energy'.format(VAR)][...].shape[2]

    # Initialize these as the right size to avoid copying the array every time:
    maxn = len(files) * INT_PER_DAY
    flux_mean = numpy.empty(shape=(maxn, lenn), dtype=numpy.float)
    dflux_mean = numpy.empty(shape=(maxn, lenn), dtype=numpy.float)
    epoch_mean = numpy.empty(maxn, dtype=datetime.datetime)
    energy_agg = numpy.empty(shape=(maxn, lenn), dtype=numpy.float)

    # This is so nanmean doesn't give us "RuntimeWarning: Mean of empty slice"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Counter, so we know what hole in the arrays we're at:
        j = 0

        for ff in files:
            print('Starting file {}...'.format(os.path.basename(ff)))

            # Open each file:
            with spacepy.pycdf.CDF(ff) as f:
                flux = f['H_Flux_{}'.format(VAR)][...]
                flux[flux < 0] = numpy.nan  # Cut out fill...
                dflux = f['H_Flux_{}_DELTA'.format(VAR)][...]
                dflux[dflux < 0] = numpy.nan
                energy = f['H_{}_Energy'.format(VAR)][...]
                energy[energy < 0] = numpy.nan

                # Get the epoch:
                epoch = f['Epoch_{}'.format(VAR)][...]

                if len(epoch) > 0:  # Apparently there are some files where we have nothing...
                    for i in range(INT_PER_DAY):
                        # Get slicing indices for the time we're looking at:
                        starttime = epoch[0].replace(hour=0, minute=0, second=0, microsecond=0) \
                            + i * datetime.timedelta(days=1) / INT_PER_DAY
                        stoptime = starttime + datetime.timedelta(days=1) / INT_PER_DAY
                        startidx = numpy.searchsorted(epoch, starttime)
                        stopidx = numpy.searchsorted(epoch, stoptime)

                        # We need to check if the energy binning across these times is
                        # consistent so we can in fact average.  If it's not, we need
                        # to be much more clever.
                        try:
                            if startidx < energy.shape[0]:
                                # Need to mask out NaN(s) because no equality:
                                assert numpy.all((energy[startidx:stopidx] \
                                                  == energy[startidx, 0, :]) \
                                                 [~numpy.isnan(energy[startidx:stopidx])])
                        except AssertionError:
                            print('Inconsistent binning in file {} in time ran'
                                  'ge {} to {}'.format(os.path.basename(ff),
                                                       starttime.strftime('%F %H:%M:%S.%f'),
                                                       stoptime.strftime('%F %H:%M:%S.%f')))
                            # Need to break and continue...
                            continue

                        # Here we average over look direction (axis 1) and time (axis 0):
                        flux_mean[j] = numpy.reshape(numpy.nanmean(numpy.nanmean(flux[startidx:stopidx], axis=1),
                                                                   axis=0), (1, lenn))
                        dflux_mean[j] = numpy.reshape(uncert_prop(uncert_prop(dflux[startidx:stopidx], 1), 0),
                                                      (1, lenn))
                        epoch_mean[j] = starttime + datetime.timedelta(days=1) / INT_PER_DAY / 2
                        # If it's constant over the entire file, which we've
                        # checked, we can just use the first one:
                        if startidx < energy.shape[0]:
                            energy_agg[j] = energy[startidx, 0, :]
                        else:
                            energy_agg[j] = numpy.nan

                        # Step the counter:
                        j += 1

    # This will only work with the same version of python as when used with this script:
    with open('../data/ic_event_{}_flux.pickle{}'.format(VAR,
                                                         sys.version_info[0]),
              'wb') as fp:
        pickle.dump({'flux':flux_mean[:j],
                     'dflux':dflux_mean[:j],
                     'epoch':epoch_mean[:j],
                     'energy':energy_agg[:j]},
                    fp)

if __name__ == "__main__":
    main()
