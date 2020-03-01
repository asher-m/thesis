#!/usr/bin/env python
"""
Script to run through EPILo files and average down H_ChanR|P|T
variables along some axis.
"""

# There are ie ic pe pc files
# Looks like I should use 'ic' files

import argparse
import copy
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

PAR = (0, 30)
PERP = (75, 105)
APAR = (150, 180)



def main(varname):
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
        lenn = f['H_{}_Energy'.format(varname)][...].shape[2]

    # Initialize these as the right size to avoid copying the array every time:
    maxn = len(files) * INT_PER_DAY
    flux_mean = numpy.empty(shape=(maxn, lenn), dtype=numpy.float)
    # Also break these out so we have parallel and antiparallel:
    flux_mean_par = numpy.empty(shape=(maxn, lenn), dtype=numpy.float)
    flux_mean_perp = numpy.empty(shape=(maxn, lenn), dtype=numpy.float)
    flux_mean_apar = numpy.empty(shape=(maxn, lenn), dtype=numpy.float)
    # Also get deltas:
    dflux_mean = numpy.empty(shape=(maxn, lenn), dtype=numpy.float)
    dflux_mean_par = numpy.empty(shape=(maxn, lenn), dtype=numpy.float)
    dflux_mean_perp = numpy.empty(shape=(maxn, lenn), dtype=numpy.float)
    dflux_mean_apar = numpy.empty(shape=(maxn, lenn), dtype=numpy.float)
    # Then the epoch and energy depend of each of these:
    epoch_mean = numpy.empty(maxn, dtype=datetime.datetime)
    energy_agg = numpy.empty(shape=(maxn, lenn), dtype=numpy.float)

    # Get ready to cut out elevation 5:
    elevation = numpy.choose(numpy.arange(80) % 10, (0, 1, 1, 0, 2, 2, 3, 3, 4, 5))

    # This is so nanmean doesn't give us "RuntimeWarning: Mean of empty slice"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Counter, so we know what hole in the arrays we're at:
        j = 0

        for ff in files:
            print('{}: Starting file {}...'.format(varname, os.path.basename(ff)))

            # Open each file:
            with spacepy.pycdf.CDF(ff) as f:
                flux = f['H_Flux_{}'.format(varname)][...]
                flux[flux < 0] = numpy.nan  # Cut out fill...
                dflux = f['H_Flux_{}_DELTA'.format(varname)][...]
                dflux[dflux < 0] = numpy.nan
                energy = f['H_{}_Energy'.format(varname)][...]
                energy[energy < 0] = numpy.nan
                pa = f['PA_{}'.format(varname)][...]
                pa[pa < 0] = numpy.nan

                # Cut out elevation 5:
                flux[:, elevation == 5, :] = numpy.nan

                # Get the epoch:
                epoch = f['Epoch_{}'.format(varname)][...]

                if len(epoch) > 0:  # Apparently there are some files where we have nothing...
                    for i in range(INT_PER_DAY):
                        # Get slicing indices for the time we're looking at:
                        starttime = epoch[0].replace(hour=0, minute=0, second=0, microsecond=0) \
                            + i * datetime.timedelta(days=1) / INT_PER_DAY
                        stoptime = starttime + datetime.timedelta(days=1) / INT_PER_DAY
                        startidx = numpy.searchsorted(epoch, starttime)
                        stopidx = numpy.searchsorted(epoch, stoptime)

                        # Just continue if we find nothing in this time range:
                        if startidx == stopidx:
                            continue

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
                        flux_mean[j] = numpy.reshape(numpy.nanmean(numpy.nanmean(flux[startidx:stopidx], axis=1), axis=0), (1, lenn))
                        dflux_mean[j] = numpy.reshape(uncert_prop(uncert_prop(dflux[startidx:stopidx], 1), 0), (1, lenn))
                        # Handle all the angled stuff:
                        for ang in (PAR, PERP, APAR):
                            # Make the flux slice:
                            aslice = numpy.logical_or(numpy.logical_and(ang[0] <= pa[startidx:stopidx], pa[startidx:stopidx] <= ang[1]), numpy.isnan(pa[startidx:stopidx]))
                            # While the old way technically worked, this is more
                            # clear and should be "guaranteed" to work.
                            aflux = copy.copy(flux[startidx:stopidx])
                            aflux[~aslice] = numpy.nan
                            adflux = copy.copy(dflux[startidx:stopidx])
                            adflux[~aslice] = numpy.nan
                            flux_mean_par[j] = numpy.reshape(numpy.nanmean(aflux, axis=0), (1, lenn))
                            dflux_mean_par[j] = numpy.reshape(uncert_prop(adflux, 0), (1, lenn))

                        epoch_mean[j] = starttime + datetime.timedelta(days=1) / INT_PER_DAY / 2

                        # If it's constant over the entire file, which we've
                        # checked, we can just use the first one.  Need this
                        # extra check in case we have an empty time period:
                        if startidx < energy.shape[0]:
                            energy_agg[j] = energy[startidx, 0, :]
                        else:
                            energy_agg[j] = numpy.nan

                        # Step the counter:
                        j += 1

    # This will only work with the same version of python as when used with this script:
    with open('../data/flux_event_{}.pickle{}'.format(varname,
                                                         sys.version_info[0]),
              'wb') as fp:
        pickle.dump({'flux':flux_mean[:j],
                     'dflux':dflux_mean[:j],
                     'flux_par':flux_mean_par[:j],
                     'dflux_par':dflux_mean_par[:j],
                     'flux_perp':flux_mean_perp[:j],
                     'dflux_perp':dflux_mean_perp[:j],
                     'flux_apar':flux_mean_apar[:j],
                     'dflux_apar':dflux_mean_apar[:j],
                     'epoch':epoch_mean[:j],
                     'energy':energy_agg[:j]},
                    fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(help='name of the variable to process, can be'
                        ' \'ChanP\', \'ChanT\', \'ChanR\', or \'all\'',
                        dest='varname',
                        action='store')
    args = parser.parse_args()
    # Make sure we have something we can look at. If this assertion fails,
    # it's because we received an invalid varname.
    assert args.varname in ('ChanP', 'ChanT', 'ChanR', 'all')
    if args.varname != 'all':
        main(args.varname)
    else:
        for varname in ('ChanP', 'ChanT', 'ChanR'):
            print('='*80)
            print('Starting {}...'.format(varname))
            print('='*80)
            main(varname)
