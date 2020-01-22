#!/usr/bin/env python
"""
Perform a fit on the data according to some model, as provided.
"""

import matplotlib.pyplot as plt
import numpy
import pickle
import scipy
import scipy.optimize
import sys

# This import is a bit ugly, but it's less ugly than retyping this every time:
from common import uncert_prop, VAR
from ic_models import fisk_2008_eq38 as model

# This is pretty case-specific right now, but I can blow it up to be more
# general later.

# For right now I'm going to use the hourly-averaged rates, because the time
# base doesn't make a difference because I'll be averaging down along the time
# axis again anyways.



def main(events_file):
    # Open the arrays:
    with open('../data/ic_event_{}_flux.pickle{}'.format(VAR, sys.version_info[0]), 'rb') as fp:
        arrs = pickle.load(fp)

    # Flux:
    flux = arrs['flux']
    # Flux delta:
    dflux = arrs['dflux']
    # Datetime/epoch:
    epoch = arrs['epoch']
    # Energy bins:
    energy = arrs['energy']

    # Open the saved event times:
    with open(events_file, 'rb') as fp:
        e = pickle.load(fp)
        # Note that e is like: [((t0_start, y0_start), (t0_stop, y0_stop)),
        #                       ((t1_start, y1_start), (t1_stop, y1_stop)), ...]

    # Now, for each event (row) in e:
    for event in e:
        # Cut all vars in time that we're looking at:
        startidx = numpy.searchsorted(epoch, event[0, 0])
        stopidx = numpy.searchsorted(epoch, event[1, 0])

        cflux = flux[startidx:stopidx]
        cdflux = dflux[startidx:stopidx]
        cenergy = energy[startidx:stopidx]

        # Now average flux over the time that we're interested in:
        cflux = numpy.nanmean(cflux, axis=0)
        cdflux = uncert_prop(cdflux, axis=0)

        # Make sure energy is consistent within this time range.
        # This is a different condition than tried in the mean-concat script,
        # because that checks within hours.  This checks between hours
        # (and therefore, possibly files as well).
        assert numpy.all((cenergy == cenergy[0, 0, :])[~numpy.isnan(cenergy)])
        # If this works (because we have the same binning in the file)
        # we can just use the first set of not-all-NaNs bins
        # (because they're all the same):
        nonnan = numpy.where(numpy.any(~numpy.isnan(cenergy[:, 0]), axis=1) \
                             == True)[0][0]
        cenergy = cenergy[nonnan, 0, :]

        # Lastly, we can count the number of non-NaNs we have, so we know where
        # to stop the array trunking (so we don't try to plot NaNs):
        lenn = numpy.sum(~numpy.isnan(cenergy))

        # We now have an array with 15 values, and an array with the uncertainties
        # in those values.  We should be able to fit this now.
        popt, pcov = scipy.optimize.curve_fit(model,
                                              cenergy[:lenn],  # ONLY fitting first 9 energies!
                                              cflux[:lenn],  # That's because that's about the range that we have any sensible data for.
                                              sigma=cdflux[:lenn],
                                              absolute_sigma=True)
        # I believe we DO in fact have absolute sigma, correct?  (See note
        # about this.)
        # And just plot it for now:
        plt.plot(energy, cflux, 'ro')
        energy_range = numpy.linspace(0, 300, 100)

        plt.plot(energy_range, model(energy_range, *popt))
        plt.xlim((50, 300))
        plt.xlabel('Energy (keV)')
        plt.ylabel('j')


if __name__ == "__main__":
    main(events_file=sys.argv[1])