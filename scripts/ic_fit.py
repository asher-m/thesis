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
from common import uncert_prop
from ic_mean_concat_clickthrough import energy
from ic_models import fisk_2008_eq38 as model

# This is pretty case-specific right now, but I can blow it up to be more
# general later.

# For right now I'm going to use the hourly-averaged rates, because the time
# base doesn't make a difference because I'll be averaging down along the time
# axis again anyways.



def main(events_file):
    # Open the arrays:
    with open('../data/ic_event_datetime_flux.pickle{}'.format(sys.version_info[0]), 'rb') as fp:
        arrs = pickle.load(fp)

    # Flux:
    flux = arrs['flux']
    # Flux delta:
    dflux = arrs['dflux']
    # Datetime/epoch:
    epoch = arrs['epoch']

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

        # Now average flux over the time that we're interested in:
        cflux = numpy.nanmean(cflux, axis=0)
        cdflux = uncert_prop(cdflux, axis=0)

        # We now have an array with 15 values, and an array with the uncertainties
        # in those values.  We should be able to fit this now.
        popt, pcov = scipy.optimize.curve_fit(model,
                                              energy[:9],  # ONLY fitting first 9 energies!
                                              cflux[:9],  # That's because that's about the range that we have any sensible data for.
                                              sigma=cdflux[:9],
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