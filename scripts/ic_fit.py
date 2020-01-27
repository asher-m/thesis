#!/usr/bin/env python
"""
Perform a fit on the data according to some model, as provided.
"""

import argparse
import matplotlib.pyplot as plt
import numpy
import pickle
import scipy
import scipy.optimize
import sys

# This import is a bit ugly, but it's less ugly than retyping this every time:
from common import uncert_prop, VAR, FIT_TRUNK_LOWER, FIT_TRUNK_UPPER
from ic_models import fisk_2008_eq38_modified_centered as model

# This is pretty case-specific right now, but I can blow it up to be more
# general later.

# For right now I'm going to use the hourly-averaged rates, because the time
# base doesn't make a difference because I'll be averaging down along the time
# axis again anyways.

XLIM_LOWER = 30
XLIM_UPPER = 500
YLIM_LOWER = 1e-3
YLIM_UPPER = 10


def main(events_file):
    # Open the arrays:
    with open('../data/ic_event_{}_flux.pickle{}'\
              .format(VAR, sys.version_info[0]),
              'rb') as fp:
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
    for i, event in enumerate(e):
        # Cut all vars in time that we're looking at:
        startidx = numpy.searchsorted(epoch, event[0, 0])
        stopidx = numpy.searchsorted(epoch, event[1, 0])

        cflux = flux[startidx:stopidx]
        cdflux = dflux[startidx:stopidx]
        cenergy = energy[startidx:stopidx]

        # Need to flip all of these if we're working with ChanT:
        if VAR == 'ChanT':
            cflux = cflux[:, ::-1]
            cdflux = cdflux[:, ::-1]
            cenergy = cenergy[:, ::-1]

        # Now average flux over the time that we're interested in:
        cflux = numpy.nanmean(cflux, axis=0)
        cdflux = uncert_prop(cdflux, axis=0)

        # import pdb; pdb.set_trace()

        # Make sure energy is consistent within this time range.
        # This is a different condition than tried in the mean-concat script,
        # because that checks within hours.  This checks between hours
        # (and therefore, possibly files as well).
        # ALSO, we don't need to index along look direction axis because we
        # already checked in the mean-concat script.
        assert numpy.all((cenergy == cenergy[0, :])[~numpy.isnan(cenergy)])
        # If this works (because we have the same binning in the file)
        # we can just use the first set of not-all-NaNs bins
        # (because they're all the same):
        nonnan = numpy.where(numpy.any(~numpy.isnan(cenergy), axis=1) \
                             == True)[0][0]
        cenergy = cenergy[nonnan]

        # Lastly, we can count the number of non-NaNs we have, so we know where
        # to stop the array trunking (so we don't try to plot NaNs):
        lenn = numpy.sum(~numpy.isnan(cenergy))
        # And get the first not-nan so we know where to cut off the first few
        # NaNs, (for ChanT, for example):
        first_nonnan = numpy.where(~numpy.isnan(cenergy) == True)[0][0]

        # THEN cut down to the energies we're interested in studying:
        e_startidx = numpy.searchsorted(cenergy[first_nonnan:first_nonnan+lenn],
                                        FIT_TRUNK_LOWER)
        e_stopidx = numpy.searchsorted(cenergy[first_nonnan:first_nonnan+lenn],
                                       FIT_TRUNK_UPPER)

        # Set this up now in case the optimization doesn't fail:
        plt.figure(figsize=(10, 8))

        try:
            # We now have an array with 15 values, and an array with the uncertainties
            # in those values.  We should be able to fit this now.
            popt, pcov = scipy.optimize.curve_fit(model,
                                                  cenergy[first_nonnan:first_nonnan+lenn][e_startidx:e_stopidx],
                                                  cflux[first_nonnan:first_nonnan+lenn][e_startidx:e_stopidx],
                                                  sigma=cdflux[first_nonnan:first_nonnan+lenn][e_startidx:e_stopidx],
                                                  absolute_sigma=True)
            # I believe we DO in fact have absolute sigma, correct?  (See note
            # about this.)

            energy_range = numpy.logspace(numpy.log10(XLIM_LOWER),
                                          numpy.log10(XLIM_UPPER),
                                          1000)
            fmtstr = 'Model params [' + '{:4G}, ' * (len(popt) - 1) + '{:4G}' + ']'
            plt.plot(energy_range,
                     model(energy_range, *popt),
                     label=fmtstr.format(*popt))
        except:
            print('='*80)
            print('{:^80}'.format('Something failed on optimization {}.'.format(i)))
            print('='*80)

        # And just plot it for now:
        # Plot the points used for fit:
        plt.plot(cenergy[first_nonnan:first_nonnan+lenn][e_startidx:e_stopidx],
                 cflux[first_nonnan:first_nonnan+lenn][e_startidx:e_stopidx],
                 'k.',
                 label='Points used for fit')
        # Don't cut down energies because we don't care about just displaying:
        plt.errorbar(cenergy[first_nonnan:first_nonnan+lenn],
                     cflux[first_nonnan:first_nonnan+lenn],
                     yerr=cdflux[first_nonnan:first_nonnan+lenn],
                     color='red')

        plt.xlim((XLIM_LOWER, XLIM_UPPER))
        plt.ylim((YLIM_LOWER, YLIM_UPPER))

        plt.xscale('log')
        plt.xlabel('Energy (keV)')

        plt.yscale('log')
        plt.ylabel('j')

        plt.legend(loc=1,
                   prop={'family':'monospace'})

        plt.title('{} Event: {} to {}'\
                  .format(VAR,
                          event[0, 0].strftime('%F %H%M'),
                          event[1, 0].strftime('%F %H%M'),
                          )
                  )

        plt.tight_layout()

        plt.savefig('../figures/spectrum_{}_{:02d}.png'.format(VAR, i),
                    dpi=300)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(help='events definition file (from clickthrough)',
                        dest='events_file',
                        action='store')
    args = parser.parse_args()
    main(args.events_file)
