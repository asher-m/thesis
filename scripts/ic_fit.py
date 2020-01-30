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
from common import uncert_prop, nan_cut, energy_trunc, PLOTTING_XLIM_LOWER, \
    PLOTTING_XLIM_UPPER, PLOTTING_YLIM_LOWER, PLOTTING_YLIM_UPPER
from ic_models import fisk_2008_eq38_modified as model

# This is pretty case-specific right now, but I can blow it up to be more
# general later.

# For right now I'm going to use the hourly-averaged rates, because the time
# base doesn't make a difference because I'll be averaging down along the time
# axis again anyways.



def main(events_file, varname):
    # Open the arrays:
    with open('../data/ic_event_{}_flux.pickle{}'\
              .format(varname, sys.version_info[0]),
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

    # Also get the energy trunc's:
    fit_lower_bound, fit_upper_bound = energy_trunc(varname)

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
        if varname == 'ChanT':
            cflux = cflux[:, ::-1]
            cdflux = cdflux[:, ::-1]
            cenergy = cenergy[:, ::-1]

        # Now average flux over the time that we're interested in:
        cflux = numpy.nanmean(cflux, axis=0)
        cdflux = uncert_prop(cdflux, axis=0)

        # Cut down the energy array to just the non-NaN elements (and find
        # first/last indices of non-nan elements for the second axis).
        cenergy, _, first_nonnan, last_nonnan = nan_cut(cenergy)
        # Note: We're throwing away the pre-cutdown array because it makes
        # it more explicit everywhere else.

        # THEN cut down to the energies we're interested in studying:
        e_startidx = numpy.searchsorted(cenergy[first_nonnan:last_nonnan], fit_lower_bound)
        e_stopidx = numpy.searchsorted(cenergy[first_nonnan:last_nonnan], fit_upper_bound)

        # Set this up now in case the optimization doesn't fail:
        plt.figure(figsize=(10, 8))

        try:
            # We now have an array with 15 values, and an array with the uncertainties
            # in those values.  We should be able to fit this now.
            popt, pcov = scipy.optimize.curve_fit(model,
                                                  cenergy[first_nonnan:last_nonnan][e_startidx:e_stopidx],
                                                  cflux[first_nonnan:last_nonnan][e_startidx:e_stopidx],
                                                  sigma=cdflux[first_nonnan:last_nonnan][e_startidx:e_stopidx],
                                                  absolute_sigma=True)
            # I believe we DO in fact have absolute sigma, correct?  (See note
            # about this.)

            energy_range = numpy.logspace(numpy.log10(PLOTTING_XLIM_LOWER),
                                          numpy.log10(PLOTTING_YLIM_UPPER),
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
        plt.plot(cenergy[first_nonnan:last_nonnan][e_startidx:e_stopidx],
                 cflux[first_nonnan:last_nonnan][e_startidx:e_stopidx],
                 'k.',
                 label='Points used for fit')

        # Don't cut down energies because we don't care about just displaying:
        plt.errorbar(cenergy[first_nonnan:last_nonnan],
                     cflux[first_nonnan:last_nonnan],
                     yerr=cdflux[first_nonnan:last_nonnan],
                     color='red')

        plt.xlim((PLOTTING_XLIM_LOWER, PLOTTING_XLIM_UPPER))
        plt.ylim((PLOTTING_YLIM_LOWER, PLOTTING_YLIM_UPPER))

        plt.xscale('log')
        plt.xlabel('Energy (keV)')

        plt.yscale('log')
        plt.ylabel('j')

        plt.legend(loc=1,
                   prop={'family':'monospace'})

        plt.title('{} Event: {} to {}'\
                  .format(varname,
                          event[0, 0].strftime('%F %H%M'),
                          event[1, 0].strftime('%F %H%M'),
                          )
                  )

        plt.tight_layout()

        plt.savefig('../figures/spectrum_{}_{:02d}.png'.format(varname, i),
                    dpi=300)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(help='events definition file (from clickthrough)',
                        dest='events_file',
                        action='store')
    parser.add_argument(help='name of the variable to fit, can be'
                        ' \'ChanP\', \'ChanT\', \'ChanR\', or \'all\'',
                        dest='varname',
                        action='store')
    args = parser.parse_args()
    # Make sure we have something we can look at. If this assertion fails,
    # it's because we received an invalid varname.
    assert args.varname in ('ChanP', 'ChanT', 'ChanR', 'all')
    if args.varname != 'all':
        main(args.events_file, args.varname)
    else:
        for varname in ('ChanP', 'ChanT', 'ChanR'):
            print('='*80)
            print('{:^80}'.format('Starting {}...'.format(varname)))
            print('='*80)
            main(args.events_file, varname)
