#!/usr/bin/env python
"""
Perform a fit on the data according to some model, as provided.
"""

import argparse
import matplotlib.pyplot as plt
import numpy
import pickle
import sys
import types

# This import is a bit ugly, but it's less ugly than retyping this every time:
from common import fit, fit_prep, \
    PLOTTING_XLIM_LOWER, PLOTTING_XLIM_UPPER, PLOTTING_YLIM_LOWER, \
    PLOTTING_YLIM_UPPER, REDS_RAW, GREENS_RAW, BLUES_RAW, PLOTTING_FIGSIZE
from models import fisk_2008_eq38_modified as model



params = {}
""" Table of parameters to save. """

def main(events_file, varnames):
    # First get the param dictionary:
    global params

    # Open the saved event times:
    with open(events_file, 'rb') as fp:
        e = pickle.load(fp)
        # Note that e is like: [((t0_start, y0_start), (t0_stop, y0_stop)),
        #                       ((t1_start, y1_start), (t1_stop, y1_stop)), ...]

    # Now, for each event (row) in e:
    for i, event in enumerate(e):
        # Cut all vars in time that we're looking at:
        starttime = event[0, 0]
        stoptime = event[1, 0]

        # Set figsize:
        plt.figure(figsize=PLOTTING_FIGSIZE)

        # Make the energy range for the spectrum/fit plotting so we don't have
        # to remake it every time:
        energy_range = numpy.logspace(numpy.log10(PLOTTING_XLIM_LOWER),
                                      numpy.log10(PLOTTING_XLIM_UPPER),
                                      1000)

        for j, varname in enumerate(varnames):
            # Set the colorset we want to use for this:
            colors = (REDS_RAW, BLUES_RAW, GREENS_RAW)[j % 3]

            # Only plot this spectrum if we have at least one fit:
            plot_this_spectrum = False
            # Open the arrays:
            with open('../data/flux_event_{}.pickle{}'\
                      .format(varname, sys.version_info[0]),
                      'rb') as fp:
                arrs = pickle.load(fp)
            flux = types.SimpleNamespace(**arrs)

            # Do the fit here:
            fits = fit(varname, model, flux.epoch, flux.energy, flux.flux,
                       flux.dflux, starttime, stoptime, i)

            for k, f in enumerate(fits):
                # We must have at least one fit, so okay:
                plot_this_spectrum = True

                humanname, e, f, df, popt, pcov = f

                p = plt.plot(energy_range,
                             model(energy_range, *popt),
                             label='{}: exp: {:6f}'.format(varname,
                                                            popt[-1]),
                             color=colors[k])

                # Also plot the points used for the fit:
                plt.plot(e, f, "o", color=p[-1].get_color())

            if plot_this_spectrum is True:
                # Also plot the spectrum:
                startidx = numpy.searchsorted(flux.epoch, starttime)
                stopidx = numpy.searchsorted(flux.epoch, stoptime)
                # Do all the cutting and flipping and stuff:
                cenergy, cflux, cdflux = fit_prep(varname,
                                                  startidx,
                                                  stopidx,
                                                  flux.energy,
                                                  flux.flux,
                                                  flux.dflux)
                # Don't cut down energies because we don't care about just displaying:
                plt.errorbar(cenergy,
                             cflux,
                             yerr=cdflux,
                             label='{}'.format(varname),
                             color=colors[-1])

        plt.xlim((PLOTTING_XLIM_LOWER, PLOTTING_XLIM_UPPER))
        plt.ylim((PLOTTING_YLIM_LOWER, PLOTTING_YLIM_UPPER))
        plt.xscale('log')
        plt.xlabel('Energy (keV)')
        plt.yscale('log')
        plt.ylabel('j')
        plt.legend(loc=1, prop={'family':'monospace'})
        plt.title('Event ID {:02d}: {} to {}'\
                  .format(i,
                          event[0, 0].strftime('%F %H%M'),
                          event[1, 0].strftime('%F %H%M')))
        plt.tight_layout()
        plt.savefig('../figures/spectrum_{:02d}.png'.format(i))
        plt.savefig('../figures/pdf/spectrum_{:02d}.pdf'.format(i))
        # plt.show()
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(help='events definition file (from clickthrough)',
                        dest='events_file',
                        action='store')
    parser.add_argument(help='name of the variable to fit, can be'
                        ' \'ChanP\', \'ChanT\', \'ChanR\', \'all\', or some '
                        'combination of these, (not including \'all\')',
                        dest='varnames',
                        action='store',
                        nargs='+')
    args = parser.parse_args()
    for n in args.varnames:
        # If this failed, there's a typo in your command line args:
        assert n in ('ChanP', 'ChanT', 'ChanR', 'all')
    if args.varnames[0] != 'all':
        main(args.events_file, args.varnames)
    else:
        main(args.events_file, ('ChanP', 'ChanT', 'ChanR'))
