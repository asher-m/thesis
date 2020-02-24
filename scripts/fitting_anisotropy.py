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



def main(events_file, mag_file, varnames):
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

        fig, axes = plt.subplots(nrows=3, ncols=1,
                                 figsize=(PLOTTING_FIGSIZE[0],
                                          PLOTTING_FIGSIZE[1]*3),
                                 sharex=True, sharey=True)
        # import pdb; pdb.set_trace()

        # Make the energy range for the spectrum/fit plotting so we don't have
        # to remake it every time:
        energy_range = numpy.logspace(numpy.log10(PLOTTING_XLIM_LOWER),
                                      numpy.log10(PLOTTING_XLIM_UPPER),
                                      1000)

        for j, varname in enumerate(varnames):
            # Set the colorset we want to use for this:
            colors = (REDS_RAW, BLUES_RAW, GREENS_RAW)[j % 3]

            # Open the arrays:
            with open('../data/flux_event_{}.pickle{}'\
                      .format(varname, sys.version_info[0]),
                      'rb') as fp:
                arrs = pickle.load(fp)
            flux = types.SimpleNamespace(**arrs)

            fits = {
                    'Parallel': [fit(varname, model, flux.epoch, flux.energy,
                                     flux.flux_par, flux.dflux_par, starttime,
                                     stoptime, i, namesauce='parallel'),
                                 flux.flux_par, flux.dflux_par],
                    'Perpendicular': [fit(varname, model, flux.epoch, flux.energy,
                                         flux.flux_perp, flux.dflux_perp,
                                         starttime, stoptime, i,
                                         namesauce='perpendicular'),
                                      flux.flux_perp, flux.dflux_perp],
                    'Antiparallel': [fit(varname, model, flux.epoch, flux.energy,
                                         flux.flux_apar, flux.dflux_apar,
                                         starttime, stoptime, i,
                                         namesauce='antiparallel'),
                                     flux.flux_apar, flux.dflux_apar]
                    }

            for s, tt in enumerate(fits.items()):
                here_title, fff = tt
                # I'm so sorry...
                ax = axes[s]
                ax.set_title('Event ID {:02d}: {}'.format(i, here_title))
                ff = fff[0]
                here_flux = fff[1]
                here_dflux = fff[2]
                for k, f in enumerate(ff):
                    humanname, e, f, df, popt, pcov = f

                    p = ax.plot(energy_range,
                                model(energy_range, *popt),
                                label='{}: exp: {:4f}'.format(varname,
                                                              popt[-1]),
                                color=colors[k])

                    # Also plot the points used for the fit:
                    ax.plot(e, f, "o", color=p[-1].get_color())

                # Also plot the spectrum:
                startidx = numpy.searchsorted(flux.epoch, starttime)
                stopidx = numpy.searchsorted(flux.epoch, stoptime)
                # Do all the cutting and flipping and stuff:
                cenergy, cflux, cdflux = fit_prep(varname,
                                                  startidx,
                                                  stopidx,
                                                  flux.energy,
                                                  here_flux,
                                                  here_dflux)
                # Don't cut down energies because we don't care about just displaying:
                ax.errorbar(cenergy,
                            cflux,
                            yerr=cdflux,
                            label='{}'.format(varname),
                            color=colors[-1])

                ax.legend(loc=1, prop={'family':'monospace'})
                # Nice to have on every plot:
                ax.set_ylabel('j')

            # Sharing axes, so:
            ax.set_xlim((PLOTTING_XLIM_LOWER, PLOTTING_XLIM_UPPER))
            ax.set_xscale('log')
            ax.set_xlabel('Energy (keV)')
            ax.set_ylim((PLOTTING_YLIM_LOWER, PLOTTING_YLIM_UPPER))
            ax.set_yscale('log')

        fig.tight_layout()
        fig.savefig('../figures/aspectrum_{:02d}.png'.format(i))
        fig.savefig('../figures/pdf/aspectrum_{:02d}.pdf'.format(i))
        # plt.show()
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(help='events definition file (from clickthrough)',
                        dest='events_file',
                        action='store')
    parser.add_argument(help='mag file from field concat script',
                        dest='mag_file',
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
        main(args.events_file, args.mag_file, args.varnames)
    else:
        main(args.events_file, args.mag_file, ('ChanP', 'ChanT', 'ChanR'))
