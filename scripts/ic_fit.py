#!/usr/bin/env python
"""
Perform a fit on the data according to some model, as provided.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy
import os
import pickle
import scipy
import scipy.optimize
import sys

# This import is a bit ugly, but it's less ugly than retyping this every time:
from common import uncert_prop, nan_cut, energy_trunc, get_eta_squared, \
    PLOTTING_XLIM_LOWER, PLOTTING_XLIM_UPPER, PLOTTING_YLIM_LOWER, \
    PLOTTING_YLIM_UPPER, FITTING_HOW, REDS_RAW, GREENS_RAW, BLUES_RAW
from ic_models import fisk_2008_eq38_modified as model



param_table = {}
""" Table of parameters to save. """

def fit_prep(varname, startidx, stopidx, energy, flux, dflux):
    """ Function to unify how we're slicing energy, flux, and dflux,
    just so it doesn't need to be copied everywhere."""
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

    return cenergy[first_nonnan:last_nonnan], \
        cflux[first_nonnan:last_nonnan], \
        cdflux[first_nonnan:last_nonnan]

def fit(varname, model, epoch, energy, flux, dflux, starttime, stoptime, idx=None):
    """ Function to perform fit of the data per varname, according to a
    bunch of different options.

    Returns a list like:
        [("<humanname_0>", cenergy_0, cflux_0, cdflux_0, (popt_0...), (pcov_0...)),
         ("<humanname_1>", cenergy_1, cflux_1, cdflux_1, (popt_1...), (pcov_1...)),
         ...]
    """
    # Just change the name here because replacing it is more work:
    how = FITTING_HOW
    # Also make the base 'humanname' prototype to format into:
    humanname_base = 't_range: {}; e_range: {}'
    # And the list of fits that we made:
    fits = []

    for params in how[varname]:
        if 't_range' in params:
            # Adjust the st*times according to how the dict says:
            timedelta = stoptime - starttime
            starttime = starttime + timedelta * params['t_range'][0]
            stoptime = starttime + timedelta * params['t_range'][1]

        # Just hand back what it'd normally be:
        startidx = numpy.searchsorted(epoch, starttime)
        stopidx = numpy.searchsorted(epoch, stoptime)

        # Do all the cutting and flipping and stuff:
        cenergy, cflux, cdflux = fit_prep(varname,
                                          startidx,
                                          stopidx,
                                          energy,
                                          flux,
                                          dflux)

        if 'e_range' in params:
            fit_lower_bound, fit_upper_bound = params['e_range']
        else:
            # Get the energy trunc's (in real numbers, get idx's next):
            fit_lower_bound, fit_upper_bound = energy_trunc(varname)

        # Cut down to the energies we're interested in studying:
        e_startidx = numpy.searchsorted(cenergy, fit_lower_bound)
        e_stopidx = numpy.searchsorted(cenergy, fit_upper_bound)

        try:
            # Now we can go ahead and do the fit:
            popt, pcov = scipy.optimize.curve_fit(model,
                                                  cenergy[e_startidx:e_stopidx],
                                                  cflux[e_startidx:e_stopidx],
                                                  sigma=cdflux[e_startidx:e_stopidx],
                                                  absolute_sigma=True)

        except:
            print('='*80)
            print('{:^80}'.format('Something failed on optimization for event {:02d}'.format(idx)))
            print('{:^80}'.format(' with params {}.'.format(params)))
            print('='*80)
            continue

        # Finally, we can append everything to the list that we found.
        # First make the human name:
        if 't_range' in params:  # Get the t_range.
            t_range = params['t_range']
        else:
            t_range = (0., 1.)
        if 'e_range' in params:  # Get the e_range.
            e_range = params['e_range']
        else:
            e_range = energy_trunc(varname)
        humanname = humanname_base.format(t_range, e_range)
        # We have everything else, so we can just add it to the list:
        fits.append((humanname,
                     cenergy[e_startidx:e_stopidx],
                     cflux[e_startidx:e_stopidx],
                     cdflux[e_startidx:e_stopidx],
                     popt,
                     pcov,
                     ))

    return fits

def main(events_file, mag_file, varnames):
    # First get the param dictionary:
    global param_table

    # Open the saved event times:
    with open(events_file, 'rb') as fp:
        e = pickle.load(fp)
        # Note that e is like: [((t0_start, y0_start), (t0_stop, y0_stop)),
        #                       ((t1_start, y1_start), (t1_stop, y1_stop)), ...]

    with open(mag_file, 'rb') as fp:
        mag = pickle.load(fp)
    mag_str = mag['mag']
    mag_epoch = mag['epoch']

    # Now, for each event (row) in e:
    for i, event in enumerate(e):
        # Cut all vars in time that we're looking at:
        starttime = event[0, 0]
        stoptime = event[1, 0]

        # Do the math for the B field if it's not there yet.  First make
        # key for the event:
        param_table_keyname = "{}_to_{}".format(starttime.strftime('%F-%H-%M'),
                                                stoptime.strftime('%F-%H-%M'))
        # Then check if the key's in the dict:
        if param_table_keyname not in param_table:
            # We need to add this information to it:
            eta_squared = get_eta_squared(mag_str[numpy.searchsorted(mag_epoch,
                                                                     starttime)\
                                                  :numpy.searchsorted(mag_epoch,
                                                                      stoptime)])
            # Create the dict:
            param_table[param_table_keyname] = {}
            # Add eta-squared to it.
            param_table[param_table_keyname]['eta-squared'] = eta_squared

        # Set figsize:
        plt.figure(figsize=(10, 8))

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
            with open('../data/ic_event_{}_flux.pickle{}'\
                      .format(varname, sys.version_info[0]),
                      'rb') as fp:
                arrs = pickle.load(fp)
            flux = arrs['flux']
            dflux = arrs['dflux']
            epoch = arrs['epoch']
            energy = arrs['energy']

            # Do the fit here:
            fits = fit(varname, model, epoch, energy, flux,
                       dflux, starttime, stoptime, i)

            for k, f in enumerate(fits):
                # We must have at least one fit, so okay:
                plot_this_spectrum = True

                humanname, e, f, df, popt, pcov = f

                # Make labelstring and actually display the fit:
                fmtstr = '{} fit: {}; params: [' \
                    + '{:4G}, ' * (len(popt) - 1) \
                    + '{:4G}' + ']'
                p = plt.plot(energy_range,
                             model(energy_range, *popt),
                             label=fmtstr.format(varname, humanname, *popt),
                             color=colors[k])

                # Also plot the points used for the fit:
                plt.plot(e, f, "o", color=p[-1].get_color())

                # Now add the params to the table:
                param_table[param_table_keyname][varname + ': ' + humanname] = popt[-1]


            if plot_this_spectrum is True:
                # Also plot the spectrum:
                startidx = numpy.searchsorted(epoch, starttime)
                stopidx = numpy.searchsorted(epoch, stoptime)
                # Do all the cutting and flipping and stuff:
                cenergy, cflux, cdflux = fit_prep(varname,
                                                  startidx,
                                                  stopidx,
                                                  energy,
                                                  flux,
                                                  dflux)
                # Don't cut down energies because we don't care about just displaying:
                plt.errorbar(cenergy,
                             cflux,
                             yerr=cdflux,
                             label='{} spectrum'.format(varname),
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
        # plt.show()
        plt.close()

if __name__ == "__main__":
    # Open the old param table if it exists.
    if os.path.exists('../data/param_table.json'):
        with open('../data/param_table.json', 'r') as fp:
            param_table = json.load(fp)

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

    # Lastly, if called as a script, we have to save the param table:
    with open('../data/param_table.json', 'w') as fp:
        json.dump(param_table, fp, sort_keys=True, indent=4)
