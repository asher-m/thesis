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
    PLOTTING_YLIM_UPPER
from ic_models import fisk_2008_eq38_modified as model

# This is pretty case-specific right now, but I can blow it up to be more
# general later.

# For right now I'm going to use the hourly-averaged rates, because the time
# base doesn't make a difference because I'll be averaging down along the time
# axis again anyways.



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
        [("<human_name_0>", cenergy_0, cflux_0, cdflux_0, (popt_0...), (pcov_0...), start_time_ratio_0, stop_time_ratio_0),
         ("<human_name_1>", cenergy_1, cflux_1, cdflux_1, (popt_1...), (pcov_1...), start_time_ratio_1, stop_time_ratio_1)
         ...]
    """
    how = {'ChanT': [
                        {  # Blank dict so we still make the normal plot.
                        },
                        {  # Fit of full time range, trunc'ed e range:
                            'e_range':(95, 205),
                        },
                        {  # Fit of partial time range, trunc'ed e range:
                            'e_range':(95, 205),
                            't_range':(0.25, 0.75)
                        },
                        {  # Fit of partial time range, full e range:
                            'e_range':(37, 205),
                            't_range':(0.25, 0.75)
                        },
                    ],
           'ChanP': [
                        {  # Blank dict so we still make the normal plot.
                        },
                        {  # Fit of partial time range, full e range (for P):
                            'e_range':(95, 205),
                            't_range':(0.25, 0.75)
                        },
                    ]
        }
    """ Dictionary of how to plot each varname.  All plots for a particular
    varname are produced.

    These parameters override parameters in common. """

    human_name_base = 't_range: {}; e_range: {}'
    fits = []

    if varname in how:
        for params in how[varname]:
            if 't_range' in params:
                # Adjust the st*times according to how the dict says:
                timedelta = stoptime - starttime
                starttime_adj = starttime + timedelta * params['t_range'][0]
                stoptime_adj = starttime + timedelta * params['t_range'][1]
                # Then do the lookup as normal:
                startidx = numpy.searchsorted(epoch, starttime_adj)
                stopidx = numpy.searchsorted(epoch, stoptime_adj)
            else:
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
                fit_lower_bound, fut_upper_bound = params['e_range']
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
            human_name = human_name_base.format(t_range, e_range)
            # We have everything else, so we can just add it to the list:
            fits.append((human_name,
                         cenergy[e_startidx:e_stopidx],
                         cflux[e_startidx:e_stopidx],
                         cdflux[e_startidx:e_stopidx],
                         popt,
                         pcov,
                         params['t_range'][0] if 't_range' in params else 0.,
                         params['t_range'][1] if 't_range' in params else 1.,
                         ))

    else:
        # Because we don't have to treat this var in any special manner,
        # we can handle it like we did before:
        startidx = numpy.searchsorted(epoch, starttime)
        stopidx = numpy.searchsorted(epoch, stoptime)

        # Do all the cutting and flipping and stuff:
        cenergy, cflux, cdflux = fit_prep(varname,
                                          startidx,
                                          stopidx,
                                          energy,
                                          flux,
                                          dflux)

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
            print('{:^80}'.format('Something failed on optimization {}.'.format(idx)))
            print('='*80)

        # Finally, we can append everything to the list that we found.
        t_range = (0., 1.)
        e_range = energy_trunc(varname)
        human_name = human_name_base.format(t_range, e_range)
        # We have everything else, so we can just add it to the list:
        fits.append((human_name,
                     cenergy[e_startidx:e_stopidx],
                     cflux[e_startidx:e_stopidx],
                     cdflux[e_startidx:e_stopidx],
                     popt,
                     pcov,
                     0.,
                     1.))

    return fits


def main(events_file, mag_file, varname, together=False):
    # First get the param dictionary:
    global param_table

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
        if together is True:
            # Set this up now so we capture the plots (that are supposed to be
            # together) together on the same canvas:
            plt.figure(figsize=(10, 8))

        # Cut all vars in time that we're looking at:
        starttime = event[0, 0]
        stoptime = event[1, 0]

        # Do the fit here:
        fits = fit(varname, model, epoch, energy, flux,
                   dflux, starttime, stoptime, i)

        energy_range = numpy.logspace(numpy.log10(PLOTTING_XLIM_LOWER),
                                      numpy.log10(PLOTTING_XLIM_UPPER),
                                      1000)

        for f in fits:
            if together is False:
                # If we're plotting a new thing each time, we need to get it
                # on a canvas of the right size.  Essentially this
                # forces the creation of a new canvas of the right size and
                # still captures the plots of everything.
                plt.figure(figsize=(10, 8))
            humanname, e, f, df, popt, pcov, tr_start, tr_stop = f
            fmtstr = 'fit: {}; params: [' + '{:4G}, ' * (len(popt) - 1) + '{:4G}' + ']'
            p = plt.plot(energy_range,
                         model(energy_range, *popt),
                         label=fmtstr.format(humanname, *popt))
            # Also plot the points used for the fit:
            plt.plot(e, f, "o", color=p[-1].get_color())

            # Do the math for the B field if it's not there yet:
            param_table_keyname = "{}_to_{}".format(starttime.strftime('%F-%H-%M'),
                                                    stoptime.strftime('%F-%H-%M'))
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

            # Now add the params to the table:
            param_table[param_table_keyname][varname + ': ' + humanname] = popt[-1]

            if together is False:
                # Also plot the inner 50% time spectrum:
                startidx = numpy.searchsorted(epoch, starttime + tr_start * (stoptime - starttime))
                stopidx = numpy.searchsorted(epoch, starttime + tr_stop * (stoptime - starttime))
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
                             color='red',
                             label='spectrum')

                plt.xlim((PLOTTING_XLIM_LOWER, PLOTTING_XLIM_UPPER))
                plt.ylim((PLOTTING_YLIM_LOWER, PLOTTING_YLIM_UPPER))
                plt.xscale('log')
                plt.xlabel('Energy (keV)')
                plt.yscale('log')
                plt.ylabel('j')
                plt.legend(loc=1, prop={'family':'monospace'})
                plt.title('{} Event: {} to {}, (idx {:02d})'\
                          .format(varname,
                                  event[0, 0].strftime('%F %H%M'),
                                  event[1, 0].strftime('%F %H%M'),
                                  i))
                plt.tight_layout()
                plt.savefig('../figures/spectrum_{}_{:02d}_{}.png'.format(varname, i, humanname\
                                                                          .replace('_', '-')\
                                                                          .replace(' ', '')\
                                                                          .replace(';', '_')\
                                                                          .replace(':', '')\
                                                                          .replace(',', '-')),
                            dpi=300)
                plt.close()


        if together is True:
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
                         label='spectrum (full time range)')
            # Also plot the inner 50% time spectrum:
            startidx = numpy.searchsorted(epoch, starttime + 0.25 * (stoptime - starttime))
            stopidx = numpy.searchsorted(epoch, starttime + 0.75 * (stoptime - starttime))
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
                         label='spectrum (middle)')

            plt.xlim((PLOTTING_XLIM_LOWER, PLOTTING_XLIM_UPPER))
            plt.ylim((PLOTTING_YLIM_LOWER, PLOTTING_YLIM_UPPER))
            plt.xscale('log')
            plt.xlabel('Energy (keV)')
            plt.yscale('log')
            plt.ylabel('j')
            plt.legend(loc=1, prop={'family':'monospace'})
            plt.title('{} Event: {} to {}, (idx {:02d})'\
                      .format(varname,
                              event[0, 0].strftime('%F %H%M'),
                              event[1, 0].strftime('%F %H%M'),
                              i))
            plt.tight_layout()
            plt.savefig('../figures/spectrum_{}_{:02d}.png'.format(varname, i),
                        dpi=300)
            plt.close()

if __name__ == "__main__":
    # Open the old param table if it exists.
    if os.path.exists('../data/param_table.json'):
        with open('../data/param_table.json', 'r') as fp:
            param_table = json.read(param_table, fp)

    parser = argparse.ArgumentParser()
    parser.add_argument(help='events definition file (from clickthrough)',
                        dest='events_file',
                        action='store')
    parser.add_argument(help='mag file from field concat script',
                        dest='mag_file',
                        action='store')
    parser.add_argument(help='name of the variable to fit, can be'
                        ' \'ChanP\', \'ChanT\', \'ChanR\', or \'all\'',
                        dest='varname',
                        action='store')
    parser.add_argument('-t',
                        help='plot all spectra of a particular rate (ChanP,'
                        ' ChanT, or ChanR, regardless of time/energy slicing)'
                        ' on the same plot',
                        dest='together',
                        action='store_true',
                        default=False)
    args = parser.parse_args()
    # Make sure we have something we can look at. If this assertion fails,
    # it's because we received an invalid varname.
    assert args.varname in ('ChanP', 'ChanT', 'ChanR', 'all')
    if args.varname != 'all':
        main(args.events_file, args.mag_file, args.varname, args.together)
    else:
        for varname in ('ChanP', 'ChanT', 'ChanR'):
            print('='*80)
            print('{:^80}'.format('Starting {}...'.format(varname)))
            print('='*80)
            main(args.events_file, args.mag_file, varname, args.together)

    # Lastly, if called as a script, we have to save the param table:
    with open('../data/param_table.json', 'w') as fp:
        json.dump(param_table, fp, sort_keys=True, indent=4)
