# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 00:11:48 2020

@author: asher
"""

import json
import numpy
import pickle
import sys
import types

# This import is a bit ugly, but it's less ugly than retyping this every time:
from common import get_eta_squared, fit, fit_prep, MODEL as model, \
    EVENTS_FILE, MAG_FILE



varnames = ('ChanP', 'ChanT', 'ChanR')

def main(events_file, mag_file):
    params = {}
    params_complete = {}

    # Open the saved event times:
    with open(events_file, 'rb') as fp:
        e = pickle.load(fp)
        # Note that e is like: [((t0_start, y0_start), (t0_stop, y0_stop)),
        #                       ((t1_start, y1_start), (t1_stop, y1_stop)), ...]

    with open(mag_file, 'rb') as fp:
        arrs = pickle.load(fp)
    mag = types.SimpleNamespace(**arrs)

    # Now, for each event (row) in e:
    for i, event in enumerate(e):
        # Cut all vars in time that we're looking at:
        starttime = event[0, 0]
        stoptime = event[1, 0]

        # Do the math for the B field if it's not there yet.  First make
        # key for the event:
        keyname = "{}_to_{}".format(starttime.strftime('%F-%H-%M'),
                                                stoptime.strftime('%F-%H-%M'))
        # Then check if the key's in the dict:
        if keyname not in params:
            # Create the dict:
            params[keyname] = {}
            params_complete[keyname] = {}

        # We need to add this information to it:
        eta_squared = get_eta_squared(mag.mag[numpy.searchsorted(mag.epoch, starttime) \
                                              :numpy.searchsorted(mag.epoch, stoptime)])
        # ETA-SQUARED
        params[keyname]['eta_squared'] = eta_squared
        params_complete[keyname]['eta_squared'] = eta_squared

        for j, varname in enumerate(varnames):
            # Open the arrays:
            with open('../data/flux_event_{}.pickle{}'\
                      .format(varname, sys.version_info[0]),
                      'rb') as fp:
                arrs = pickle.load(fp)
            flux = types.SimpleNamespace(**arrs)

            # EXP TAIL COEFF
            fits = fit(varname, model, flux.epoch, flux.energy, flux.flux,
                       flux.dflux, starttime, stoptime, i)
            for f in fits:
                humanname, e, f, df, popt, pcov = f
                params[keyname]['{}: {}'.format(varname, humanname)] = popt[-1]
                params_complete[keyname]['{}: {}'.format(varname, humanname)] = popt[-1]

            # ANISOTROPY
            startidx = numpy.searchsorted(flux.epoch, starttime)
            stopidx = numpy.searchsorted(flux.epoch, stoptime)
            cenergy, flux_par, _ = fit_prep(varname, startidx, stopidx,
                                            flux.energy, flux.flux_par,
                                            flux.dflux_par)
            _, flux_perp, _ = fit_prep(varname, startidx, stopidx,
                                       flux.energy, flux.flux_perp,
                                       flux.dflux_perp)
            _, flux_apar, _ = fit_prep(varname, startidx, stopidx,
                                       flux.energy, flux.flux_apar,
                                       flux.dflux_apar)
            params[keyname]['{} anisotropy: par over perp'.format(varname)] = numpy.nanmean(flux_par) / numpy.nanmean(flux_perp)
            params[keyname]['{} anisotropy: apar over perp'.format(varname)] = numpy.nanmean(flux_apar) / numpy.nanmean(flux_perp)
            params_complete[keyname]['flux_par'] = flux_par
            params_complete[keyname]['flux_perp'] = flux_perp
            params_complete[keyname]['flux_apar'] = flux_apar

    with open('../data/params.json', 'w') as fp:
        json.dump(params, fp, sort_keys=True, indent=4)
    with open('../data/params_complete.pickle{}'.format(sys.version_info[0]),
              'wb') as fp:
        pickle.dump(params_complete, fp)

if __name__ == "__main__":
    main(EVENTS_FILE, MAG_FILE)
