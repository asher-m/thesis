#!/usr/bin/env python
"""
Perform a fit on the data according to some model, as provided.
"""

import datetime
import json
import matplotlib.pyplot as plt
import numpy
import pickle
import re
import sys


from common import cut_like, get_nearest, REDS_RAW



NANFILL = 1
CHAN = 'P'
RE_CHAN = re.compile('.*Chan{}.*'.format(CHAN))
STARTDATE = datetime.datetime(2018, 9, 18, 1, 0, 0)
""" This is the first time we have good R values, (everything before is
sporadic and makes this script not work/display nicely). """
YLIM = (2e-3, 2e0)

# Open files:
with open('../data/magfield_eta_squared_daily_avg.pickle{}'\
          .format(sys.version_info[0]), 'rb') as fp:
    arr = pickle.load(fp)
    etasq_epoch = arr['epoch']
    etasq = arr['eta-squared']

with open('../data/R.pickle{}'.format(sys.version_info[0]), 'rb') as fp:
    arr = pickle.load(fp)
    R_epoch = arr['epoch']
    R = arr['R']

with open('../data/flux_event_Chan{}.pickle{}'\
          .format(CHAN, sys.version_info[0]), 'rb') as fp:
    arr = pickle.load(fp)
    flux_epoch = arr['epoch']
    flux = arr['flux']
    flux_energy = arr['energy']

with open('../data/flux_event_times_cons.pickle{}'\
          .format(sys.version_info[0]), 'rb') as fp:
    arr = pickle.load(fp)
    events = arr[:, :, 0]

with open('../data/param_table.json', 'r') as fp:
    param_table = json.load(fp)

def main():
    plot_etasq_scatter()
    plot_event_vs_etasq_scatter()

def plot_etasq_scatter():
    """ Function to plot eta-squared vs radial distance. """
    # Cut these arrays to where we start getting regular R values:
    c_etasq_epoch, c_etasq = cut_like(etasq_epoch, STARTDATE, *[etasq])
    c_R_epoch, c_R = cut_like(R_epoch, STARTDATE, *[R])
    # Now match up the values in the eta-sq epoch to R epoch:
    idxs = get_nearest(c_etasq_epoch, c_R_epoch)
    # Now we should be able to plot:
    plt.figure(figsize=(10, 8))
    plt.scatter(c_R[idxs], c_etasq, color=REDS_RAW[0], edgecolors='face')
    # Also plot nan's here:
    plt.scatter(c_R[idxs][numpy.isnan(c_etasq)],
                numpy.array([NANFILL for i in
                             range(numpy.sum(numpy.isnan(c_etasq)))]),
                color='black',
                edgecolors='face')
    plt.xlabel('R (AU)')
    plt.ylabel('$\eta^2$')
    plt.yscale('log')
    plt.ylim(YLIM)
    plt.tight_layout()
    # plt.show()
    plt.savefig('../figures/etasq_vs_R.png', dpi=300)
    plt.close()

def plot_event_vs_etasq_scatter():
    c_flux_epoch, c_flux, c_flux_energy = cut_like(flux_epoch, STARTDATE,
                                                   *[flux, flux_energy])
    c_etasq_epoch, c_etasq = cut_like(etasq_epoch, STARTDATE, *[etasq])
    c_R_epoch, c_R = cut_like(R_epoch, STARTDATE, *[R])
    # Also make these so we don't have to call scatterplot() so many times:
    list_R = []
    list_flux_max = []
    list_powerlaw_coeff = []
    list_etasq = []
    # Now that we have everything, we actually need to find the right
    # information for each event.  Let's start doing that, I guess...
    for e in events:
        start, stop = e
        param_table_keyname = "{}_to_{}".format(start.strftime('%F-%H-%M'),
                                                stop.strftime('%F-%H-%M'))
        # Cut all these arrays down into the data they have for this time
        # period:
        e_c_flux_epoch, e_c_flux, e_c_flux_energy = cut_like(c_flux_epoch,
                                                             e,
                                                             *[c_flux,
                                                               c_flux_energy],
                                                             cutside='both')
        e_c_etasq_epoch, e_c_etasq = cut_like(c_etasq_epoch, e, *[c_etasq],
                                              cutside='both')
        e_c_R_epoch, e_c_R = cut_like(c_R_epoch, e, *[c_R], cutside='both')
        # We can start doing the final(ish) calculations:
        e_c_flux_max = numpy.nanmax(numpy.nanmean(e_c_flux))
        e_c_etasq = numpy.nanmean(e_c_etasq)
        e_c_R = numpy.nanmean(e_c_R)
        # Get the coefficient of the powerlaw (the only thing in the param table).
        # This works by regexing the keys of the dict that the coeffs are stored
        # in for the event.
        chan_key = next(filter(RE_CHAN.match,
                               list(param_table[param_table_keyname].keys())))
        e_powerlaw_coeff = param_table[param_table_keyname][chan_key]
        # Finally we can append these to the lists:
        list_flux_max.append(e_c_flux_max)
        list_etasq.append(e_c_etasq)
        list_R.append(e_c_R)
        list_powerlaw_coeff.append(e_powerlaw_coeff)
    # Cast these to arrays to make some final changes:
    list_R = numpy.array(list_R)
    list_flux_max = numpy.array(list_flux_max)
    list_powerlaw_coeff = numpy.array(list_powerlaw_coeff)
    list_etasq = numpy.array(list_etasq)
    # Cut out any nans from all of these EXCEPT R:
    list_flux_max[numpy.isnan(list_flux_max)] = 1e5
    list_powerlaw_coeff[numpy.isnan(list_powerlaw_coeff)] = 1
    list_etasq[numpy.isnan(list_etasq)] = NANFILL
    # Lastly, we want to plot the small stuff last, so we'll concat these
    # arrays and have numpy sort along some axis (in reverse):
    p = numpy.array((list_R, list_etasq,
                     list_powerlaw_coeff, list_flux_max)).T
    p = p[numpy.array(p[:, 3].argsort())[::-1]]
    # I thought I was good at numpy, but that's from stackexchage:
    # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
    # And now we should be able to plot:
    plt.figure(figsize=(10, 8))
    plt.scatter(p[:, 0],
                p[:, 1],
                c=p[:, 2],
                s=10*(1e4*p[:, 3])**(1/2))
    plt.xlabel('R (AU)')
    plt.ylabel('$\eta^2$')
    plt.yscale('log')
    plt.ylim(YLIM)
    clb = plt.colorbar()
    clb.set_label('Power Law Exponent')
    plt.tight_layout()
    # plt.show()
    plt.savefig('../figures/etasq_vs_R_events.png', dpi=300)
    plt.close()
    plt.show()

if __name__ == '__main__':
    main()
