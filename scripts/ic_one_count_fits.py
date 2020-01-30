#!/usr/bin/env python3
"""
Script to produce plots and fits from 1-count rates for ChanP, ChanR, and ChanT
rates.
"""

import argparse
import datetime
import matplotlib.pyplot as plt
import numpy
import re

import isois
import isois.epilo_l2
import isois.time

from common import PLOTTING_XLIM_LOWER, PLOTTING_XLIM_UPPER, \
    PLOTTING_YLIM_LOWER, PLOTTING_YLIM_UPPER



# def tupleit(l):
    # """ Function recursively turns arbitrarily nested lists into nested
    # tuples. """
    # return tuple(map(tupleit, l)) if isinstance(l, list) else l

def get_change_indices(inarr):
    """ Get indices of changes in the cal file. """
    # Start with 0, we'll say it changes from nothing to something here...
    indices = [0]
    # import pdb; pdb.set_trace()
    for i in indices:
        where = inarr[i:] == inarr[i, :, :]
        # We have to get rid of NaNs as well.  If something's NaN, we assume
        # it's fine, (ie., set True (is equal)).  (Note that we can't just
        # index them out because numpy converts the array to 1-d when
        # indexing out NaNs, which explicitly doesn't work here):
        where[numpy.isnan(inarr[i:])] = True
        # Need 0 for zeroth axis:
        where = numpy.where(numpy.all(where, axis=(1, 2)) == False)[0]
        if len(where) > 0:
            # If we've found a change, append it to the list:
           indices.append(where[0])
    return indices

def main(calfile):
    # Open the calfile so we can get information:
    cal = isois.epilo_l2.CalFile(calfile)
    # Get a list of all available files so we can find the first/last dates:
    files = isois.get_latest('psp_isois-epilo_l2-ic')
    time_first = datetime.datetime.strptime(re.match("^.*([0-9]{8}).*$",
                                                     files[0]).group(1),
                                            "%Y%m%d")
    time_last = datetime.datetime.strptime(re.match("^.*([0-9]{8}).*$",
                                                    files[-1]).group(1),
                                           "%Y%m%d")
    dtimes = [time_first + i * datetime.timedelta(days=1)
              for i in range((time_last - time_first).days)]

    for s in ('P', 'R', 'T'):
        calinfo = cal.get(s, [isois.time.datetime_to_met(d) for d in dtimes])
        # Let's hope these all change at the same time(s).  We'll check this by
        # just comparing these lists to make sure they're the same:
        change_indices = get_change_indices(calinfo['eff'])
        assert change_indices \
            == get_change_indices(calinfo['elo']) \
            == get_change_indices(calinfo['ehi'])
        # We need to make sure this doesn't change or changes at the same
        # time the others change:
        change_indices_g = get_change_indices(calinfo['g'])
        assert (len(change_indices_g) == 1) \
            or (change_indices_g == change_indices)
        # Now, for each time we can, we can just make a plot, apparently:
        for j, idx in enumerate(change_indices):
            # These are directly copied from the fitting script:
            plt.figure(figsize=(10, 8))
            # These limits don't quite capture the entire plot(s), so just use
            # larger bounds instead.  Also, remove all these static plotting
            # limits of switching back to the one-over, etc...
            # plt.xlim((PLOTTING_XLIM_LOWER, PLOTTING_XLIM_UPPER))
            plt.xlim((1, 1e5))
            # plt.ylim((PLOTTING_YLIM_LOWER, PLOTTING_YLIM_UPPER))
            plt.ylim((1e-5, 1))
            plt.xscale('log')
            plt.xlabel('Energy (keV)')
            plt.yscale('log')
            plt.ylabel('j')
            plt.title('About {} to {}: spoofed 1-count rates'
                      ' for Chan{}'\
                      .format(dtimes[change_indices[j]].strftime('%F'),
                              dtimes[change_indices[j + 1]].strftime('%F')
                              if j < len(change_indices) - 1
                              else dtimes[-1].strftime('%F'),
                              s))

            # Need to mean over axis 0 (after indexing, this is the look dir.
            # axis), because that's what we're doing elsewhere:
            # counts = numpy.nanmean(1 / (calinfo['g'][idx] * calinfo['eff'][idx]\
                # * (calinfo['ehi'][idx] - calinfo['elo'][idx])), axis=0)
            counts = numpy.nanmean((calinfo['g'][idx] * calinfo['eff'][idx]\
                * (calinfo['ehi'][idx] - calinfo['elo'][idx])), axis=0)
            # NOTE: If changing back, MAKE SURE TO REMOVE STATIC LIMITS ON PLOT!

            # Check that all the values along the 0th axis are the same:
            assert numpy.all((calinfo['ehi'][idx] == calinfo['ehi'][idx, 0])[~numpy.isnan(calinfo['ehi'][idx])])\
                and numpy.all((calinfo['elo'][idx] == calinfo['elo'][idx, 0])[~numpy.isnan(calinfo['ehi'][idx])])
            # The energy values are (regrettably) just the mean of the high and
            # low end:
            x = calinfo['ehi'][idx, 0] + calinfo['elo'][idx, 0] / 2

            plt.plot(x, counts)

            plt.tight_layout()

            plt.savefig('../figures/spectrum_Chan{}_{}_spoofed_1-count_rate.png'.format(s, dtimes[idx].strftime('%F')),
                        dpi=300)
            # plt.show()
            plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(help='epilo calfile, a .sqlite',
                        dest='calfile',
                        action='store',
                        default='/home/share/docs/epi-lo/epilo-calibration-'
                        'table/20190219/EPILOCalibration.sqlite',
                        nargs='?')
    args = parser.parse_args()
    main(args.calfile)
