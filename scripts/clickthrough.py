#!/usr/bin/env python
import bz2
import datetime
import glob
import numpy
import os
import pickle
import platform
import re
import sys

import spacepy
import spacepy.datamanager
import spacepy.pycdf
import spacepy.pycdf.istp

if platform.system() == 'Linux':  # probably, hopefully running on isoc
    import isois
    import isois.isois_cdf
    import isois.level3_master


def edges(datestr, epoch=False):
    """ Shamelessly copied from level3.py at 1cb66199 (for which I'm the author).

    Make all edges for rebin, (sometimes from bin centers
    and pmvals, sometimes just from scratch according to
    cadence). 

    Follows ordering of CDF depends. 

    :param bool epoch: Yield epoch from calcualting time edges for CDF.    
    """
    # Time bins: keep in mind this is an hourly product.
    today = datetime.datetime.strptime(datestr, '%Y%m%d')
    strt_tt2000 = spacepy.pycdf.lib.datetime_to_tt2000(today)
    stop_tt2000 = spacepy.pycdf.lib.datetime_to_tt2000(
        today + datetime.timedelta(days=1))
    # Minutes, the "plus-minus value" (so half bin spacing)
    delta_minutes = 30
    delta = delta_minutes * 60e9  # convert to nanoseconds
    # Time values centered in bin:
    times = numpy.arange(strt_tt2000 + delta, stop_tt2000, delta * 2)
    # If flagged for epoch, return now:
    if epoch is True:
        return times
    # By explicitly making the upper edge of the last bin the stop_tt2000 value
    # the leap second is captured in the last bin.  This is in contrast
    # to making the edges by adding the delta to all the values which risks
    # not capturing leap seconds in the last bin.
    edges_times = numpy.array(
        list(times - delta) + list((stop_tt2000,)))

    # PA bins, for now:
    edges_lookdir = dict()
    for lkdr in ('omni', 'pa', 'sa'):
        pabins, papmvals = isois.level3_master.master_isois_3_summary_lookdir_bins(
            lkdr)
        edges_lookdir[lkdr] = numpy.array(
            list((pabins[0] - papmvals[0, 0],)) + list(pabins + papmvals[:, 1]))

    # Energy bins:
    ebins, epmvals = isois.level3_master.master_isois_3_summary_energy_bins()
    ebins *= 1e3
    epmvals *= 1e3
    edges_energy = numpy.array(
        list((ebins[0] - epmvals[0, 0],)) + list(ebins + epmvals[:, 1]))

    return edges_times, edges_lookdir, edges_energy


def read_data():
    # how many intervals we're splitting days into (24 is on the hour)
    intervals_per_day = 24
    # place to store outdata
    outdata = dict()
    # get energy bin centers, use arbitrary date (because not touching epoch)
    _, edges_lookdir, edges_energy = edges('20180101')
    outdata['energy'] = numpy.array(
        (edges_energy[:-1] + edges_energy[1:]) / 2)

    # get files
    files = isois.get_latest('psp_isois-epilo_l2-ic')
    # some test cases:
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:1]
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:10] # just first 10
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:100]

    for v in ('ChanT', 'ChanP', 'ChanR'):
        # just create holding places
        outdata[v] = dict()
        outdata[v]['flux'] = numpy.empty((24 * len(files), 32), dtype=numpy.float32)
        outdata[v]['flux'].fill(numpy.nan)
        outdata[v]['epoch'] = numpy.empty((24 * len(files)), dtype=numpy.int64)
        outdata[v]['epoch'].fill(-1)

    j = 0

    for f in files:
        print('Starting file {}...'.format(os.path.basename(f)))
        finfo = isois.isois_cdf.cdf_filename_parser(f)
        c = spacepy.pycdf.CDF(f)  # open cdf
        for v in ('ChanT', 'ChanP', 'ChanR'):
            epoch = c.raw_var('Epoch_{}'.format(v))[...]
            flux = c['H_Flux_{}'.format(v)][...]
            energy = c['H_{}_Energy'.format(v)][...]
            pa = c['PA_{}'.format(v)][...]
            # get edges
            edges_times, edges_lookdir, edges_energy = edges(finfo['date'])

            # do energy rebin
            flux = spacepy.datamanager.rebin(
                flux,
                energy,
                edges_energy,
                axis=2
            )
            # do pa rebin
            flux = spacepy.datamanager.rebin(
                flux,
                pa,
                edges_lookdir['omni'],
                axis=1
            )
            # do time rebin
            flux = spacepy.datamanager.rebin(
                flux,
                epoch,
                edges_times,
                axis=0
            )

            outdata[v]['flux'][j:j + 24] = numpy.squeeze(flux)
            outdata[v]['epoch'][j:j + 24] = numpy.array(edges(finfo['date'], epoch=True))

        j += 24

    with bz2.BZ2File('../data/clickdata.pickle{}.bz2'.format(sys.version_info[0]),
                     'wb', compresslevel=1) as fp:
        pickle.dump(outdata, fp)

if __name__ == '__main__':
    read_data()
