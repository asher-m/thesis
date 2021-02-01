#!/usr/bin/env python
import bz2
import datetime
import glob
import multiprocessing as mp
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

channels = (
    'ChanT',
    'ChanP',
    # 'ChanR'  # makes things take forever because of memory limits
)


def process_file(f):
    print('Starting file {}...'.format(os.path.basename(f)))
    finfo = isois.isois_cdf.cdf_filename_parser(f)
    c = spacepy.pycdf.CDF(f)  # open cdf
    r = dict()
    r['epoch'] = edges(finfo['date'], epoch=True)

    for v in channels:
        r[v] = dict()

        epoch = c.raw_var('Epoch_{}'.format(v))[...]
        # ...don't know how to check to make sure there isn't fill in here...
        # get relevant vars
        f_copy = c['H_Flux_{}'.format(v)].copy()
        e_copy = c['H_{}_Energy'.format(v)].copy()
        p_copy = c['PA_{}'.format(v)].copy()
        # fill
        spacepy.pycdf.istp.nanfill(f_copy)
        spacepy.pycdf.istp.nanfill(e_copy)
        spacepy.pycdf.istp.nanfill(p_copy)
        # read plain data
        f = f_copy[...]
        e = e_copy[...]
        p = p_copy[...]

        # get edges
        edges_times, edges_lookdir, edges_energy = edges(finfo['date'])

        # do energy rebin
        f = spacepy.datamanager.rebin(
            f,
            e,
            edges_energy,
            axis=2
        )
        # do pa rebin
        f = spacepy.datamanager.rebin(
            f,
            p,
            edges_lookdir['omni'],
            axis=1
        )
        # do time rebin
        f = spacepy.datamanager.rebin(
            f,
            epoch,
            edges_times,
            axis=0
        )

        r[v]['flux'] = f

    c.close()
    return r


def edges(datestr, epoch=False):
    """ Shamelessly copied from level3.py at 1cb66199 (for which I'm the author).

    Make all edges for rebin, (sometimes from bin centers
    and pmvals, sometimes just from scratch according to
    cadence). 

    Follows ordering of CDF depends. 

    :param str datestr: datestr of day
    :param bool epoch: yield epoch from calcualting time edges for CDF
    """
    # Time bins: keep in mind this is an hourly product.
    today = datetime.datetime.strptime(datestr, '%Y%m%d')
    strt_tt2000 = spacepy.pycdf.lib.datetime_to_tt2000(today)
    stop_tt2000 = spacepy.pycdf.lib.datetime_to_tt2000(
        today + datetime.timedelta(days=1))  # weirdness for leapseconds
    # Minutes, the "plus-minus value" (so half bin spacing)
    delta_minutes = 30
    delta = delta_minutes * 60e9  # convert to nanoseconds
    # Time values centered in bin:
    times = numpy.arange(strt_tt2000 + delta, stop_tt2000, delta * 2)
    # If flagged for epoch, return now:
    if epoch is True:
        return times
    edges_times = numpy.array(list(times - delta) + list((stop_tt2000,)))

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
    # multiprocess stuff
    cpus = mp.cpu_count()
    cpus_to_use = cpus - 4 if cpus - 4 > 0 else cpus
    # cpus_to_use = 4

    # place to store outdata
    outdata = dict()

    # get energy bin centers, use arbitrary date (because not touching epoch)
    _, edges_lookdir, edges_energy = edges('20180101')
    outdata['energy'] = numpy.array(
        (edges_energy[:-1] + edges_energy[1:]) / 2)

    # get files
    files = isois.get_latest('psp_isois-epilo_l2-ic')
    # some test cases:
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:10] # just first 10
    # files = isois.get_latest('psp_isois-epilo_l2-ic', date='20190404')  # really big day


    outdata['epoch'] = numpy.empty((0), dtype=numpy.int64)
    outdata['epoch'].fill(-1)
    for v in channels:
        # just create holding places
        outdata[v] = dict()
        outdata[v]['flux'] = numpy.empty((0, 32), dtype=numpy.float32)
        outdata[v]['flux'].fill(numpy.nan)

    with mp.Pool(cpus_to_use) as pool:
        data = pool.map(process_file, files)

    print('Concatenating data...')
    for d in data:
        outdata['epoch'] = numpy.concatenate((outdata['epoch'], d['epoch']))
        for v in channels:
            outdata[v]['flux'] = numpy.concatenate(
                (outdata[v]['flux'], numpy.squeeze(d[v]['flux'])))

    srt_idx = numpy.argsort(outdata['epoch'])
    outdata['epoch'] = outdata['epoch'][srt_idx]
    for v in channels:
        outdata[v]['flux'] = outdata[v]['flux'][srt_idx]

    print('Zipping...')
    with bz2.BZ2File('../data/clickdata.pickle{}.bz2'.format(sys.version_info[0]),
                     'wb', compresslevel=1) as fp:
        pickle.dump(outdata, fp)

    return outdata


if __name__ == '__main__':
    read_data()
