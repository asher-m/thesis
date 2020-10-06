#!/usr/bin/env python3
""" Data interface for the thesis.

Data interface for the thesis. All data i/o should happen here.
"""


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
import spacepy.pycdf
import spacepy.pycdf.istp

if platform.system() == 'Linux':  # probably, hopefully running on isoc
    import isois

EVENTS_FILE = '../data/eventtimes.pickle{}'.format(sys.version_info[0])
DATASETS = {
    'psp_isois-epilo_l2-ic': [
        {
            'epoch': 'Epoch_ChanT',
            'flux': 'H_Flux_ChanT',
            'flux_unc': 'H_Flux_ChanT_DELTA',
            'pa': 'PA_ChanT',
            'sa': 'SA_ChanT',
            'energy': 'H_ChanT_Energy',
            'reverse': True  # reverse array along energy axis
        },
        {
            'epoch': 'Epoch_ChanP',
            'flux': 'H_Flux_ChanP',
            'flux_unc': 'H_Flux_ChanP_DELTA',
            'pa': 'PA_ChanP',
            'sa': 'SA_ChanP',
            'energy': 'H_ChanP_Energy'
        }
    ]
}


def read_data(verbose=True, raw_epoch=True, use_recent=True):
    """ Function to read event data from CDFs (without concat). """
    if platform.system() != 'Linux' and use_recent is not True:
        raise OSError('Received request to remake eventdata cache but '
                      'cannot use isois library on non-Linux systems!')

    if use_recent is True:
        files = sorted(glob.glob('../data/eventdata_*.pickle{}.bz2'.format(sys.version_info[0])))  # nopep8
        if len(files) > 0:
            now = datetime.datetime.now()
            m = re.match(r'^eventdata_(\d{8}).*', os.path.basename(files[-1]))
            most_recent = datetime.datetime.strptime(m.group(1), '%Y%m%d')
            if now - most_recent < datetime.timedelta(weeks=1):
                if verbose is True:
                    print('Found cached data from file {}, '
                          'using contents...'.format(os.path.basename(files[-1])))
                with bz2.BZ2File(files[-1], 'rb') as fp:
                    outdata = pickle.load(fp)
                # Able to read, so return
                return outdata

    # If not, do the regular:
    outdata = []

    events = read_events()

    for i, e in enumerate(events):
        strtday = floor_datetime(e[0])
        stopday = ceil_datetime(e[1])
        if verbose:
            print('Working on event {} (index {:02d}):'.format(
                strtday.strftime('%Y-%j'),
                i
            ))

        event_outdata = {d: {} for d in DATASETS}

        for d in DATASETS:
            if verbose:
                print('\tWorking on dataset {}:'.format(d))

            event_data = {g[v]: [] for g in DATASETS[d]
                          for v in g if not v == 'reverse'}
            for i in range((stopday - strtday).days):  # open file for everyday
                f = isois.get_latest(
                    d, date=(strtday + i * datetime.timedelta(days=1)).strftime('%Y%m%d'))[0]
                if verbose:
                    print('\t\tReading file {}...'.format(f))

                cdf = spacepy.pycdf.CDF(f)
                for g in DATASETS[d]:  # for every group of variables
                    for v in g:  # for every variable in the group
                        if v == 'reverse':
                            continue
                        # some bools to make this easier:
                        reverse = True if 'reverse' in g.keys() and g['reverse'] is True \
                            else False
                        energy_var = True if v == 'energy' or v == 'flux' or v == 'flux_unc' \
                            else False
                        # read data and append to right place in the right way
                        if v == 'epoch' and raw_epoch is True:
                            event_data[g[v]].append(cdf.raw_var(g[v])[...])
                        elif energy_var:
                            varcopy = spacepy.pycdf.VarCopy(cdf[g[v]])
                            # fill nan(s):
                            spacepy.pycdf.istp.nanfill(varcopy)
                            vardat = varcopy[...]
                            # reverse if necessary:
                            if reverse:
                                vardat = vardat[:, :, ::-1]
                            # figure out what nan(s) we have:
                            nonnan = numpy.any(~numpy.isnan(vardat), axis=(0, 1))  # nopep8
                            event_data[g[v]].append(vardat[:, :, nonnan])
                        else:
                            varcopy = spacepy.pycdf.VarCopy(cdf[g[v]])
                            # fill nan(s):
                            spacepy.pycdf.istp.nanfill(varcopy)
                            vardat = varcopy[...]
                            event_data[g[v]].append(vardat)

            for v in event_data:
                event_outdata[d][v] = numpy.concatenate(event_data[v])

        outdata.append(event_outdata)

    # Save for faster access:
    print('Working on writing data to cache and bz2 compression...')
    with bz2.BZ2File('../data/eventdata_{}.pickle{}.bz2'.format(datetime.datetime.now().strftime('%Y%m%d'),
                                                                sys.version_info[0]),
                     'wb',
                     compresslevel=1) as fp:
        pickle.dump(outdata, fp)

    return outdata


def read_events():
    with open(EVENTS_FILE, 'rb') as fp:
        # event datetimes are only in last column
        return pickle.load(fp)[..., 0]


def floor_datetime(date, delta=datetime.timedelta(days=1)):
    return date - (datetime.datetime.min - date) % delta


def ceil_datetime(date, delta=datetime.timedelta(days=1)):
    return date + (datetime.datetime.min - date) % delta


def model(e, j_0, k):
    """ Fisk & Gloeckler 2008 Eq. 38 changed a bit. """
    return j_0 * e**k
