#!/usr/bin/env python3
""" Data interface for the thesis.

Data interface for the thesis. All data i/o should happen here.
"""


import datetime
import numpy
import pickle
import sys

import isois
import spacepy
import spacepy.pycdf


EVENTS_FILE = \
    '../data/TEST_flux_event_times_cons.pickle{}'.format(sys.version_info[0])
MAG_FILE = '../data/B.pickle{}'.format(sys.version_info[0])
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


def read_data(verbose=False, raw_epoch=True):
    """ Function to read event data from CDFs (without concat). """
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

            event_data = {v: [] for g in DATASETS[d]
                          for v in g.values()}
            for i in range((stopday - strtday).days):  # open file for everyday
                f = isois.get_latest(
                    d, date=(strtday + i * datetime.timedelta(days=1)).strftime('%Y%m%d'))[0]
                if verbose:
                    print('\t\tReading file {}...'.format(f))

                cdf = spacepy.pycdf.CDF(f)
                for g in DATASETS[d]:  # for every group of variables
                    for v in g:  # for every variable in the group
                        # some bools to make this easier:
                        reverse = True if 'reverse' in g.keys() and g['reverse'] is True \
                            else False
                        energy_var = True if v == 'energy' or v == 'flux' or v == 'flux_unc' \
                            else False
                        # read data and append to right place in the right way
                        if v == 'epoch' and raw_epoch is True:
                            event_data[g[v]].append(cdf.raw_var(g[v])[...])
                        elif reverse and energy_var:
                            vardat = cdf[g[v]][:, :, ::-1]
                            vardat[vardat < 0] = numpy.nan
                            # figure out what nan(s) we have:
                            nonnan = numpy.any(~numpy.isnan(vardat), axis=(0, 1))
                            event_data[g[v]].append(vardat[:, :, nonnan])
                        elif energy_var:
                            # need to trunc off fill on either/both ends:
                            vardat = cdf[g[v]][...]
                            vardat[vardat < 0] = numpy.nan
                            # figure out what nan(s) we have:
                            nonnan = numpy.any(~numpy.isnan(vardat), axis=(0, 1))
                            event_data[g[v]].append(vardat[:, :, nonnan])
                        else:
                            event_data[g[v]].append(cdf[g[v]][...])

            for v in event_data:
                event_outdata[d][v] = numpy.concatenate(event_data[v])

        outdata.append(event_outdata)

    return outdata


def read_events():
    with open(EVENTS_FILE, 'rb') as fp:
        # event datetimes are only in last column
        return pickle.load(fp)[..., 0]


def floor_datetime(date, delta=datetime.timedelta(days=1)):
    return date - (datetime.datetime.min - date) % delta


def ceil_datetime(date, delta=datetime.timedelta(days=1)):
    return date + (datetime.datetime.min - date) % delta
