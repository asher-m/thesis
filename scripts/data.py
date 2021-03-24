#!/usr/bin/env python3
""" Data interface for the thesis.

Data interface for the thesis. All data i/o should happen here.
"""


import bz2
import datetime
import functools
import glob
import multiprocessing
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

EVENTS_FILE = '../data/eventtimes{}.pickle{}'
DATASETS = {
    'psp_isois-epilo_l2-ic': [
        {
            'epoch': 'Epoch_ChanT',
            'flux': 'H_Flux_ChanT',
            'flux_unc': 'H_Flux_ChanT_DELTA',
            'pa': 'PA_ChanT',
            'sa': 'SA_ChanT',
            'energy': 'H_ChanT_Energy',
            'energy_unc_plus': 'H_ChanT_Energy_DELTAPLUS',
            'energy_unc_minus': 'H_ChanT_Energy_DELTAMINUS',
            'reverse': True  # reverse array along energy axis
        },
        {
            'epoch': 'Epoch_ChanP',
            'flux': 'H_Flux_ChanP',
            'flux_unc': 'H_Flux_ChanP_DELTA',
            'pa': 'PA_ChanP',
            'sa': 'SA_ChanP',
            'energy': 'H_ChanP_Energy',
            'energy_unc_plus': 'H_ChanP_Energy_DELTAPLUS',
            'energy_unc_minus': 'H_ChanP_Energy_DELTAMINUS'
        }
    ]
}

cpus_on_system = multiprocessing.cpu_count()
cpus_to_use = cpus_on_system - 4 if cpus_on_system - 4 > 0 else cpus_on_system

# make vector for convenience
d2t = numpy.vectorize(spacepy.pycdf.lib.datetime_to_tt2000)
t2d = numpy.vectorize(spacepy.pycdf.lib.tt2000_to_datetime)


def _read_data_process(verbose, raw_epoch, d, f):
    cdf = spacepy.pycdf.CDF(f)  # open cdf
    file_data = dict()  # holding place for this file's data
    if verbose:
        print('\t\tReading file {}...'.format(f))
    for g in DATASETS[d]:  # for every group of variables
        for v in g:  # for every variable in the group
            if v == 'reverse':
                continue
            # some bools to make this easier:
            reverse = True if 'reverse' in g.keys() and g['reverse'] is True \
                else False
            energy_var = True if 'energy' in v or v == 'flux' or v == 'flux_unc' \
                else False
            # read data and append to right place in the right way
            if v == 'epoch' and raw_epoch is True:
                file_data[g[v]] = cdf.raw_var(g[v])[...]
            elif energy_var:
                varcopy = spacepy.pycdf.VarCopy(cdf[g[v]])
                # fill nan(s):
                spacepy.pycdf.istp.nanfill(varcopy)
                vardat = varcopy[...]
                # reverse if necessary:
                if reverse:
                    vardat = vardat[:, :, ::-1]
                file_data[g[v]] = vardat
            else:
                varcopy = spacepy.pycdf.VarCopy(cdf[g[v]])
                # fill nan(s):
                spacepy.pycdf.istp.nanfill(varcopy)
                vardat = varcopy[...]
                file_data[g[v]] = vardat

    return file_data


def read_data(globstr='', verbose=True, raw_epoch=True, use_cache=True):
    """ Function to read event data from CDFs (without concat). """
    if use_cache is True:
        files = sorted(glob.glob('../data/eventdata_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'
                                 + (('_' + globstr) if len(globstr) > 0 else (''))
                                 + '.pickle{}.bz2'.format(sys.version_info[0])))  # nopep8
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

    if platform.system() != 'Linux':
        raise OSError('Received request to remake eventdata cache or failed to find cache '
                      'and resorting to rebuilding but '
                      'cannot use isois library on non-Linux systems!')

    # create multiprocessing pool
    with multiprocessing.Pool(cpus_to_use) as pool:
        outdata = []  # create holding place
        events = read_events(globstr)  # read events

        for i, e in enumerate(events):
            strt = e[0]
            stop = e[1]
            if verbose:
                print('Working on event {} (index {:02d}):'.format(
                    strt.strftime('%Y-%j'),
                    i
                ))

            event_outdata = {d: {} for d in DATASETS}

            for d in DATASETS:
                if verbose:
                    print('\tWorking on dataset {}:'.format(d))

                # make destination for data
                event_data = {g[v]: [] for g in DATASETS[d]
                              for v in g if not v == 'reverse'}

                # bake function for mp.Pool.map
                _read_data_process_baked = functools.partial(
                    _read_data_process,
                    verbose,
                    raw_epoch,
                    d
                )

                # get list of files for this start/stoptime
                files = get_files(d, strt, stop)

                # do the mp.Pool.map
                file_data = pool.map(
                    _read_data_process_baked,
                    files
                )

                # recombine data in list for concat
                for p in file_data:
                    if p is not None:
                        for v in p:
                            event_data[v].append(p[v])

                try:
                    for v in event_data:
                        event_outdata[d][v] = numpy.concatenate(event_data[v])
                except ValueError as e:
                    print('Error on concatenation!  Got:\n\tValueError: {}'.format(e))

            outdata.append(event_outdata)

    # Save for faster access:
    print('Working on writing data to cache and bz2 compression...')
    with bz2.BZ2File('../data/eventdata_{}{}.pickle{}.bz2'.format(datetime.datetime.now().strftime('%Y%m%d'),
                                                                  (('_' + globstr)
                                                                   if len(globstr) > 0 else ('')),
                                                                  sys.version_info[0]),
                     'wb',
                     compresslevel=1) as fp:
        pickle.dump(outdata, fp)

    return outdata


def read_events(globstr=''):
    with open(EVENTS_FILE.format((('_' + globstr) if len(globstr) > 0 else ('')),
                                 sys.version_info[0]), 'rb') as fp:
        # event datetimes are only in last column
        return pickle.load(fp)[..., 0]


def get_files(d, strt, stop):
    daterange = [strt.date() + datetime.timedelta(days=d)
                 for d in range((stop.date() - strt.date()).days + 1)]
    # sanity check
    assert(strt.date() in daterange)
    assert(stop.date() in daterange)

    files = list()
    for date in daterange:
        ftoday = isois.get_latest(d, date=date)
        if len(ftoday) > 0:
            files.append(ftoday[-1])
    return files


def model(e, j_0, k):
    """ Fisk & Gloeckler 2008 Eq. 38 changed a bit. """
    return j_0 * e**k
