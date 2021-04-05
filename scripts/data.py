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


cpus_on_system = multiprocessing.cpu_count()
cpus_to_use = cpus_on_system - 4 if cpus_on_system - 4 > 0 else cpus_on_system

# make vector for convenience
d2t = numpy.vectorize(spacepy.pycdf.lib.datetime_to_tt2000)
t2d = numpy.vectorize(spacepy.pycdf.lib.tt2000_to_datetime)


def model(e, j_0, k):
    """ Fisk & Gloeckler 2008 Eq. 38 changed a bit. """
    return j_0 * e**k


class Data:
    datasets = {
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
    """Dataset dict to parse/use."""
    events_file = '../data/eventtimes{}.pickle{}'
    """Eventsfile file format.  Probably shouldn't be changed."""

    def __init__(self, globstr=None, eventtimes_file=None, eventdata_file=None):
        """Read event-related data.

        :param globstr str: globstring of eventtimes file
        :param eventtimes_file str: path of eventtimes file
        :param eventdata_file str: path of eventdata file
        """
        self._globstr = None
        self._eventtimes_file = None
        self._eventdata_file = None
        self.eventtimes = None
        self.eventdata = None

        if (eventtimes_file is not None) ^ (eventdata_file is not None):
            raise ValueError(
                'Either both eventtimes_file and eventdata_file must be defined or neither of them can be defined!'
            )
        if globstr and eventtimes_file:
            raise ValueError(
                'Cannot define both globstr and eventtimes_file/eventdata_file!'
            )

        if globstr:
            self._globstr = globstr
        if eventtimes_file:  # only need to check one because of the assertion above
            # check if eventtimes_file actually exists
            assert(os.path.isfile(eventtimes_file))
            self._eventtimes_file = eventtimes_file
            assert(os.path.isfile(eventdata_file))
            self._eventdata_file = eventdata_file

    def read_events(self):
        """Read eventtimes into self.eventtimes."""
        if self._eventtimes_file is not None:
            with open(self._eventtimes_file, 'rb') as fp:
                # event datetimes are only in last column
                self.eventtimes = pickle.load(fp)[..., 0]

        else:  # use globstr instead
            with open(self.events_file.format((('_' + self._globstr) if self._globstr is not None else ('')),
                                              sys.version_info[0]), 'rb') as fp:
                # event datetimes are only in last column
                self.eventtimes = pickle.load(fp)[..., 0]

    def read_data(self, verbose=True, raw_epoch=True, use_cache=True, force_cache=False):
        """Read eventdata into self.eventdata.

        Read eventdata.  If class initialized with eventdata_file, the use_cache
        and force_cache parameters are ignored and cache is loaded directly from the
        targeted file.

        :param verbose bool: print status messages as reading progresses
        :param raw_epoch bool: use TT2000 for epoch in eventdata when not reading from cache (if using cache uses what's already there)
        :param use_cache bool: use cache if possible (cache less than one week old)
        :param force_cache bool: use cache if it exists (ignores age)
        """
        # return if eventdata is already found:
        if self.eventdata is not None and force_cache is False:
            return

        # make sure args are sensible:
        # always want use_cache if force_cache is True
        use_cache = use_cache or force_cache

        # try to use cache
        if self._eventdata_file is None and use_cache is True:
            # glob for files and sort
            files = glob.glob(
                '../data/eventdata_'
                + ((self._globstr + '_') if self._globstr is not None else (''))
                + '[0-9]' * 8
                + '.pickle{}.bz2'.format(sys.version_info[0])
            )
            files.sort()

            # if any files found
            if len(files) > 0:
                now = datetime.datetime.now()
                # get the date of the latest file
                m = re.match(r'^.*(\d{8}).*',
                             os.path.basename(files[-1]))
                most_recent = datetime.datetime.strptime(m.group(1), '%Y%m%d')

                # if it was made in the last week
                if now - most_recent < datetime.timedelta(weeks=1) or force_cache:
                    if verbose is True:
                        print(
                            'Found cached data from file {}, using contents'
                            '...'.format(os.path.basename(files[-1]))
                        )

                    # read and return
                    with bz2.BZ2File(files[-1], 'rb') as fp:
                        outdata = pickle.load(fp)
                    self.eventdata = outdata
                    return

        # otherwise, if given an eventdata file
        elif self._eventdata_file is not None:
            # read and return
            with bz2.BZ2File(self._eventdata_file, 'rb') as fp:
                outdata = pickle.load(fp)
            self.eventdata = outdata
            return

        if platform.system() != 'Linux':
            raise OSError(
                'Can\'t remake cache on non-Linux (non-isoc) system!'
            )

        # didn't use cache, read instead
        with multiprocessing.Pool(cpus_to_use) as pool:
            outdata = []  # create holding place
            if self.eventtimes is None:
                self.read_events()  # read events

            for i, e in enumerate(self.eventtimes):
                strt = e[0]
                stop = e[1]
                if verbose:
                    print('Working on event {} (index {:02d}):'.format(
                        strt.strftime('%Y-%j'),
                        i
                    ))

                event_outdata = {d: {} for d in self.datasets}
                for d in self.datasets:
                    if verbose:
                        print('\tWorking on dataset {}:'.format(d))

                    # make destination for data
                    event_data = {g[v]: [] for g in self.datasets[d]
                                  for v in g if not v == 'reverse'}

                    # bake function for mp.Pool.map
                    _read_data_process_baked = functools.partial(
                        Data._read_data_process,
                        self.datasets,
                        verbose,
                        raw_epoch,
                        d
                    )

                    # get list of files for this start/stoptime
                    files = self._get_files(d, strt, stop)

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

                    # concat all variables across all files into single array(s)
                    try:
                        for v in event_data:
                            event_outdata[d][v] = numpy.concatenate(
                                event_data[v])
                    except ValueError as e:
                        print(
                            'Error on concatenation!  Got:\n\tValueError: {}'.format(
                                e)
                        )

                    # #TODO: need to handle different groups of variables within the same dataset
                    # # right now everything is thrown together into one big soup
                    # # need to somehow return in different groups

                    # # cut down in time
                    # # look up indices in epoch of where to cut
                    # e_idx_strt = numpy.searchsorted(event_outdata[d][self.datasets[d]['epoch']], strt)
                    # e_idx_stop = numpy.searchsorted(event_outdata[d][self.datasets[d]['epoch']], stop)
                    # # do the cut
                    # for v in event_outdata[d]:
                    #     event_outdata[d][v] = event_outdata[d][v][e_idx_strt:e_idx_stop]

                outdata.append(event_outdata)

        # Save for faster access:
        print('Working on writing data to cache and bz2 compression...')
        with bz2.BZ2File(
            '../data/eventdata_{}{}.pickle{}.bz2'.format(  # no good way to break this line...
                ((self._globstr + '_') if self._globstr is not None else ('')),
                datetime.datetime.now().strftime('%Y%m%d'),
                sys.version_info[0]
            ),
            'wb',
            compresslevel=1
        ) as fp:
            pickle.dump(outdata, fp)

        self.eventdata = outdata

    @staticmethod
    def _read_data_process(datasets, verbose, raw_epoch, d, f):
        """Read one CDF and return its contents.

        :param datasets dict: datasets dictionary (see Data.datasets)
        :param verbose bool: print debug information
        :param raw_epoch bool: use raw_epoch (TT2000 if True) or datetimes (if False)
        :param d str: the dataset currently being used
        :param f str: the file being read
        """
        cdf = spacepy.pycdf.CDF(f)  # open cdf
        file_data = dict()  # holding place for this file's data
        if verbose:
            print('\t\tReading file {}...'.format(f))
        for g in datasets[d]:  # for every group of variables
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

    def _get_files(self, d, strt, stop):
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
