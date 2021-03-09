import bz2  # nopep8
import datetime  # nopep8
import functools  # nopep8
import glob  # nopep8
import multiprocessing  # nopep8
import numpy  # nopep8
import os  # nopep8
import pickle  # nopep8
import platform  # nopep8
import re  # nopep8
import sys  # nopep8

import spacepy  # nopep8
import spacepy.pycdf  # nopep8
import spacepy.pycdf.istp  # nopep8

if platform.system() == 'Linux':  # probably, hopefully running on isoc
    import isois  # nopep8

import data  # nopep8


DATASETS = {
    'psp_isois-epilo_l2-ic': [
        {
            'epoch': 'Epoch_ChanP',
            'flux': 'H_Flux_ChanP',
            'flux_unc': 'H_Flux_ChanP_DELTA',
            'pa': 'PA_ChanP',
            # 'sa': 'SA_ChanP',  # no spiral angle in isois v1 cdfs
            'energy': 'H_ChanP_Energy',
            'energy_unc_plus': 'H_ChanP_Energy_DELTAPLUS',
            'energy_unc_minus': 'H_ChanP_Energy_DELTAMINUS'
        }
    ]
}

dd = 'psp_isois-epilo_l2-ic'
dd_keys = DATASETS[dd][0]
datadir = '/home/share/data/data_public/archive/release_02/EPILo/level2/'

cpus_on_system = multiprocessing.cpu_count()
cpus_to_use = cpus_on_system - 4 if cpus_on_system - 4 > 0 else cpus_on_system


def _read_data_process(verbose, raw_epoch, d, strtday, i):
    """Process to read one data file.

    Copied from data module."""
    # verbose, raw_epoch, d, i, strtday = gargs  # unpack args; makes multiprocessing easier
    file_data = dict()  # holding place for this file's data
    # get today's file
    daystr = (strtday + i * datetime.timedelta(days=1)).strftime('%Y%m%d')
    # dataset is defined by d (above)
    f_today = glob.glob(os.path.join(datadir, d + '_' + daystr + '*'))
    # it's possible that we occasionally have no files for today
    if len(f_today) > 0:
        f = f_today[0]
    else:
        return
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


def read_data(verbose=True, raw_epoch=True, use_cache=True):
    """Read event data from CDFs (without concat).

    Copied from data module."""
    globstr = 'joyce-apj-tab2'

    if use_cache is True:
        files = sorted(glob.glob('../data/eventdata_joyce-apj-tab2_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'
                                 + (('_' + globstr) if len(globstr) > 0 else (''))
                                 + '.pickle{}.bz2'.format(sys.version_info[0])))  # nopep8
        if len(files) > 0:
            now = datetime.datetime.now()
            m = re.match(r'^eventdata_joyce-apj-tab2_(\d{8}).*', os.path.basename(files[-1]))  # nopep8
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
        events = data.read_events(globstr)  # read events

        for i, e in enumerate(events):
            strtday = data.floor_datetime(e[0])
            stopday = data.ceil_datetime(e[1])
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

                _read_data_process_baked = functools.partial(
                    _read_data_process,
                    verbose,
                    raw_epoch,
                    d,
                    strtday
                )

                file_data = pool.map(
                    _read_data_process_baked,
                    range((stopday - strtday).days)
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


def main():
    al = list()

    for i, e in enumerate(data.read_data('joyce-apj-tab2')):
        # unpack relevant vars
        epoch = e[dd][dd_keys['epoch']]
        flux = e[dd][dd_keys['flux']]
        flux_unc = e[dd][dd_keys['flux_unc']]
        energy = e[dd][dd_keys['energy']]
        energy_unc_plus = e[dd][dd_keys['energy_unc_plus']]
        energy_unc_minus = e[dd][dd_keys['energy_unc_minus']]

        # cut down to only sectors of interest
        flux = flux[:, numpy.arange(20, 50), :]

        # exclude 25, 31, 34, 35, 44
        # minus 20 to realign with array
        ex = numpy.array([25, 31, 34, 35, 44]) - 20
        flux[:, ex, :] = numpy.nan  # replace with nan

        flux = numpy.nanmean(flux, axis=(0, 1))

        # need to replace nan(s) with 0(s) at this point (for numpy.average)
        idx = numpy.isnan(flux)
        flux[idx] = 0

        # sanity check
        assert(numpy.all((energy_unc_plus == energy_unc_plus[0, :, :])[~numpy.isnan(energy_unc_plus)]))  # nopep8
        assert(numpy.all((energy_unc_minus == energy_unc_minus[0, :, :])[~numpy.isnan(energy_unc_minus)]))  # nopep8
        weights = numpy.nanmean(energy_unc_plus[0], axis=0) + numpy.nanmean(energy_unc_minus[0], axis=0)  # nopep8
        weights[idx] = 0  # again replace with 0

        avgflux = numpy.average(flux, weights=weights)

        al.append(avgflux*1e3)

    print(al)

if __name__ == '__main__':
    main()
