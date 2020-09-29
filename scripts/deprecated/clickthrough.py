#!/usr/bin/env python
"""
Script to import EPILo data files averaged into some arrays of flux and datetime
or epoch and display using spacepy.plot.utils.EventClicker so events can be
identified.
"""

import argparse
import datetime
import glob
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy
import os
import os.path
import pickle
import re
import sys

import spacepy.plot

from common import EVENTS_FILE



def main(events_file, review=False):
    """ Main function to generate plot of all files that can be found in
    data dir.

    click is a bool that triggers spacepy.plot.utils.EventClicker walk through
    of event selection.
    review is a string of the file to review.

    These options cannot be used together. """
    # Get colormesh that we'll paint out nan's from:
    cmap = matplotlib.cm.get_cmap('jet')
    cmap.set_bad(color='black')

    plotting_at = 0
    # First things first we need to open the arrays and make the pcolormeshes.
    for f in glob.glob('../data/flux_event*'):
        m = re.match(r'^flux_event_(Chan[RPT]{{1}}).pickle{}$'\
                     .format(sys.version_info[0]),
                     os.path.basename(f))
        if m is not None:
            varname = m.group(1)
        else:
            # Due to similar file names, this may sometimes be necessary:
            continue
        # Open the arrays:
        with open(f, 'rb') as fp:
            arrs = pickle.load(fp)

        # Flux:
        flux = arrs['flux']
        # We don't need the flux delta:
        # # Flux delta:
        # dflux = arrs['dflux']
        # Datetime/epoch:
        epoch = arrs['epoch']
        # Energy bins:
        energy = arrs['energy']

        # Need to flip all of these if we're working with ChanT:
        if varname == 'ChanT':
            flux = flux[:, ::-1]
            # dflux = dflux[:, ::-1]
            energy = energy[:, ::-1]

        # We need to get the max size of array (because it changes when the
        # binning table changes.  See 2018-09-03/05).
        # np.argmax gets the index of the max value.
        first_nonnan = numpy.where(~numpy.isnan(energy[numpy.argmax(numpy.sum(~numpy.isnan(energy), axis=1))]) == True)[0][0]
        lenn = numpy.sum(~numpy.isnan(energy[numpy.argmax(numpy.sum(~numpy.isnan(energy), axis=1))]))
        last_nonnan = first_nonnan + lenn

        # Now we can finally plot these:
        plt.pcolormesh(epoch,
                       energy[0, first_nonnan:last_nonnan] * 10**plotting_at,
                       flux[:, first_nonnan:last_nonnan].T,
                       cmap=cmap,
                       norm=matplotlib.colors.LogNorm(),
                       shading='flat',
                       edgecolors='None',
                       rasterized=True)

        # This essentially just shifts up the plot enough so that we can see
        # each one when we review the events on one canvas.
        # THIS IS NEEDED because EventClicker is made to work on one axis
        # (matplotlib object) and doesn't yet work over multiple.
        # import pdb; pdb.set_trace()
        plotting_at += numpy.ceil(numpy.log10(energy[0, last_nonnan - 1]) \
                                  - numpy.log10(energy[0, first_nonnan])) + 1

    plt.colorbar()
    plt.ylim((10, 10**plotting_at))
    plt.yscale('log')

    # Now we can call the review or clickthrough function:
    if review is False:
        clickthrough(events_file)
    else:
        reviewthrough(events_file, plotting_at)

def clickthrough(events_file):
    # This is preliminary, but might do max and half max..?
    c = spacepy.plot.utils.EventClicker(n_phases=2,
                                        interval=datetime.timedelta(days=30))
    c.analyze()

    if os.path.exists(events_file):
        os.rename(events_file, os.path.join(os.path.dirname(events_file),
                                            os.path.basename(events_file).split('.')[0] \
                                            + '.pickle{}'.format(sys.version_info[0]) \
                                            + '.back-{}'.format(datetime\
                                                                .datetime\
                                                                .now()\
                                                                .strftime('%Y%m%d%H%M'))))
    with open(events_file) as fp:
        pickle.dump(c.get_events(), fp)

def reviewthrough(events_file, plotting_at):
    with open(events_file, 'rb') as fp:
        e = pickle.load(fp)
        # Note that e is like: [((t0_start, y0_start), (t0_stop, y0_stop)),
        #                       ((t1_start, y1_start), (t1_stop, y1_stop)), ...]

    # Now, for each event (row) in e:
    for i, event in enumerate(e):
        startx = event[0, 0]
        stopx = event[1, 0]
        plt.vlines(startx, 10, 10**plotting_at, color='red')
        plt.vlines(stopx, 10, 10**plotting_at, color='cyan')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', help='enable script for event review',
                        dest='review',
                        action='store_true')
    args = parser.parse_args()
    if args.review is True:
        main(EVENTS_FILE, args.review)
    else:
        main(EVENTS_FILE)
