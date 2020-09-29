#!/usr/bin/env python
"""
Perform a fit on the data according to some model, as provided.
"""

import matplotlib.pyplot as plt
import numpy
import pickle
from scipy.ndimage.filters import gaussian_filter1d
import types

# This import is a bit ugly, but it's less ugly than retyping this every time:
from common import cut_like, EVENTS_FILE, MAG_FILE, PLOTTING_FIGSIZE



def main(events_file, mag_file):
    # Open the saved event times:
    with open(events_file, 'rb') as fp:
        e = pickle.load(fp)
        # Note that e is like: [((t0_start, y0_start), (t0_stop, y0_stop)),
        #                       ((t1_start, y1_start), (t1_stop, y1_stop)), ...]

    with open(mag_file, 'rb') as fp:
        arrs = pickle.load(fp)
    mag = types.SimpleNamespace(**arrs)
    

    # Now, for each event (row) in e:
    for i, event in enumerate(e):
        # Cut all vars in time that we're looking at:
        starttime = event[0, 0]
        stoptime = event[1, 0]

        fig, axes = plt.subplots(nrows=2, ncols=1,
                                 figsize=(PLOTTING_FIGSIZE[0],
                                          PLOTTING_FIGSIZE[1]*2),
                                 sharex=True, sharey=True)
        # Determine if B is in/out or changes direction:
        c_mag_epoch, c_mag = cut_like(mag.epoch, (starttime, stoptime),
                                      mag.mag, cutside='both')
        for j, n in enumerate(("R", "T", "N")):
            axes[0].plot(c_mag_epoch, c_mag[:, j], label=n)
            axes[1].plot(c_mag_epoch, gaussian_filter1d(c_mag[:, j], 2), label=n)
        axes[0].plot(c_mag_epoch,
                     numpy.linalg.norm(c_mag, axis=1),
                     label='Norm')
        axes[1].plot(c_mag_epoch,
                     gaussian_filter1d(numpy.linalg.norm(c_mag, axis=1),
                                       2),
                     label='Norm')


        axes[0].set_title(f'Event ID {i:02d}: B field, raw')
        axes[1].set_title(f'Event ID {i:02d}: B field, filtered')
        axes[0].legend(loc=1)
        axes[1].legend(loc=1)
        plt.tight_layout()
        plt.savefig('../figures/B_RTN_{:02d}.png'.format(i))
        plt.savefig('../figures/pdf/B_RTN_{:02d}.pdf'.format(i))
        plt.close()

if __name__ == "__main__":
    main(EVENTS_FILE, MAG_FILE)

