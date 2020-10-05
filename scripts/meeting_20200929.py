#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy

import spacepy
import spacepy.pycdf
import spacepy.datamanager

import data


def rebin(epoch, flux, pa, sa, cadence=None):
    """ Rebin fluxes. 
    
    Cadence in nanoseconds; default None averages entire range. """
    # Look dir:
    flux_omni = spacepy.datamanager.rebin(flux, pa, [0, 180], axis=1)
    lookdir_bins = numpy.linspace(0, 180, 4)
    flux_pa = spacepy.datamanager.rebin(flux, pa, lookdir_bins, axis=1)
    flux_sa = spacepy.datamanager.rebin(flux, sa, lookdir_bins, axis=1)

    # Epoch: basically need hours or else it's too hard to plot
    if cadence is not None:
        epoch_fake = numpy.arange(epoch[0] + cadence / 2, epoch[-1], cadence)
        epoch_bins = list(epoch_fake - cadence) + list((epoch[-1],))
    else:
        epoch_fake = numpy.mean(epoch)
        epoch_bins = [epoch[0], epoch[-1]]
    flux_omni = spacepy.datamanager.rebin(flux_omni, epoch, epoch_bins, axis=0)
    flux_pa = spacepy.datamanager.rebin(flux_pa, epoch, epoch_bins, axis=0)
    flux_sa = spacepy.datamanager.rebin(flux_sa, epoch, epoch_bins, axis=0)

    # Now break into appropriate stuff:
    flux_omni = numpy.squeeze(flux_omni, axis=1)
    flux_pa = {
        'par': flux_pa[:, 0],
        'perp': flux_pa[:, 1],
        'apar': flux_pa[:, 2]
    }
    flux_sa = {
        'par': flux_sa[:, 0],
        'perp': flux_sa[:, 1],
        'apar': flux_sa[:, 2]
    }
    return epoch_fake, flux_omni, flux_pa, flux_sa


def spectrogram(epoch, flux_omni, flux_pa, flux_sa, energy, pname, keepfig=False):
    """ Make a sprectrogram for one event.

    flux_pa and flux_sa are dicts keyed by 'par', 'apar', 'perp'. """
    # color map
    cmap = matplotlib.cm.get_cmap('jet')
    cmap.set_bad(color='black')
    # log norm
    norm = matplotlib.colors.LogNorm(vmin=1e-4, vmax=1e6)

    fig, axes = plt.subplots(nrows=3, ncols=3,
                             figsize=(20, 20),
                             sharex=True, sharey=True)

    # Check to make sure energy is consistent across time:
    assert(numpy.all(energy == energy[0]))

    # Change epoch back to datetime so we can plot nicely;
    # need to do with for loop because spacepy.pycdf.lib.tt2000_to_datetime
    # cannot handle vectors:
    epoch = [spacepy.pycdf.lib.tt2000_to_datetime(int(e)) for e in epoch]

    # Sharing x and y, so set them smartly:
    ax = axes[0, 0]
    ax.set_yscale('log')

    # First plot omni:
    ax = axes[0, 1]
    ax.pcolormesh(
        epoch,
        energy[0, 0],
        numpy.ma.array(flux_omni, mask=numpy.isnan(flux_omni)).T,
        cmap=cmap,
        norm=norm,
        shading='flat',
        edgecolors='None',
        rasterized=True
    )

    # Now plot pa:
    for i, k in enumerate(flux_pa):
        ax = axes[1, i]
        ax.pcolormesh(
            epoch,
            energy[0, 0],
            numpy.ma.array(flux_pa[k], mask=numpy.isnan(flux_pa[k])).T,
            cmap=cmap,
            norm=norm,
            shading='flat',
            edgecolors='None',
            rasterized=True
        )

    # Now plot sa:
    for i, k in enumerate(flux_sa):
        ax = axes[2, i]
        ax.pcolormesh(
            epoch,
            energy[0, 0],
            numpy.ma.array(flux_sa[k], mask=numpy.isnan(flux_sa[k])).T,
            cmap=cmap,
            norm=norm,
            shading='flat',
            edgecolors='None',
            rasterized=True
        )

    # Save:
    fig.tight_layout()
    fig.savefig(pname)

    if keepfig is not False:
        return fig, axes
    else:
        plt.close(fig)


def main():
    eventdata = data.read_data(verbose=True)
    for i, e in enumerate(eventdata):
        for d in data.DATASETS:
            for g in data.DATASETS[d]:
                epoch = e[d][g['epoch']]
                flux = e[d][g['flux']]
                flux_unc = e[d][g['flux_unc']]
                pa = e[d][g['pa']]
                sa = e[d][g['sa']]
                energy = e[d][g['energy']]

                # Check if we can actually handle this event:
                if not len(epoch) > 1:
                    continue

                epoch_fake, flux_omni, flux_pa, flux_sa = rebin(
                    epoch,
                    flux,
                    pa,
                    sa,
                    cadence=60 * 60e9  # 1 hour in nanoseconds
                )
            
                spectrogram(
                    epoch_fake,
                    flux_omni,
                    flux_pa,
                    flux_sa, 
                    energy, 
                    'meeting_20200929/spectrogram_event-{:02d}_{}_{}.png'.format(
                        i,
                        spacepy.pycdf.lib.tt2000_to_datetime(epoch[0]).strftime('%Y-%j'),
                        g['flux'].lower())
                )


    return eventdata


if __name__ == '__main__':
    eventdata = main()
