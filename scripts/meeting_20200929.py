#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy
import scipy.optimize

import spacepy
import spacepy.pycdf
import spacepy.datamanager

import data


FLUX_LIMLOW, FLUX_LIMHIGH = 1e-3, 1e4
P0 = numpy.array([1e10, -3/2])


def rebin(epoch, flux, pa, sa, cadence=None, flux_unc=None):
    """ Rebin fluxes. 
    
    Cadence in nanoseconds; default None averages entire range. """
    # Look dir:
    flux_omni = spacepy.datamanager.rebin(flux, sa, [0, 180], axis=1)
    lookdir_bins = numpy.linspace(0, 180, 4)
    flux_pa = spacepy.datamanager.rebin(flux, pa, lookdir_bins, axis=1)
    flux_sa = spacepy.datamanager.rebin(flux, sa, lookdir_bins, axis=1)
    if flux_unc is not None:
        flux_unc_omni = spacepy.datamanager.rebin(
            flux_unc, sa, [0, 180], axis=1, bintype='unc')
        flux_unc_pa = spacepy.datamanager.rebin(
            flux_unc, pa, lookdir_bins, axis=1, bintype='unc')
        flux_unc_sa = spacepy.datamanager.rebin(
            flux_unc, sa, lookdir_bins, axis=1, bintype='unc')

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
    if flux_unc is not None:
        flux_unc_omni = spacepy.datamanager.rebin(
            flux_unc_omni, epoch, epoch_bins, axis=0, bintype='unc')
        flux_unc_pa = spacepy.datamanager.rebin(
            flux_unc_pa, epoch, epoch_bins, axis=0, bintype='unc')
        flux_unc_sa = spacepy.datamanager.rebin(
            flux_unc_sa, epoch, epoch_bins, axis=0, bintype='unc')

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

    if flux_unc is not None:
        flux_unc_omni = numpy.squeeze(flux_unc_omni, axis=1)
        flux_unc_pa = {
            'par': flux_unc_pa[:, 0],
            'perp': flux_unc_pa[:, 1],
            'apar': flux_unc_pa[:, 2]
        }
        flux_unc_sa = {
            'par': flux_unc_sa[:, 0],
            'perp': flux_unc_sa[:, 1],
            'apar': flux_unc_sa[:, 2]
        }
        return epoch_fake, flux_omni, flux_pa, flux_sa, flux_unc_omni, flux_unc_pa, flux_unc_sa
    else:
    return epoch_fake, flux_omni, flux_pa, flux_sa


def spectrogram(epoch, flux_omni, flux_pa, flux_sa, energy, pname, keepfig=False):
    """ Make a sprectrogram for one event.

    flux_pa and flux_sa are dicts keyed by 'par', 'apar', 'perp'. """
    # color map
    cmap = matplotlib.cm.get_cmap('jet')
    cmap.set_bad(color='black')
    # log norm
    norm = matplotlib.colors.LogNorm(vmin=FLUX_LIMLOW, vmax=FLUX_LIMHIGH)

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
    for i, k in enumerate(('par', 'perp', 'apar')):
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
    for i, k in enumerate(('par', 'perp', 'apar')):
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


def spectrum(epoch, flux_omni, flux_unc_omni, flux_pa, flux_unc_pa, flux_sa, flux_unc_sa, energy, pname, keepfig=False):
    """ Make a sprectrogram for one event.

    flux_pa and flux_sa are dicts keyed by 'par', 'apar', 'perp'. """
    def label_axis(ax):
        ax.set_yscale('log')
        ax.set_ylabel('j')
        ax.set_ylim(FLUX_LIMLOW, FLUX_LIMHIGH)
        ax.set_xscale('log')
        ax.set_xlabel('Energy (keV)')
        ax.set_xlim(20, 200)

    energy_bounds = ((37, 160), (95, 160))

    fig, axes = plt.subplots(nrows=3, ncols=3,
                             figsize=(20, 20),
                             sharex=True, sharey=True)

    # Check to make sure energy is consistent across time:
    assert(numpy.all(energy == energy[0]))

    erange = numpy.linspace(energy[0, 0, 0], energy[0, 0, -1])

    # Sharing x and y, so set them smartly:
    label_axis(axes[0, 0])

    for j, eset in enumerate(energy_bounds):
        elower = numpy.searchsorted(energy[0, 0], eset[0])
        eupper = numpy.searchsorted(energy[0, 0], eset[1])

        print(j, ';', eset, ':', elower, eupper)
        # import pdb; pdb.set_trace()


        # First plot omni:
        ax = axes[0, 1]
        popt, pcov = scipy.optimize.curve_fit(
            data.model,
            energy[0, 0, elower:eupper],
            flux_omni[0, elower:eupper],
            sigma=flux_unc_omni[0, elower:eupper],
            absolute_sigma=True,
            p0=P0
        )
        ax.plot(erange, data.model(erange, *popt))
        if j == 0:  # if first time around do the plot:
            ax.errorbar(
                energy[0, 0],
                flux_omni[0],
                yerr=flux_unc_omni[0]
            )
        print(j, ';', eset, ':', *popt)

        # Now plot pa:
        for i, k in enumerate(('par', 'perp', 'apar')):
            ax = axes[1, i]
            popt, pcov = scipy.optimize.curve_fit(
                data.model,
                energy[0, 0, elower:eupper],
                flux_pa[k][0, elower:eupper],
                sigma=flux_unc_pa[k][0, elower:eupper],
                absolute_sigma=True,
                p0=P0
            )
            ax.plot(erange, data.model(erange, *popt))
            if j == 0:
                ax.errorbar(
                    energy[0, 0],
                    flux_pa[k][0],
                    yerr=flux_unc_pa[k][0]
                )

        # Now plot sa:
        for i, k in enumerate(('par', 'perp', 'apar')):
            ax = axes[2, i]
            popt, pcov = scipy.optimize.curve_fit(
                data.model,
                energy[0, 0, elower:eupper],
                flux_sa[k][0, elower:eupper],
                sigma=flux_unc_sa[k][0, elower:eupper],
                absolute_sigma=True,
                p0=P0
            )
            ax.plot(erange, data.model(erange, *popt))
            if j == 0:
                ax.errorbar(
                    energy[0, 0],
                    flux_sa[k][0],
                    yerr=flux_unc_sa[k][0]
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


                epoch_fake, flux_omni, flux_pa, flux_sa, flux_unc_omni, flux_unc_pa, flux_unc_sa = rebin(
                    epoch,
                    flux,
                    pa,
                    sa,
                    flux_unc=flux_unc
                )
            
                spectrum(
                    epoch_fake,
                    flux_omni,
                    flux_unc_omni,
                    flux_pa,
                    flux_unc_pa,
                    flux_sa, 
                    flux_unc_sa,
                    energy, 
                    'meeting_20200929/spectrum_event-{:02d}_{}_{}.png'.format(
                        i,
                        spacepy.pycdf.lib.tt2000_to_datetime(epoch[0]).strftime('%Y-%j'),
                        g['flux'].lower())
                )

    return eventdata


if __name__ == '__main__':
    eventdata = main()
