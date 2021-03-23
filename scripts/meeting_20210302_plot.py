import matplotlib as mpl   # nopep8
# setup matplotlib
mpl.use('Agg')   # nopep8
import matplotlib.pyplot as plt   # nopep8
import numpy as np   # nopep8

import data   # nopep8


d = 'psp_isois-epilo_l2-ic'
d_keys = data.DATASETS[d][1]  # ChanP


def main():
    plt.figure(figsize=(10, 10))
    plt.xlabel('Energy (MeV)')
    plt.xscale('log')
    plt.xlim(10e-3, 1e2)
    plt.ylabel('Flux (per MeV)')
    plt.yscale('log')
    plt.ylim(1e-6, 1e4)

    r = list()
    for i, e in enumerate(data.read_data('joyce-apj-tab2')):
        # unpack relevant vars
        epoch = e[d][d_keys['epoch']]
        flux = e[d][d_keys['flux']]
        flux_unc = e[d][d_keys['flux_unc']]
        energy = e[d][d_keys['energy']]
        energy_unc_plus = e[d][d_keys['energy_unc_plus']]
        energy_unc_minus = e[d][d_keys['energy_unc_minus']]

        # get only lookdirs of interest
        lookdir_idx = np.ones((80), dtype=bool)
        # lookdir_idx[20:50] = True
        lookdir_idx[np.array((25, 31, 34, 35, 44))] = False

        # select only what we're interested in
        flux = flux[:, lookdir_idx, :]

        # print(flux.shape)
        # print(np.nonzero(lookdir_idx))

        flux_unc = flux_unc[:, lookdir_idx, :]
        energy = energy[:, lookdir_idx, :]
        energy_unc_plus = energy_unc_plus[:, lookdir_idx, :]
        energy_unc_minus = energy_unc_minus[:, lookdir_idx, :]

        # sanity check
        assert(np.all((energy == energy[0:1, 0:1, :])[~np.isnan(energy)]))
        assert(np.all((energy_unc_plus == energy_unc_plus[0:1, 0:1, :])[~np.isnan(energy_unc_plus)]))  # nopep8
        assert(np.all((energy_unc_minus == energy_unc_minus[0:1, 0:1, :])[~np.isnan(energy_unc_minus)]))  # nopep8

        # collapse all arrays into just 1d (shouldn't techincally matter for energy-related vars)
        flux = np.nanmean(flux, axis=1)
        flux = np.nanmean(flux, axis=0)

        try:
            plt.plot(energy[0, 0, :] / 1e3, flux * 1e3, '.', c='grey')
        except:
            import pdb
            pdb.set_trace()

    plt.tight_layout()
    plt.savefig('../figures/meeting_20210302_plot.png')

    return r


if __name__ == '__main__':
    main()
