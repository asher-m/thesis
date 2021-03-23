import numpy as np

import data

d = 'psp_isois-epilo_l2-ic'
d_keys = data.DATASETS[d][1]  # ChanP


def main():
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
        
        print(flux.shape)
        print(np.nonzero(lookdir_idx))

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
        # flux = np.nanmean(flux, axis=(0, 1))
        energy = np.nanmean(energy, axis=(0, 1))
        energy_unc_plus = np.nanmean(energy_unc_plus, axis=(0, 1))
        energy_unc_minus = np.nanmean(energy_unc_minus, axis=(0, 1))

        # get hi/lo indices
        idx_lo = np.searchsorted(energy, 60)
        idx_hi = np.searchsorted(energy, 800, side='right')
        print(idx_lo)
        print(idx_hi)
        idx_lo = 0
        idx_hi = 25
        # maybe check energy - energy_unc_minus and energy + energy_unc_plus (so bin edges instead of centers)
        # check if these indices are correct

        # select only energies of interest
        flux = flux[idx_lo:idx_hi]
        flux_unc = flux_unc[idx_lo:idx_hi]
        # energy = energy[idx_lo:idx_hi]  # not needed anymore
        energy_unc_plus = energy_unc_plus[idx_lo:idx_hi]
        energy_unc_minus = energy_unc_minus[idx_lo:idx_hi]

        # more sanity checks
        assert(not np.any(np.isnan(energy_unc_plus)))
        assert(not np.any(np.isnan(energy_unc_minus)))

        # calculate weights
        weights = energy_unc_plus + energy_unc_minus

        # calculate average flux
        avgflux = np.average(flux, weights=weights) * 1e3

        # out
        print("Event {i:02d}, average flux: {avgflux:4e}".format(**dict(i=i, avgflux=avgflux)))  # nopep8
        r.append(avgflux)

    return r


if __name__ == '__main__':
    main()
