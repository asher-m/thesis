import numpy as np

import data

d = 'psp_isois-epilo_l2-ic'
d_keys = data.DATASETS[d][1]

def main():
    for i, e in enumerate(data.read_data('joyce-apj-tab2')):
        # unpack relevant vars
        epoch = e[d][d_keys['epoch']]
        flux = e[d][d_keys['flux']]
        flux_unc = e[d][d_keys['flux_unc']]
        energy = e[d][d_keys['energy']]
        energy_unc_plus = e[d][d_keys['energy_unc_plus']]
        energy_unc_minus = e[d][d_keys['energy_unc_minus']]

        # cut down to only sectors of interest
        flux = flux[:, np.arange(20, 50), :]

        # exclude 25, 31, 34, 35, 44
        ex = np.array([25, 31, 34, 35, 44]) - 20  # minus 20 to realign with array
        flux[:, ex, :] = np.nan  # replace with nan

        flux = np.nanmean(flux, axis=(0, 1))

        # need to replace nan(s) with 0(s) at this point (for np.average)
        flux[np.isnan(flux)] = 0

        # sanity check
        assert(np.all((energy_unc_plus == energy_unc_plus[0, :, :])[~np.isnan(energy_unc_plus)]))
        assert(np.all((energy_unc_minus == energy_unc_minus[0, :, :])[~np.isnan(energy_unc_minus)]))
        weights = np.nanmean(energy_unc_plus[0], axis=0) + np.nanmean(energy_unc_minus[0], axis=0)
        weights[np.isnan(weights)] = 0  # again replace with 0

        avgflux = np.average(flux, weights=weights)

        print(f"Event {i}, average flux: {avgflux * 1e3}")


if __name__ == '__main__':
    main()
