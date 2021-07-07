#!/usr/bin/env python3
""" Data interface for the thesis. """


import argparse
import datetime
import glob
import os


import data


d2t, t2d, d2s = data.d2t, data.t2d, data.d2s


class Data(data.Data):
    # replace datasets with truncated datasets including only ChanP but with epoch_delta
    datasets = {
        'psp_isois-epilo_l2-ic': {
            'ChanP': {
                'epoch': 'Epoch_ChanP',
                'epoch_delta': 'Epoch_ChanP_DELTA',  # add epoch_delta for other analysis
                'flux': 'H_Flux_ChanP',
                'flux_unc': 'H_Flux_ChanP_DELTA',
                'pa': 'PA_ChanP',
                # 'sa': 'SA_ChanP',  # no spiral angle in isois v1 cdfs (v2?)
                'energy': 'H_ChanP_Energy',
                'energy_unc_plus': 'H_ChanP_Energy_DELTAPLUS',
                'energy_unc_minus': 'H_ChanP_Energy_DELTAMINUS'
            }
        }
    }

    # remove the eventset this module is meant to handle
    ignore_globstr = {
        s for s in data.Data.ignore_globstr if s != 'joyce-apj-tab2'
    }

    # access release 2 datafiles
    def _get_files(self, d, strt, stop):
        """Override super method of _get_files to use new source directory."""
        daterange = [strt.date() + datetime.timedelta(days=d)
                     for d in range((stop.date() - strt.date()).days + 1)]
        # sanity check
        assert(strt.date() in daterange)
        assert(stop.date() in daterange)

        files = list()
        for date in daterange:
            daystr = date.strftime('%Y%m%d')
            ftoday = glob.glob(
                os.path.join(
                    '/home/share/data/data_public/archive/release_02/EPILo/level2/',
                    d + '_' + daystr + '*'
                )
            )
            if len(ftoday) > 0:
                files.append(ftoday[-1])
        return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script form of data module.  Use to make data caches.')
    parser.add_argument('globstr', type=str,
                        help='glob str for eventtimes file')

    args = parser.parse_args()

    d = Data(args.globstr)
    d.read_data(use_cache=False)
