import datetime
import glob
import os


import data


class Data(data.Data):
    datasets = {
        'psp_isois-epilo_l2-ic': [
            {
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
        ]
    }

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
