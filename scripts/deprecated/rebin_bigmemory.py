import datetime
import numpy
import os
import sys

import spacepy
import spacepy.datamanager
import spacepy.pycdf



def process_file(f):
    print('Starting file {}...'.format(os.path.basename(f)))
    r = dict()

    finfo = {'date': f.split('_')[3]}
    c = spacepy.pycdf.CDF(f)  # open cdf

    for v in ('ChanT', 'ChanP', 'ChanR'):
        r[v] = dict()

        epoch = c.raw_var('Epoch_{}'.format(v))[...]
        flux = c['H_Flux_{}'.format(v)][...]
        energy = c['H_{}_Energy'.format(v)][...]
        pa = c['PA_{}'.format(v)][...]

        flux[flux < 0] = numpy.nan
        energy[energy < 0] = numpy.nan
        pa[pa < 0] = numpy.nan

        # get edges
        edges_times, edges_lookdir, edges_energy = edges(finfo['date'])

        # do energy rebin
        if v == 'ChanR':
            import pdb; pdb.set_trace()
        flux = spacepy.datamanager.rebin(
            flux,
            energy,
            edges_energy,
            axis=2
        )
        # do pa rebin
        flux = spacepy.datamanager.rebin(
            flux,
            pa,
            edges_lookdir['omni'],
            axis=1
        )
        # do time rebin
        flux = spacepy.datamanager.rebin(
            flux,
            epoch,
            edges_times,
            axis=0
        )

        r[v]['flux'] = flux
        r[v]['epoch'] = edges(finfo['date'], epoch=True)

    c.close()

    return r
	
def master_isois_3_summary_energy_bins():
    """Create new energy bins for ISOIS level 3 summary.  Returns bins and pmvals.
    Bins are an array of the bin centers, and the plus/minus vals are an
    array of the p/mvals as [(mval, pval), (mval, pval), ...]; (the name
    'pmvals' is sort of a misnomer because it's really 'mpvals').
    """
    upper_energy_cutoff = 1e3
    lower_energy_cutoff = 0

    # From talking with Jon and Carol we want bins as low as 10 - 20 keV
    # and as high as the upper limit on EPIHi (800 MeV, iirc).
    # This works nicely as:
    base = 1  # MeV, around which the bins are centered.
    bins = base * numpy.sqrt(2) ** numpy.arange(-12, 20)

    edges = numpy.empty((bins.shape[0] + 1))
    edges.fill(numpy.nan)
    # Not super fast, but fast enough for this:
    edges[1:-1] = [numpy.sqrt(bins[i] * bins[i+1])
                   for i in range(len(bins) - 1)]
    edges[0] = lower_energy_cutoff
    edges[-1] = upper_energy_cutoff

    pmvals = numpy.array([(bins[i] - edges[i], edges[i + 1] - bins[i])
                          for i in range(len(bins))])

    return bins, pmvals


def master_isois_3_summary_lookdir_bins(lkdr='omni'):
    """ Create pitch angle bins for l3 summary. """
    lkdr = lkdr.lower()
    assert(lkdr in ('omni', 'pa', 'sa'))
    if lkdr == 'omni':
        return numpy.array([90]), numpy.array([[90, 90]])
    elif lkdr == 'pa' or lkdr == 'sa':
        edges = numpy.arange(0, 180 + 1, 20)
        bins = (edges[:-1] + edges[1:]) / 2
        pmvals = numpy.array([(bins[i] - edges[i], edges[i + 1] - bins[i])
                              for i in range(len(bins))])
        return bins, pmvals
    # Ehh, if somehow we got here just do this:
    return None, None
	
def edges(datestr, epoch=False):
    """ Shamelessly copied from level3.py at 1cb66199 (for which I'm the author).

    Make all edges for rebin, (sometimes from bin centers
    and pmvals, sometimes just from scratch according to
    cadence).

    Follows ordering of CDF depends.

    :param bool epoch: Yield epoch from calcualting time edges for CDF.
    """
    # Time bins: keep in mind this is an hourly product.
    today = datetime.datetime.strptime(datestr, '%Y%m%d')
    strt_tt2000 = spacepy.pycdf.lib.datetime_to_tt2000(today)
    stop_tt2000 = spacepy.pycdf.lib.datetime_to_tt2000(
        today + datetime.timedelta(days=1))
    # Minutes, the "plus-minus value" (so half bin spacing)
    delta_minutes = 30
    delta = delta_minutes * 60e9  # convert to nanoseconds
    # Time values centered in bin:
    times = numpy.arange(strt_tt2000 + delta, stop_tt2000, delta * 2)
    # If flagged for epoch, return now:
    if epoch is True:
        return times
    # By explicitly making the upper edge of the last bin the stop_tt2000 value
    # the leap second is captured in the last bin.  This is in contrast
    # to making the edges by adding the delta to all the values which risks
    # not capturing leap seconds in the last bin.
    edges_times = numpy.array(
        list(times - delta) + list((stop_tt2000,)))

    # PA bins, for now:
    edges_lookdir = dict()
    for lkdr in ('omni', 'pa', 'sa'):
        pabins, papmvals = master_isois_3_summary_lookdir_bins(
            lkdr)
        edges_lookdir[lkdr] = numpy.array(
            list((pabins[0] - papmvals[0, 0],)) + list(pabins + papmvals[:, 1]))

    # Energy bins:
    ebins, epmvals = master_isois_3_summary_energy_bins()
    ebins *= 1e3
    epmvals *= 1e3
    edges_energy = numpy.array(
        list((ebins[0] - epmvals[0, 0],)) + list(ebins + epmvals[:, 1]))

    return edges_times, edges_lookdir, edges_energy

if __name__ == '__main__':
    if len(sys.argv) > 1:
        f = sys.argv[1]
    else:
        f = 'psp_isois-epilo_l2-ic_20190404_v2.0.0.cdf'
    import pdb; pdb.set_trace()
    ## want to $ s <return> to step into rebin; look at top to see memory use
    process_file(f)
