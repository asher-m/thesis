# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 01:21:56 2020

@author: asher
"""
import collections
import collections.abc
import numpy

from cycler import cycler



PLOTTING_XLIM_LOWER = 30
PLOTTING_XLIM_UPPER = 500
PLOTTING_YLIM_LOWER = 1e-3
PLOTTING_YLIM_UPPER = 10

FITTING_HOW = {'ChanT': [
                            {  # Blank dict so we still make the normal plot.
                                },
                            {  # Fit of full time range, trunc'ed e range:
                                'e_range':(95, 160),
                                },
                            ],
               'ChanP': [
                   {  # Blank dict so we still make the normal plot.
                       },
                   ],
               'ChanR': [
                   # {  # Leave this dict commented out for absolutely no plots.
                   #     }
                   ]
               }
""" Dictionary of how to plot each varname.  All plots for a particular
varname are produced.  Valid keys are:
    'e_range'
    't_range'
    --- Nothing else has been implemented.

These parameters override parameters in energy_trunc and are used in the fit
script. """

REDS_RAW = ['#B22222',
            '#FF0000',  # Kinda hard to see.
            '#8B0000',
            '#800000',
            '#FF6347',
            # '#FF4500'  # Also kinda hard to see.
            ]
REDS = (cycler('color', REDS_RAW) * cycler('linestyle', ['-']))
""" Reds for use in plotting a particular species or type. """

GREENS_RAW = [
    '#7FFF00',
    '#32CD32',
    '#00FF00',
    '#228B22',
    '#008000',
    '#006400']
GREENS = (cycler('color', GREENS_RAW) * cycler('linestyle', ['-']))
""" Greens for use in plotting a particular species or type. """

BLUES_RAW = [
    '#4169E1',
    '#0000FF',
    '#0000CD',
    '#00008B',
    '#000080',
    '#191970',
    '#8A2BE2',
    '#4B0082',
    '#00BFFF',
    '#1E90FF']
BLUES = (cycler('color', BLUES_RAW) * cycler('linestyle', ['-']))
""" Blues for use in plotting a particular species or type. """


def get_eta_squared(inarr):
    """ Function to perform array mean(s) for mag data.
    Calculate eta-squared. """
    # Using nanmean to be safe here, but probably strictly NOT
    # be necessary if the file is in-spec.
    B0 = numpy.nanmean(numpy.sqrt(numpy.sum(inarr**2, axis=1)))
    B = numpy.sqrt(numpy.sum(inarr**2, axis=1))
    return numpy.nanmean((B - B0)**2 / B0**2)

def uncert_prop(inarr, axis):
    """ Propagate the uncertainty of numbers on some axis when averaging down
    along that axis. """
    # Uncertainty carries like addition of quadrature divided by number of
    # things.
    return numpy.sqrt(numpy.nansum(inarr**2, axis=axis)) \
        / numpy.sum(numpy.invert(numpy.isnan(inarr)), axis=axis)
    # Alternatively, we may need to use:
    # return numpy.sqrt(numpy.nansum(inarr**2, axis=axis))

def nan_cut(inarr):
    """ Check that all (non-NaN) elements of array are constant along zeroth
    axis (0), then cuts out just the non-NaN elements from the first axis (1)
    and hands them back along with the indices of the first and last non-NaN
    elements so the same can be done to associated arrays.

    Returns the input array cut down as described (or not, see return args),
    and the indices to cut associated arrays in the same manner,
    (indices of first and last non-NaNs along first axis (1)). """
    # import pdb; pdb.set_trace()

    # Make sure energy is consistent within this time range.
    # This is a different condition than tried in the mean-concat script,
    # because that checks within hours.  This checks between hours
    # (and therefore, possibly files as well).
    # ALSO, we don't need to index along look direction axis because we
    # already checked in the mean-concat script.
    assert numpy.all((inarr == inarr[0])[~numpy.isnan(inarr)])
    # If this works (because we have the same binning in the file)
    # we can just use the first set of not-all-NaNs bins
    # (because they're all the same):
    nonnan = numpy.where(numpy.any(~numpy.isnan(inarr), axis=1) \
                         == True)[0][0]
    inarr = inarr[nonnan]

    # Lastly, we can count the number of non-NaNs we have, so we know where
    # to stop the array trunking (so we don't try to plot NaNs):
    lenn = numpy.sum(~numpy.isnan(inarr))
    # And get the first not-nan so we know where to cut off the first few
    # NaNs, (for ChanT, for example):
    first_nonnan = numpy.where(~numpy.isnan(inarr) == True)[0][0]
    last_nonnan = first_nonnan + lenn
    inarr_cutdown = inarr[first_nonnan:last_nonnan]

    return inarr, inarr_cutdown, first_nonnan, last_nonnan

def energy_trunc(varname):
    """ Function to return auto bounds on energy range for fits for each var
    type.


    Returns as (lower_bound, upper_bound).

    Some of these parameters are ignored when fitting in the fit script.  See
    the fit function in there. """
    # Fitting parameters:
    if varname == 'ChanT':
        return 37, 160
    elif varname == 'ChanR':
        return 95, 160
    elif varname == 'ChanP':
        # From talking to Jon, looks like P should be ~95 to 200.
        return 95, 160

def cut_like(inarr, like, *arrs, cutside='left', side='left', idx=False):
    """ A function to cut a bunch of arrays like we'd cut some array when
    searching for the lowest/highest allowable index.

    "Left" cutting means you're retaining the "right" side of the array.
    "Right" cutting means you're retaining the "left" side of the array.
    "Both" cutting means you're cutting off the "left" and "right" side of
    the array and you're interested in the middle.  This also requires
    "like" to be a 2 element iterable.

    If "idx" is keyword'ed to "True", "like" is assumed to be the index (or
    indices) where to cut.

    Specifically, like when we're cutting things in the time domain and we
    need to cut like 10 arrays all the same way with the same indices. """
    assert cutside in ('left', 'right', 'both')
    assert isinstance(idx, bool)
    ret = []
    if cutside == 'both':
        assert isinstance(like, collections.abc.Iterable) and len(like) == 2
        if idx is False:
            cutlow = numpy.searchsorted(inarr, like[0], side=side)
            cuthigh = numpy.searchsorted(inarr, like[1], side=side)
        else:
            cutlow, cuthigh = like
        ret.append(inarr[cutlow:cuthigh])
        for a in arrs:
            ret.append(a[cutlow:cuthigh])
    else:
        if idx is False:
            cutidx = numpy.searchsorted(inarr, like, side=side)
        else:
            cutidx = like
        if cutside == 'left':
            ret.append(inarr[cutidx:])
            for a in arrs:
                ret.append(a[cutidx:])
        else:
            ret.append(inarr[:cutidx])
            for a in arrs:
                ret.append(a[:cutidx])
    return tuple(ret)

def get_nearest(to_value, from_set):
    """ Function that takes a value, and gets the index nearest to
    that value from_set.

    I wish I could figure out how to write this to be faster because it's
    interesting, but it's not really worth my time... """
    if isinstance(to_value, numpy.ndarray):
        idx_min = []
        for v in to_value:
            idx_min.append(numpy.argmin(numpy.abs(from_set - v)))
        idx_min = numpy.array(idx_min)
    else:
        idx_min = numpy.argmin(numpy.abs(from_set - to_value))
    return idx_min
