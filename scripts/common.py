# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 01:21:56 2020

@author: asher
"""

import numpy



PLOTTING_XLIM_LOWER = 30
PLOTTING_XLIM_UPPER = 500
PLOTTING_YLIM_LOWER = 1e-3
PLOTTING_YLIM_UPPER = 10


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


    Returns as (lower_bound, upper_bound)."""
    # Fitting parameters:
    if varname == 'ChanT':
        return 37, 400
    elif varname == 'ChanR':
        return 95, 200
    elif varname == 'ChanP':
        return 95, 305
