# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 01:21:56 2020

@author: asher
"""

import numpy



# This can be ChanR, ChanT, or ChanP
VAR = "ChanR"

assert VAR in ['ChanT', 'ChanR', 'ChanP']

# Fitting parameters:
FIT_TRUNK_LOWER = 100
FIT_TRUNK_UPPER = 180


def uncert_prop(inarr, axis):
    """ Propagate the uncertainty of numbers on some axis when averaging down
    along that axis. """
    # Uncertainty carries like addition of quadrature divided by number of
    # things.
    return numpy.sqrt(numpy.nansum(inarr**2, axis=axis)) \
        / numpy.sum(numpy.invert(numpy.isnan(inarr)), axis=axis)
    # Alternatively, we may need to use:
    # return numpy.sqrt(numpy.nansum(inarr**2, axis=axis))
