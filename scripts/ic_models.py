#!/usr/bin/env python
"""
Various models for fitting.
"""

import numpy



def fisk_2008_eq38(e, e_0, j_0):
    return j_0 * e**-1.5 * numpy.exp(-e / e_0)