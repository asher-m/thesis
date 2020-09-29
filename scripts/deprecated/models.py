#!/usr/bin/env python
"""
Various models for fitting.
"""

import numpy



def fisk_2008_eq38(e, e_0, e_b, j_0):
    """
    Fisk & Gloeckler 2008 Eq. 38
    Note that I've added a translation parameter to the exponential.
    """
    return j_0 * e**-1.5 * numpy.exp(-(e - e_b) / e_0)

def fisk_2008_eq38_modified(e, j_0, k):
    """
    Fisk & Gloeckler 2008 Eq. 38
    Note that I've added a translation parameter to the exponential.
    Changed a bit.
    """
    return j_0 * e**k

def fisk_2008_eq38_modified_centered(e, j_0, e_0, k):
    """
    Fisk & Gloeckler 2008 Eq. 38
    Note that I've added a translation parameter to the exponential.
    Changed a bit.
    """
    return j_0 * (e - e_0)**k
