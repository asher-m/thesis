#!/usr/bin/env python
"""
Script to run through EPILo files and average down H_ChanR|P|T
variables along some axis.
"""

# There are ie ic pe pc files
# Looks like I should use 'ic' files

import numpy
import os
import pickle
import sys

import isois
import spacepy.pycdf



VARNAMES = ('ChanT', 'ChanP', 'ChanR')

R_0 = 6.975e5
""" Solar radius in km """
AU = 1.495978707e8
""" AU in km """
b = 4 * R_0
""" Corotation boundary """
vel = 400
""" Solar wind velocity in km/s """
P = 27.2753
""" Rotational period of the sun in days """
Omega = (2 * numpy.pi) / (P * 24 * 3600)
""" Angular velocity of the sun in rads/s """

def main(b=b, vel=vel, dump=True):
    with open('../data/RTN.pickle{}'.format(sys.version_info[0]),
              'rb') as fp:
        d = pickle.load(fp)

    for v in VARNAMES:
        d['Footpoint_{}'.format(v)] = ((((Omega * (AU * d['R_{}'.format(v)] - b) / vel) * (180 / numpy.pi))
                                        + d['Lon_{}'.format(v)])
                                       % 360)

    if dump is True:
        with open('../data/footpoint.pickle{}'.format(sys.version_info[0]),
                  'rb') as fp:
            pickle.dump(d, fp)
    else:
        return d

if __name__ == "__main__":
    main()
