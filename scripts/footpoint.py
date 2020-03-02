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
""" Frozen-in radius in km """
vel = 400
""" Solar wind velocity in km/s """
P = 27.2753
""" Rotational period of the sun in days """
Omega = (2 * numpy.pi) / (P * 24 * 3600)
""" Angular velocity of the sun in rads/s """

def main():
    files = isois.get_latest('psp_isois-epilo_l2-ic')
    # Some test cases:
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:1]
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:10] # just first 10
    # files = isois.get_latest('psp_isois-epilo_l2-ic')[:100]

    d = {}

    for v in VARNAMES:
        d['R_{}'.format(v)] = []
        d['Lat_{}'.format(v)] = []
        d['Lon_{}'.format(v)] = []
        d['Epoch_{}'.format(v)] = []

    for ff in files:
        print('Starting file {}...'.format(os.path.basename(ff)))

        # Open each file:
        with spacepy.pycdf.CDF(ff) as f:
            for v in VARNAMES:
                d['R_{}'.format(v)].append(f['HGC_R_{}'.format(v)][...])
                d['Lat_{}'.format(v)].append(f['HGC_Lat_{}'.format(v)][...])
                d['Lon_{}'.format(v)].append(f['HGC_Lon_{}'.format(v)][...])
                d['Epoch_{}'.format(v)].append(f['Epoch_{}'.format(v)][...])

    for v in VARNAMES:
        d['R_{}'.format(v)] = numpy.concatenate(d['R_{}'.format(v)])
        d['Lat_{}'.format(v)] = numpy.concatenate(d['Lat_{}'.format(v)])
        d['Lon_{}'.format(v)] = numpy.concatenate(d['Lon_{}'.format(v)])
        d['Epoch_{}'.format(v)] = numpy.concatenate(d['Epoch_{}'.format(v)])

    for v in VARNAMES:
        d['Footpoint_{}'.format(v)] = ((((Omega * (AU * d['R_{}'.format(v)] - b) / vel) * (180 / numpy.pi))
                                + d['Lon_{}'.format(v)])
                               % 360)

    with open('../data/footpoints.pickle{}'.format(sys.version_info[0]),
              'wb') as fp:
        pickle.dump(d, fp)

if __name__ == "__main__":
    main()
