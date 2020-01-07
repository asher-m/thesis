#!/usr/bin/env python
"""
Script to import EPILo data files averaged into some arrays of flux and datetime
or epoch and display using spacepy.plot.utils.EventClicker so events can be
identified.
"""

import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy
import pickle
import sys

import spacepy.plot

with open('datetime_and_flux.pickle{}'.format(sys.version_info[0]), 'rb') as fp:
    arrs = pickle.load(fp)

# Flux:
g = arrs['flux']
# Datetime/epoch:
t = arrs['epoch']

# Have to process out nans from g:
# g = numpy.where(numpy.isnan(g), 1e-20, g)

# Have to process out 0(s) from g:
# g = numpy.where(g == 0, 10**-4, g)

# Get colormesh that we'll paint out nan's from:
cmap = matplotlib.cm.get_cmap('jet')
cmap.set_bad(color='black')
# cmap.set_bad(color='gray')

energy = [
    69.8,
    84.5,
    91.1,
    98.0,
    110,
    132,
    160,
    198,
    286,
    493,
    886,
    1630,
    3030,
    5640,
    8320,
    ]

# Need transpose of g, because I guess that's how it always worked...
plt.pcolormesh(t, energy, g.T, cmap=cmap, norm=matplotlib.colors.LogNorm(), shading='flat', edgecolors='None', rasterized=True)
plt.yscale('log')
plt.ylim((10, 2000))
plt.colorbar()

# This is preliminary, but might do max and half max..?
c = spacepy.plot.utils.EventClicker(n_phases=2)
c.analyze()
