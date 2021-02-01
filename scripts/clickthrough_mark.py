# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import bz2
import datetime
import matplotlib.cm as cmp
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import pickle

import spacepy.plot
import spacepy.pycdf


def make_dt_lbls(dt):
    return dt.strftime('%F (%Y-%j)')

channel = 'ChanT'
t2d = np.vectorize(spacepy.pycdf.lib.tt2000_to_datetime)
d2t = np.vectorize(spacepy.pycdf.lib.datetime_to_tt2000)
d2s = np.vectorize(make_dt_lbls)

with bz2.BZ2File('../data/clickdata.pickle3.bz2', 'rb') as fp:
    d = pickle.load(fp)

cmap = cmp.get_cmap('jet')
cmap.set_bad(color='gray')
norm = clr.LogNorm()



plt.figure(figsize=(16, 9))
plt.pcolormesh(d['epoch'],
    d['energy'],
    d[channel]['flux'].T,
    cmap=cmap,
    norm=norm,
    shading='flat',
    edgecolors='None',
    rasterized=True
)

# make axis labels
strtday = spacepy.pycdf.lib.tt2000_to_datetime(int(d['epoch'][0]))
stopday = spacepy.pycdf.lib.tt2000_to_datetime(int(d['epoch'][-1]))
strtmnth = strtday.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
dt = datetime.timedelta(weeks=1)
lbls = [strtmnth + i * dt for i in range((stopday - strtmnth) // dt)]
plt.xticks(d2t(lbls), d2s(lbls), rotation=90)

plt.colorbar()
plt.yscale('log')
plt.tight_layout()
plt.xlim(
    int(d['epoch'][0]) - 7 * 24 * 60 * 60 * 1e9,
    int(d['epoch'][0]) + 37 * 24 * 60 * 60 * 1e9,
)

c = spacepy.plot.utils.EventClicker(
    n_phases=2,
    interval=30 * 24 * 60 * 60 * 1e9,  # 1 month in nanos
    ymax=200,
    ymin=20,
)
c.analyze()
