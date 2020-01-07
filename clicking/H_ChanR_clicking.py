import matplotlib.pyplot as plt
import numpy as np
import spacepy.plot

# Cut off the first (spoofed) datetime:
g = np.load('flux.npy')[1:, :]
t = np.load('datetime.npy', allow_pickle=True)

# Have to process out nans from g:
g = np.where(np.isnan(g), 1e-20, g)

# Need transpose of g, because I guess that's how it always worked...
plt.pcolormesh(t, np.arange(15), g.T)

# This is preliminary, but might do max and half max..?
c = spacepy.plot.utils.EventClicker(n_phases=2)
c.analyze()
