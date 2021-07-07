import sys

import data_jat as data
d2t, t2d, d2s = data.d2t, data.t2d, data.d2s

d = data.Data(globstr=sys.argv[1] if len(sys.argv) > 1 else '')
d.read_events()

# '{i:<02d},\t{strt:>20s},\t{strt_tt2000:>32d},\t{stop:>20s},\t{strt_tt2000:>32d}'
print('{i:2s},\t{strt:>20s},\t{strt_tt2000:>20s},\t{stop:>20s},\t{stop_tt2000:>20s}'.format(
    **dict(i='EN', strt='Starttime', strt_tt2000='Starttime (TT2000)',
           stop='Stoptime', stop_tt2000='Stoptime (TT2000)')
))

zipped = zip(
    [i for i in range(len(d.eventtimes))],
    [s for s in d2s(d.eventtimes[:, 0])],
    [s for s in d2t(d.eventtimes[:, 0])],
    [s for s in d2s(d.eventtimes[:, 1])],
    [s for s in d2t(d.eventtimes[:, 1])]
)

for vals in zipped:
    print('{:02d},\t{:>20s},\t{:>20d},\t{:>20s},\t{:>20d}'.format(*vals))
