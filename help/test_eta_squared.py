import pickle, common, numpy as np
with open('../data/magfield.pickle3', 'rb') as fp:
    arr = pickle.load(fp)
mag = arr['mag']
epoch = arr['epoch']
