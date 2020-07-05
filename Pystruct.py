#!/usr/bin/env python
# coding: utf-8


import numpy as np

print("Database import")
path_v = '../databases/velocities.npy'
db = np.load(path_v)

taus = np.round(np.logspace(np.log2(2),np.log2(1000),23,base=2)).astype('int')
print("taus= ",taus)
ps = [2,4,6]
struct = np.zeros(shape=(len(taus),len(ps)))
npart = 30000
every=2

f = open("data/struct_function", 'w')

for kk, tau in enumerate(taus):
    print("tau=%6d"%(tau))
    for ii in range(npart):
        if ii%5000==0: print("num. part=",ii)
        for jj in np.arange(0,2000-tau,every):
            diff = db[ii,jj+tau,0] - db[ii,jj,0]
            for pp, p in enumerate(ps):
                struct[kk,pp] = struct[kk,pp] + (diff)**p
    struct[kk] = struct[kk] / ( npart * (2000-tau)/every )
    # np.savetxt("data/struct_function", struct, fmt='%.18e', delimiter=' ')
    f.write("%.6d %20.10g %20.10g %20.10g\n"%(taus[kk],struct[kk,0],struct[kk,1],struct[kk,2]))
f.close()

print("Done!")
