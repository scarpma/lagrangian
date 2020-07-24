#!/usr/bin/env python
# coding: utf-8

run=74
number=2000
npart=None
media=0

import numpy as np

print("Database import")
if run==0:
    #path_v = f'../databases/gaussian_process.npy'
    path_v = f'../databases/velocities.npy'
elif media==0:
    path_v = f'wgangps/runs/{run}/gen_trajs_{number}.npy'
    #path_v = f'wgangp/runs/{run}/gen_trajs_{number}.npy'
else:
    path_v = f'wgangp/runs/{run}/gen_trajs_{number}_media.npy'

print("Loading ... ", path_v)
db = np.load(path_v)
if db.shape[-1]!=1: 
    db = db.reshape((db.shape[0],db.shape[1],1))
    print("Database reshaped")
if npart == None:
    npart = db.shape[0]
    print(f"Taking entire dataset, {npart} samples") 
else: 
    idx = np.random.randint(0,db.shape[0],npart)
    db = db[idx]

taus = np.round(np.logspace(np.log2(2),np.log2(1000),23,base=2)).astype('int')
print("taus= ",taus)

s = np.zeros(shape=(len(taus),4))
s[:,0] = taus

def compute_struct_func(db,npart,taus):
    for ii, tau in enumerate(taus):
        for part in range(0,npart):
            if part%1000==0:print("%4d:  %7d"%(tau,part),end="\r")
            for potn, pot in enumerate([2.,4.,6.]):
                s[ii,potn+1] = s[ii,potn+1] + np.sum((np.convolve(db[part,:,0],np.r_[1,[0]*(tau-1),-1],mode="valid"))**pot)
        print("")
        s[ii,1:] = s[ii,1:] / (npart*(db.shape[1]-tau+1))
    return s


s = compute_struct_func(db,npart,taus)
if run==0: 
    #np.save(f"data/gaussian_struct_function_{npart}_part",s)
    np.save(f"data/struct_function_{npart}_part",s)
elif media == 0: np.save(f"data/struct_function_{npart}_part_gen_s_{run}_{number}",s)
#elif media == 0: np.save(f"data/struct_function_{npart}_part_gen_{run}_{number}",s)
else: np.save(f"data/struct_function_{npart}_part_gen_{run}_{number}_media",s)

print("Done!")
