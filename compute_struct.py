#!/usr/bin/env python
# coding: utf-8

run=78
number=1750
npart=None
media=0

import numpy as np

print("Database import")
if run==0:
    path_v = f'../databases/velocities.npy'
elif media==0:
    path_v = f'wgangps/runs/{run}/gen_trajs_{number}.npy'
    #path_v = f'wgangp/runs/{run}/gen_trajs_{number}.npy'
else:
    #path_v = f'wgangps/runs/{run}/gen_trajs_{number}_media.npy'
    path_v = f'wgangp/runs/{run}/gen_trajs_{number}_media.npy'

print("Loading ... ", path_v)
db = np.load(path_v)
if db.shape[-1]==3: 
    db = db[:,:,0:1]
    print("Database with more than one component. Only x taken")
elif db.shape[-1]==2000: 
    db = db.reshape((db.shape[0],db.shape[1],1))
    print("Database without last dimension detected. added last dimension on size 1.")
else: print("Database shape ok, continuing...")

if npart == None:
    npart = db.shape[0]
    print(f"Taking entire dataset, {npart} samples") 
else: 
    idx = np.random.randint(0,db.shape[0],npart)
    db = db[idx]
    print(f"Database larger than npart. Taken {npart} samples randomly.")

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
    save_path = f"data/real/struct_function_{npart}_part"
    print("save_path = ",save_path)
elif media == 0: 
    save_path = f"data/wgangps/struct_function_{npart}_part_gen_{run}_{number}"
    print("save_path = ",save_path)
#elif media == 0: 
#    save_path = f"data/wgangp/struct_function_{npart}_part_gen_{run}_{number}"
#    print("save_path = ",save_path)
else: 
    save_path = f"data/wgangps/struct_function_{npart}_part_gen_{run}_{number}_media"
    print("save_path = ",save_path)
#else: 
#    save_path = f"data/wgangp/struct_function_{npart}_part_gen_{run}_{number}_media"
#    print("save_path = ",save_path)

np.save(save_path,s)
print("Done!")
