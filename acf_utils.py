#!/usr/bin/env python
# coding: utf-8

import numpy as np

#def acf(x):
#    result = np.correlate(x, x, mode='full')
#    #for i in range(len(result)):
#    #    result[i] = result[i] / (len(result)-i) # normalizzazione in base al numero di punti sommati
#    result = result / np.arange(len(result),1,-1)
#    return  result[result.size // 2:] / result[result.size // 2]
#
#def acf_gen_x(db):
#    acfs = np.ndarray(shape=db.shape)
#    leng = db.shape[0]
#    for i in range(leng):
#        acfs[i,:,0] = acf(db[i,:,0])
#        print(i,'/',leng,end='\r')
#    return acfs

def s_acf(x):
    result = np.correlate(x, x, mode='full')
    result = result / np.arange(result.shape[-1],0,-1)
    result = result[result.size // 2:] / result[result.size // 2]
    return result

def acf_x(db,npart=None):
    if npart==None:
        dbn = db[:,:,0]
    else:
        idx = np.random.randint(0,db.shape[0],npart)
        dbn = db[idx,:,0]
    acfs = np.array([s_acf(traj) for traj in dbn])
    return acfs.mean(axis=0)


def exit_time(paths, soglia):
    for jj, path in enumerate(paths):
        etx = []
        ety = []
        etz = []
        print(f'computing exit time, {0:5}', end='\r')
        db = np.load(path)
        sig_len = len(db[0,:,0])
        n_traj = len(db[:,0,0])
        for traj, ii in zip(db, range(n_traj)):
            if ii == n_traj-1: print(f'computing exit time for {ii:5}', end=' ')
            else: print(f'computing exit time for {ii:5}', end='\r')
            a = traj[:,0]
            #b = traj[1,:]
            #c = traj[2,:]
            a -= np.mean(a)
            #b -= np.mean(b)
            #c -= np.mean(c)
            acfx = acf(a)
            #acfy = acf(b)
            #acfz = acf(c)
            for t in range(len(acfx)):
                if acfx[t] < soglia :
                    etx.append((1.)/(acfx[t]-acfx[t-1])*(0.5-acfx[t-1]) + t - 1.) ### interpolazione lineare
                    break
            #for t, y in enumerate(acfy):
            #    if y < soglia :
            #        ety.append(t)
            #        break
            #for t, y in enumerate(acfz):
            #    if y < soglia :
            #        etz.append(t)
            #        break
        np.save(f'/scratch/scarpolini/databases/exit_time_{soglia:.2f}_lagrangian', [etx])
        print('Saved!')
        

def gen_exit_time(run, number, soglia):
    paths = [f'data/acf_x_gen_{run}_{number}.npy']
    for jj, path in enumerate(paths):
        etx = []
        ety = []
        etz = []
        print(f'computing exit time, {0:5}', end='\r')
        db = np.load(path)
        sig_len = len(db[0,:,0])
        n_traj = len(db[:,0,0])
        for acfx, ii in zip(db, range(n_traj)):
            if ii%1000==0: print(f'computing exit time, {ii:5}', end='\r')
            for t in range(len(acfx)):
                if acfx[t] < soglia :
                    etx.append(float((1.)/(acfx[t]-acfx[t-1])*(0.5-acfx[t-1]) + t - 1.)) ### interpolazione lineare
                    break
            #for t, y in enumerate(acfy):
            #    if y < soglia :
            #        ety.append(t)
            #        break
            #for t, y in enumerate(acfz):
            #    if y < soglia :
            #        etz.append(t)
            #        break
    return [etx]
        
def load_acf(r):
    path = f'/scratch/scarpolini/databases/acfe_lorenz_{r:.1f}.npy'
    acf = np.load(path)
    return acf

def load_random_traj(r):
    n = round(np.random.uniform(50000))
    path = f'/scratch/scarpolini/databases/db_lorenz_{r:.1f}.npy'
    trajx = np.load(path)[n,0,:]
    return trajx
