#!/usr/bin/env python
# coding: utf-8

import numpy as np

def acf(x):
    result = np.correlate(x, x, mode='full')
    for i in range(len(result)):
        result[i] = result[i] / (len(result)-i) # normalizzazione in base al numero di punti sommati
    return  result[result.size // 2:] / result[result.size // 2]


def acf_gen_mean_x(db):
    acfs = acf(db[0,:,0])
    for i in range(1,db.shape[0]):
        acfs += acf(db[i,:,0])
    
    return acfs / (db.shape[0] + 1)

def acf_gen_x(db):
    acfs = np.ndarray(shape=db.shape)
    leng = db.shape[0]
    for i in range(leng):
        acfs[i,:,0] = acf(db[i,:,0])
        print(i,'/',leng,end='\r')
    return acfs


def exit_time(paths, soglia):
    for jj, path in enumerate(paths):
        etx = []
        ety = []
        etz = []
        r = float(path[-8:-4])
        print(f'computing exit time for r={r}, {0:5}', end='\r')
        db = np.load(path)
        sig_len = len(db[0,0,:])
        n_traj = len(db[:,0,0])
        for traj, ii in zip(db, range(n_traj)):
            if ii == 49999: print(f'computing exit time for r={r}, {ii:5}', end=' ')
            else: print(f'computing exit time for r={r}, {ii:5}', end='\r')
            a = traj[0,:]
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
        np.save(f'/scratch/scarpolini/databases/exit_time_{soglia:.2f}_lorenz_{r:.1f}', [etx])
        print('Saved!')
        
        
def gen_exit_time(paths, soglia):
    for jj, path in enumerate(paths):
        etx = []
        # r = float(path[-8:-4])
        r = 54
        print(f'computing exit time for r={r}, {0:5}', end='\r')
        db = np.load(path)
        sig_len = len(db[0,:,0])
        n_traj = len(db[:,0,0])
        for traj, ii in zip(db, range(n_traj)):
            if ii == 49999: print(f'computing exit time for r={r}, {ii:5}', end=' ')
            else: print(f'computing exit time for r={r}, {ii:5}', end='\r')
            a = traj[:,0]
            a -= np.mean(a)
            acfx = acf(a)
            for t in range(len(acfx)):
                if acfx[t] < soglia :
                    etx.append((1.)/(acfx[t]-acfx[t-1])*(0.5-acfx[t-1]) + t - 1.) ### interpolazione lineare
                    break
        np.save(f'/scratch/scarpolini/databases/gen_exit_time_{soglia:.2f}_lorenz_{r:.1f}', [etx])
        print('Saved!')

def load_acf(r):
    path = f'/scratch/scarpolini/databases/acfe_lorenz_{r:.1f}.npy'
    acf = np.load(path)
    return acf

def load_random_traj(r):
    n = round(np.random.uniform(50000))
    path = f'/scratch/scarpolini/databases/db_lorenz_{r:.1f}.npy'
    trajx = np.load(path)[n,0,:]
    return trajx
