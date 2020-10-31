#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
from params import *
import stats

# COMMAND LINE PARSING

if "-h" in sys.argv:
    print(("usage: plot_sf.py <run> <number> [--gen_renorm] [--read_gen]"
           "[--comp_real] [--scratch] [--no_gen]"))
    exit()

run = int(sys.argv.pop(1))
number = int(sys.argv.pop(1))
gen_renorm = False
comp_real = False
scratch = False
plot_gen = True
read_gen = False

while len(sys.argv) > 1:
    if sys.argv[1] == '--gen_renorm':
        gen_renorm = True
        sys.argv.pop(1)
    elif sys.argv[1] == '--comp_real':
        comp_real = True
        sys.argv.pop(1)
    elif sys.argv[1] == '--no_gen':
        plot_gen = False
        sys.argv.pop(1)
    elif sys.argv[1] == '--scratch':
        scratch = True
        sys.argv.pop(1)
    elif sys.argv[1] == '--read_gen':
        read_gen = True
        sys.argv.pop(1)
    else:
        raise NameError('Parsing error, exiting.')
        exit()

if scratch:
    read_path = (f'/scratch/scarpolini/'+DB_NAME+'/'+WGAN_TYPE+
                 f'/runs/{run}/gen_trajs_{number}.npy')
else:
    read_path = (f'/storage/scarpolini/databases/'+DB_NAME+'/'+
            WGAN_TYPE+f'/runs/{run}/gen_trajs_{number}.npy')

read_path_r = '../data/real/sf.dat'
if not comp_real:
    sfr = np.loadtxt(read_path_r)
else:
    db = np.load(REAL_DB_PATH)[:,:,COMPONENTS]
    if db.size == 3:
        assert db.shape[2] == 1 # not ready to handle multi-dim sf
    sfr = stats.compute_sf(db)
    np.savetxt(read_path_r, sfr)


if plot_gen and (not read_gen):
    write_path = f'runs/{run}/{number}_sf_we'
    nn = 0
    save_path = '../data/'+WGAN_TYPE+f'/runs/{run}/sf_{run}_{number}_we_{str(nn).zfill(4)}.dat'
    while os.path.exists(save_path):
        nn = nn + 1
        save_path = '../data/'+WGAN_TYPE+f'/runs/{run}/sf_{run}_{number}_we_{str(nn).zfill(4)}.dat'
elif plot_gen and read_gen:
    write_path = f'runs/{run}/{number}_sf_we_mean'
else:
    write_path = f'../data/real/real_sfs.dat'

if plot_gen and (not read_gen):
    gen = np.load(read_path)
    gen = gen[:,100:1900] # WE WITHOUT EXTREMES
    print('shape: ',gen.shape)
    M = gen.max()
    m = gen.min()
    print('Gen min, max: ',m,M)

if gen_renorm:
    semidisp = (DB_MAX - DB_MIN)/2.
    media = (DB_MAX + DB_MIN)/2.
    gen = gen*semidisp + media
    print('veri',m,M)
    M = gen.max()
    m = gen.min()
    print('After renorm Gen min, max',m,M)


# COMPUTE GEN SF
if plot_gen :
    if (not read_gen) :
        sfg = stats.compute_sf(gen)
        np.savetxt(save_path, sfg)
    else:
        nn = 0
        sfg_path = '../data/'+WGAN_TYPE+f'/runs/{run}/sf_{run}_{number}_we_{str(nn).zfill(4)}.dat'
        sfg = np.loadtxt(sfg_path)
        nn = nn + 1
        sfg_path = '../data/'+WGAN_TYPE+f'/runs/{run}/sf_{run}_{number}_we_{str(nn).zfill(4)}.dat'
        # SUM
        while os.path.exists(sfg_path):
            temp = np.loadtxt(sfg_path)
            assert all(sfg[:,0] == temp[:,0])
            sfg[:,1:] = sfg[:,1:] + temp[:,1:]
            nn = nn + 1
            sfg_path = '../data/'+WGAN_TYPE+f'/runs/{run}/sf_{run}_{number}_we_{str(nn).zfill(4)}.dat'
        sfg[:,1:] = sfg[:,1:] / nn

    # LOG DERIVATIVES
    dlg = np.zeros(shape=sfg.shape)
    dlg[:,0] = sfg[:,0]
    dlg[:,1:] =  np.gradient( np.log(sfg[:,1:]), np.log(sfg[:,0]), axis=0  )
dlr = np.zeros(shape=sfr.shape)
dlr[:,0] = sfr[:,0]
dlr[:,1:] =  np.gradient( np.log(sfr[:,1:]), np.log(sfr[:,0]), axis=0 )


op_gen = {'marker':'.','lw':0.4,'ms':25,'markeredgewidth':1 ,
          'markeredgecolor':"black"}
op_real = {'marker':'^','lw':0.4,'ms':14}
op_leg = {'markerscale':1, 'ncol':2}

plt.rcParams['font.size'] = 24
#plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 17
plt.rcParams['figure.dpi'] = 60
plt.rcParams['figure.figsize'] = (17, 12)
plt.rcParams['legend.markerscale'] = 2

fig, ax = plt.subplots(2,2)
ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].set_xlabel("$\\tau$")
ax[0,0].set_ylabel('$S_n(\\tau)$')
ax[0,1].set_xscale('log')
ax[0,1].set_yticks([3,6,9,12,15,18])
ax[0,1].set_xlabel("$\\tau$")
ax[0,1].set_ylabel("$F_n(\\tau)$")
ax[0,1].set_yscale('log')
ax[0,1].grid(which='minor', alpha=0.2)
ax[0,1].grid(which='major', alpha=0.9)
ax[1,0].set_xscale('log')
ax[1,0].set_ylim([-0.3,7])
ax[1,0].set_xlabel("$\\tau$")
ax[1,0].set_ylabel("$\\xi_n(\\tau)$")
ax[1,1].set_ylim([0.,4.5])
ax[1,1].set_xscale('log')
ax[1,1].set_xlabel("$\\tau$")
ax[1,1].set_ylabel("$\\xi_n(\\tau)/ \\xi_2(\\tau)$")

#sf
for ii in range(1,4):
    ax[0,0].plot(sfr[:,0],sfr[:,ii],label="DNS n="+str((ii)*2),
                 color='C'+str(ii-1),**op_real)
if plot_gen:
    for ii in range(1,4):
        ax[0,0].plot(sfg[:,0],sfg[:,ii],label="GAN n="+str((ii)*2),
                     color='C'+str(ii-1),**op_gen)

#flatnesses
ax[0,1].plot(sfr[:,0],sfr[:,2]/sfr[:,1]**2., label="DNS $F_4$",
             color='C0',**op_real)
ax[0,1].plot(sfr[:,0],sfr[:,3]/sfr[:,1]**3., label="DNS $F_6$",
             color='C1',**op_real)
if plot_gen:
    ax[0,1].plot(sfg[:,0],sfg[:,2]/sfg[:,1]**2., label="GAN $F_4$",
                 color='C0',**op_gen)
    ax[0,1].plot(sfg[:,0],sfg[:,3]/sfg[:,1]**3., label="GAN $F_6$",
                 color='C1',**op_gen)

# locslopes
for ii in range(1,4):
    ax[1,0].plot(*(dlr[:,[0,ii]].T), label="DNS n="+str((ii)*2),
                 color='C'+str(ii-1),**op_real)
if plot_gen:
    for ii in range(1,4):
        ax[1,0].plot(*(dlg[:,[0,ii]].T), label="GAN n= "+str((ii)*2),
                     color='C'+str(ii-1),**op_gen)

#locslopes ess
ax[1,1].plot(dlr[:,0], (lambda x: [4/2]*len(x))(dlr[:,0]),ls='--',
             color="C0")
ax[1,1].plot(dlr[:,0], (lambda x: [6/2]*len(x))(dlr[:,0]),ls='--',
             color="C1")
for ii in range(2):
    ax[1,1].plot(dlr[:,0],dlr[:,ii+2]/dlr[:,1],label="DNS n="+str((ii+2)*2),
                 color='C'+str(ii),**op_real)
if plot_gen:
    for ii in range(2):
        ax[1,1].plot(dlg[:,0],dlg[:,ii+2]/dlg[:,1],label="GAN n="+str((ii+2)*2),
                     color='C'+str(ii),**op_gen)

ax[0,0].legend(**op_leg)
ax[0,1].legend(**op_leg)
ax[1,0].legend(**op_leg)
ax[1,1].legend(**op_leg)

fig.tight_layout()
fig.savefig(write_path, fmt='png', dpi=60)
