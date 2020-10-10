#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import os
import numpy as np
import matplotlib.pyplot as plt
from params import *
import stats


# COMMAND LINE PARSING

if "-h" in sys.argv:
    print(("usage: plot_pdfs.py <run> <number> [--gen_renorm]"
           " [--comp_real] [--scratch] [--no_gen]"))
    exit()

run = int(sys.argv.pop(1))
number = int(sys.argv.pop(1))
gen_renorm = False
comp_real = False
scratch = False
plot_gen = True

while len(sys.argv) > 1:
    if sys.argv[1] == '--gen_renorm':
        gen_renorm = True
        sys.argv.pop(1)
    elif sys.argv[1] == '--comp_real':
        comp_real = True
        sys.argv.pop(1)
    elif sys.argv[1] == '--scratch':
        scratch = True
        sys.argv.pop(1)
    elif sys.argv[1] == '--no_gen':
        plot_gen = False
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

read_path_r = ['../data/real/pdf0.dat', '../data/real/pdf1.dat',
                    '../data/real/pdf2.dat']

if plot_gen:
    write_path = f'runs/{run}/{number}_pdfs'
else:
    write_path = '../data/real/pdfs_real'

if plot_gen :
    gen = np.load(read_path)
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


op_gen = {'marker':'.','lw':0.4,'ms':5,'label':'GAN'}
op_real = {'marker':'^','lw':0.4,'ms':7,'label':'DNS'}
op_leg = {'ncol':1}

plt.rcParams['font.size'] = 24
#plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 17
#plt.rcParams['figure.titlesize'] = 22
plt.rcParams['figure.dpi'] = 60
plt.rcParams['figure.figsize'] = (17, 15)
#plt.rcParams['lines.linewidth'] = 0.5
#plt.rcParams['lines.markersize'] = 13
plt.rcParams['legend.markerscale'] = 2
#plt.rcParams['lines.marker'] = '.'

fig, ax = plt.subplots(3,2, )
ax[0,1].set_yscale('log')
ax[1,1].set_yscale('log')
ax[2,1].set_yscale('log')
ax[0,0].set_xlabel('$v_x / \\sigma_{v_x}$')
ax[0,1].set_xlabel('$v_x / \\sigma_{v_x}$')
ax[1,0].set_xlabel('$a_x / \\sigma_{a_x}$')
ax[1,1].set_xlabel('$a_x / \\sigma_{a_x}$')
ax[2,0].set_xlabel('$r_x / \\sigma_{r_x}$')
ax[2,1].set_xlabel('$r_x / \\sigma_{r_x}$')
ax[0,0].set_ylabel('PDF$(v_x)\\cdot\\sigma_{v_x}$')
ax[1,0].set_ylabel('PDF$(a_x)\\cdot\\sigma_{a_x}$')
ax[2,0].set_ylabel('PDF$(r_x)\\cdot\\sigma_{r_x}$')
ax[0,0].set_xlim([-4,4])
ax[1,0].set_xlim([-4,4])
ax[2,0].set_xlim([-4,4])

# VELOCITY

#real
if not comp_real:
    vr, vm, vstd = stats.make_hist(read_path_r[0], std=True, out=True)
else:
    db = np.load(REAL_DB_PATH)[:,:,COMPONENTS]
    bins = stats.create_log_bins(db.min(),db.max(),600,1e-2) #VAR
    vr = stats.make_hist(db, bins=bins)
    np.savetxt(read_path_r[0], vr)
    vr, vm, vstd = stats.make_hist(read_path_r[0], std=True, out=True)
ax[0,0].plot(*(vr), **op_real)
ax[0,1].plot(*(vr), **op_real)

#gen
if plot_gen:
    bins = stats.create_log_bins(gen.min(),gen.max(),600,1e-2) #VAR
    vg = stats.make_hist(gen, bins=bins)
    vg[1,:] = vg[1,:] * vstd
    vg[0,:] = ( vg[0,:] - vm ) / vstd
    ax[0,0].plot(*(vg), **op_gen)
    ax[0,1].plot(*(vg), **op_gen)
print("Velocity done,",end=" ")

# ACCELERATION
if plot_gen: gm = np.gradient(gen,axis=1)

#real
if not comp_real:
    ar, am, astd = stats.make_hist(read_path_r[1], std=True, out=True)
else:
    dba = np.gradient(db, axis=1)
    bins = stats.create_log_bins(dba.min(),dba.max(),600,5e-4) #VAR
    ar = stats.make_hist(dba, bins=bins)
    np.savetxt(read_path_r[1], ar)
    ar, am, astd = stats.make_hist(read_path_r[1], std=True, out=True)
ax[1,0].plot(*(ar), **op_real)
ax[1,1].plot(*(ar), **op_real)

#gen
if plot_gen:
    bins = stats.create_log_bins(gm.min(),gm.max(),600,5.e-4) #VAR
    ag = stats.make_hist(gm, bins=bins)
    ag[1,:] = ag[1,:] * astd
    ag[0,:] = ( ag[0,:] - am ) / astd
    ax[1,0].plot(*(ag), **op_gen)
    ax[1,1].plot(*(ag), **op_gen)
print("Acceleration done,",end=" ")

# SECOND DERIVATIVE
if plot_gen: gg = np.gradient(gm,axis=1)

#real
if not comp_real:
    rr, rm, rstd = stats.make_hist(read_path_r[2], std=True, out=True)
else:
    dbr = np.gradient(dba, axis=1)
    bins = stats.create_log_bins(dbr.min(),dbr.max(),600,1e-4) #VAR
    rr = stats.make_hist(dbr, bins=bins)
    np.savetxt(read_path_r[2], rr)
    rr, rm, rstd = stats.make_hist(read_path_r[2], std=True, out=True)
ax[2,0].plot(*(rr), **op_real)
ax[2,1].plot(*(rr), **op_real)

#gen
if plot_gen:
    bins = stats.create_log_bins(gg.min(),gg.max(),600,1.e-4) #VAR
    rg = stats.make_hist(gg, bins=bins)
    rg[1,:] = rg[1,:] * rstd
    rg[0,:] = ( rg[0,:] - rm) / rstd
    ax[2,0].plot(*(rg), **op_gen)
    ax[2,1].plot(*(rg), **op_gen)
print("S. derivative done. Saving.")

ax[0,0].legend(**op_leg)
ax[0,1].legend(**op_leg)
ax[1,0].legend(**op_leg)
ax[1,1].legend(**op_leg)
ax[2,0].legend(**op_leg)
ax[2,1].legend(**op_leg)

fig.tight_layout()
fig.savefig(write_path, fmt='png', dpi=60)

