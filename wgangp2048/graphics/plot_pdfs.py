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
           " [--comp_real] [--scratch] [--no_gen] [--only_hist]"
           " [--read_gen]"))
    exit()

run = int(sys.argv.pop(1))
number = int(sys.argv.pop(1))
gen_renorm = False
comp_real = False
scratch = False
plot_gen = True
only_hist = False
read_gen = False

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
    elif sys.argv[1] == '--only_hist':
        only_hist = True
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

read_path_r = ['../data/real/pdf0.dat', '../data/real/pdf1.dat',
                    '../data/real/pdf2.dat']


if only_hist:
    if not os.path.exists('../data/'+WGAN_TYPE+f'/runs/{run}'):
        os.mkdir('../data/'+WGAN_TYPE+f'/runs/{run}')

    nn = 0
    save_path = '../data/'+WGAN_TYPE+f'/runs/{run}/pdf_{run}_{number}_we_{str(nn).zfill(4)}.dat'
    while os.path.exists(save_path):
        nn = nn + 1
        save_path = '../data/'+WGAN_TYPE+f'/runs/{run}/pdf_{run}_{number}_we_{str(nn).zfill(4)}.dat'
elif plot_gen and read_gen:
    # write_path = f'runs/{run}/{number}_pdfs'
    write_path = f'runs/{run}/{number}_pdfs_we_mean' # WE WITHOUT EXTREMES
elif plot_gen:
    # write_path = f'runs/{run}/{number}_pdfs'
    write_path = f'runs/{run}/{number}_pdfs_we' # WE WITHOUT EXTREMES
else:
    write_path = '../data/real/pdfs_real'


if plot_gen and not read_gen:
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

if not only_hist:
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

# READ GEN HISTOGRAMS, SUM, NORMALIZE AND STANDARDIZE
if read_gen:
    nn = 0
    read_path_g = '../data/'+WGAN_TYPE+f'/runs/{run}/pdf_{run}_{number}_we_{str(nn).zfill(4)}.dat'
    vg = np.loadtxt(read_path_g)[0:2,:]
    ag = np.loadtxt(read_path_g)[2:4,:]
    rg = np.loadtxt(read_path_g)[4:6,:]
    nn = nn + 1
    read_path_g = '../data/'+WGAN_TYPE+f'/runs/{run}/pdf_{run}_{number}_we_{str(nn).zfill(4)}.dat'
    # SUM
    while os.path.exists(read_path_g):
        temp = np.loadtxt(read_path_g)
        assert all(vg[0,:] == temp[0,:])
        vg[1,:] = vg[1,:] + temp[1,:]
        assert all(ag[0,:] == temp[2,:])
        ag[1,:] = ag[1,:] + temp[3,:]
        assert all(rg[0,:] == temp[4,:])
        rg[1,:] = rg[1,:] + temp[5,:]
        nn = nn + 1
        read_path_g = '../data/'+WGAN_TYPE+f'/runs/{run}/pdf_{run}_{number}_we_{str(nn).zfill(4)}.dat'

    # mi creo un array in cui ci sono i dx per calcolare gli integrali
    # v_for_norm = [vg[0,1]-vg[0,0]]
    # a_for_norm = [ag[0,1]-ag[0,0]]
    # r_for_norm = [rg[0,1]-rg[0,0]]
    # for jj in range(1,vg.shape[1]-1):
    #     v_for_norm.append((vg[0,jj+1]-vg[0,jj-1])/2.)
    #     a_for_norm.append((ag[0,jj+1]-ag[0,jj-1])/2.)
    #     r_for_norm.append((rg[0,jj+1]-rg[0,jj-1])/2.)
    # v_for_norm.append(vg[0,-1]-vg[0,-2])
    # a_for_norm.append(ag[0,-1]-ag[0,-2])
    # r_for_norm.append(rg[0,-1]-rg[0,-2])
    v_bin = stats.create_log_bins(-12,12,600,1.e-2) #VAR
    a_bin = stats.create_log_bins(-6,6,600,5.e-4) #VAR
    r_bin = stats.create_log_bins(-6,6,600,1.e-4) #VAR
    v_for_norm = []
    a_for_norm = []
    r_for_norm = []
    for jj in range(v_bin.shape[0]-1):
        v_for_norm.append(v_bin[jj+1]-v_bin[jj])
        a_for_norm.append(a_bin[jj+1]-a_bin[jj])
        r_for_norm.append(r_bin[jj+1]-r_bin[jj])
    v_for_norm = np.array(v_for_norm)
    a_for_norm = np.array(a_for_norm)
    r_for_norm = np.array(r_for_norm)

    # FROM HISTOGRAM TO DENSITY
    vg[1,:] = vg[1,:] / v_for_norm
    ag[1,:] = ag[1,:] / a_for_norm
    rg[1,:] = rg[1,:] / r_for_norm

    v_norm = 0
    a_norm = 0
    r_norm = 0
    for jj in range(vg.shape[1]):
        v_norm += vg[1,jj] * v_for_norm[jj]
        a_norm += ag[1,jj] * a_for_norm[jj]
        r_norm += rg[1,jj] * r_for_norm[jj]

    # NORMALIZE
    vg[1,:] = vg[1,:] / v_norm
    ag[1,:] = ag[1,:] / a_norm
    rg[1,:] = rg[1,:] / r_norm

    # STANDARDIZE
    mean_v = 0.
    std_v = 0.
    mean_a = 0.
    std_a = 0.
    mean_r = 0.
    std_r = 0.
    for jj in range(vg.shape[1]):
        mean_v += v_for_norm[jj]*vg[1,jj]*vg[0,jj]
        mean_a += a_for_norm[jj]*ag[1,jj]*ag[0,jj]
        mean_r += r_for_norm[jj]*rg[1,jj]*rg[0,jj]
    for jj in range(vg.shape[1]):
        std_v += v_for_norm[jj]*vg[1,jj]*(vg[0,jj]-mean_v)**2.
        std_a += a_for_norm[jj]*ag[1,jj]*(ag[0,jj]-mean_a)**2.
        std_r += r_for_norm[jj]*rg[1,jj]*(rg[0,jj]-mean_r)**2.

    std_v = np.sqrt(std_v)
    std_a = np.sqrt(std_a)
    std_r = np.sqrt(std_r)

    vg[0,:] = (vg[0,:] - mean_v) / std_v
    vg[1,:] = vg[1,:] * std_v
    ag[0,:] = (ag[0,:] - mean_a) / std_a
    ag[1,:] = ag[1,:] * std_a
    rg[0,:] = (rg[0,:] - mean_r) / std_r
    rg[1,:] = rg[1,:] * std_r


# VELOCITY
#real
if not comp_real and not only_hist:
    vr, vm, vstd = stats.make_hist(read_path_r[0], std=True, out=True)
elif not only_hist:
    db = np.load(REAL_DB_PATH)[:,:,COMPONENTS]
    # bins = stats.create_log_bins(db.min(),db.max(),600,1e-2) #VAR
    # BINS LINEARI
    # bins = np.linspace(-12,12,2000)
    # BINS LOGARITMICI FISSI
    bins = stats.create_log_bins(-12,12,600,1.e-2) #VAR

    vr = stats.make_hist(db, bins=bins)
    np.savetxt(read_path_r[0], vr)
    vr, vm, vstd = stats.make_hist(read_path_r[0], std=True, out=True)
if not only_hist:
    ax[0,0].plot(*(vr), **op_real)
    ax[0,1].plot(*(vr), **op_real)

#gen
if plot_gen:
    # bins = stats.create_log_bins(gen.min(),gen.max(),600,1e-2) #VAR
    # BINS LINEARI
    # bins = np.linspace(-12,12,2000)
    # BINS LOGARITMICI FISSI
    bins = stats.create_log_bins(-12,12,600,1.e-2) #VAR
    if only_hist:
        vg = stats.make_hist(gen, bins=bins, hist=True)
    elif not read_gen:
        vg = stats.make_hist(gen, bins=bins, std=True)
        # VECCHIA STANDARDIZZAZIONE RISPETTO STD DEI DATI VERI
        # vg[1,:] = vg[1,:] * vstd
        # vg[0,:] = ( vg[0,:] - vm ) / vstd
        ax[0,0].plot(*(vg), **op_gen)
        ax[0,1].plot(*(vg), **op_gen)
    elif read_gen:
        ax[0,0].plot(*(vg), **op_gen)
        ax[0,1].plot(*(vg), **op_gen)

print("Velocity done,",end=" ")

# ACCELERATION
if plot_gen and not read_gen: gm = np.gradient(gen,axis=1)
#real
if not comp_real and not only_hist:
    ar, am, astd = stats.make_hist(read_path_r[1], std=True, out=True)
elif not only_hist:
    dba = np.gradient(db, axis=1)
    #bins = stats.create_log_bins(dba.min(),dba.max(),600,5e-4) #VAR
    # BINS LINEARI
    # bins = np.linspace(-6,6,2000)
    # BINS LOGARITMICI FISSI
    bins = stats.create_log_bins(-6,6,600,5.e-4) #VAR

    ar = stats.make_hist(dba, bins=bins)
    np.savetxt(read_path_r[1], ar)
    ar, am, astd = stats.make_hist(read_path_r[1], std=True, out=True)
if not only_hist:
    ax[1,0].plot(*(ar), **op_real)
    ax[1,1].plot(*(ar), **op_real)

#gen
if plot_gen:
    # bins = stats.create_log_bins(gm.min(),gm.max(),600,5.e-4) #VAR
    # BINS LINEARI
    # bins = np.linspace(-6,6,2000)
    # BINS LOGARITMICI FISSI
    bins = stats.create_log_bins(-6,6,600,5.e-4) #VAR

    if only_hist:
        ag = stats.make_hist(gm, bins=bins, hist=True)
    elif not read_gen:
        ag = stats.make_hist(gm, bins=bins, std=True)
        # VECCHIA STANDARDIZZAZIONE RISPETTO STD DEI DATI VERI
        # ag[1,:] = ag[1,:] * astd
        # ag[0,:] = ( ag[0,:] - am ) / astd
        ax[1,0].plot(*(ag), **op_gen)
        ax[1,1].plot(*(ag), **op_gen)
    elif read_gen:
        ax[1,0].plot(*(ag), **op_gen)
        ax[1,1].plot(*(ag), **op_gen)
print("Acceleration done,",end=" ")

# SECOND DERIVATIVE
if plot_gen and not read_gen: gg = np.gradient(gm,axis=1)
#real
if not comp_real and not only_hist:
    rr, rm, rstd = stats.make_hist(read_path_r[2], std=True, out=True)
elif not only_hist:
    dbr = np.gradient(dba, axis=1)
    # bins = stats.create_log_bins(dbr.min(),dbr.max(),600,1e-4) #VAR
    # BINS LINEARI
    # bins = np.linspace(-6,6,2000)
    # BINS LOGARITMICI FISSI
    bins = stats.create_log_bins(-6,6,600,1.e-4) #VAR

    rr = stats.make_hist(dbr, bins=bins)
    np.savetxt(read_path_r[2], rr)
    rr, rm, rstd = stats.make_hist(read_path_r[2], std=True, out=True)
if not only_hist:
    ax[2,0].plot(*(rr), **op_real)
    ax[2,1].plot(*(rr), **op_real)

#gen
if plot_gen:
    # bins = stats.create_log_bins(gg.min(),gg.max(),600,1.e-4) #VAR
    # BINS LINEARI
    # bins = np.linspace(-6,6,2000)
    # BINS LOGARITMICI FISSI
    bins = stats.create_log_bins(-6,6,600,1.e-4) #VAR

    if only_hist:
        rg = stats.make_hist(gg, bins=bins, hist=True)
    elif not read_gen:
        rg = stats.make_hist(gg, bins=bins, std=True)
        # VECCHIA STANDARDIZZAZIONE RISPETTO STD DEI DATI VERI
        # rg[1,:] = rg[1,:] * rstd
        # rg[0,:] = ( rg[0,:] - rm) / rstd
        ax[2,0].plot(*(rg), **op_gen)
        ax[2,1].plot(*(rg), **op_gen)
    elif read_gen:
        ax[2,0].plot(*(rg), **op_gen)
        ax[2,1].plot(*(rg), **op_gen)
print("S. derivative done. Saving.")

if not only_hist:
    ax[0,0].legend(**op_leg)
    ax[0,1].legend(**op_leg)
    ax[1,0].legend(**op_leg)
    ax[1,1].legend(**op_leg)
    ax[2,0].legend(**op_leg)
    ax[2,1].legend(**op_leg)

    fig.tight_layout()
    fig.savefig(write_path, fmt='png', dpi=60)
else:
    array_to_save = np.r_[vg,ag,rg]
    np.savetxt(save_path, array_to_save)

