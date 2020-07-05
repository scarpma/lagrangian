#!/usr/bin/env python
# coding: utf-8

import os
import glob
import argparse

import tensorflow
import tensorflow.keras
import tensorflow.keras.losses
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, Conv2DTranspose, Flatten, Dropout, ReLU, Input, MaxPooling1D, Reshape, BatchNormalization, Activation, Lambda
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K

from tensorflow.keras.models import load_model
from functools import partial
from tensorflow.image import image_gradients
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


#def load_data():
#    
#    def load_and_shuffle_dbs(rv):
#        paths = []
#        for r in rv:
#            paths.append(f"/scratch/scarpolini/databases/velocities.npy")
#        
#        n_traj = 327680
#        channels = 1
#        db = np.ndarray(shape=(n_traj,2000,channels))
#        labels = []
#        for path,r,i in zip(paths,rv,range(nr)):
#            db1 = np.load(path)
#            for j in range(n_traj):
#                db[i*n_traj + j,:,0] = db1[j,0,:]
#                labels.append(r)
#        
#        labels = np.array(labels)
#        
#        def unison_shuffled_copies(a, b):
#            assert len(a) == len(b)
#            p = np.random.permutation(len(a))
#            return a[p], b[p]
#        
#        db, labels = unison_shuffled_copies(db, labels)
#        return db, labels
#    
#
#    db, labels = load_and_shuffle_dbs(rv)
#    
#    
#    
#    
#    
#    validation_split = 0.0
#    
#    sig_len = len(db[0,:,0])
#    print(f"siglen: {sig_len}")
#    channels = 1 #len(db[0,0,:])
#    print(f"channels: {channels}")
#    n_traj = len(db[:,0,0])
#    print((f"n_traj: {n_traj}"))
#    # numero della prima traiettoria usata come validation
#    first_validation = round((1. - validation_split)*n_traj)
#    print(f"first_validation: {first_validation}")
#    db_train = np.ndarray(shape=(first_validation,sig_len,channels))
#    db_test = np.ndarray(shape=(n_traj-first_validation,sig_len,channels))
#    db_train = db[:first_validation,:,0:channels]
#    db_test = db[first_validation:,:,0:channels]
#    del db
#    
#    print(db_train.shape)
#    M = np.max(np.append(db_train, db_test))
#    m = np.min(np.append(db_train, db_test))
#    print(M,m)
#    semidisp = (M-m)/2.
#    media = (M+m)/2.
#    db_train = (db_train - media)/semidisp
#    db_test = (db_test - media)/semidisp
#    M = np.max(db_train)
#    m = np.min(db_train)
#    print(M,m)
#    
#    return db_train, db_test




#db_train, db_test = load_data()



def load_data(val_split):
    path = '/scratch/scarpolini/databases/accelerations.npy'
    db = np.load(path)[:,:,0:1]
    print(db.shape)
    M = db.max()
    m = db.min()
    print(M,m)
    semidisp = (M-m)/2.
    media = (M+m)/2.
    db = (db - media)/semidisp
    M = db.max()
    m = db.min()
    print(M,m)
    end = round(val_split * db.shape[0])
    return db[:end,:,:], db[end:,:,:]
