#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import subprocess
import sys
import glob
import argparse
import shlex

import tensorflow
import tensorflow.keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (Dense, Conv1D, Conv2D, Conv2DTranspose,
                                     Flatten, Dropout, ReLU, Input, Reshape,
                                     BatchNormalization, Activation)
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def load_data(val_split):
    path = '/scratch/scarpolini/databases/velocities_2048.npy'
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
    if val_split < 1.:
        return db[:end,:,:], db[end:,:,:]
    elif val_split == 1.0 :
        return db, None