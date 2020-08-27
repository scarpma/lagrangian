#!/usr/bin/env python
# coding: utf-8

from db_utils import *

def build_critic(fs, fm, init_sigma, init_mean, alpha):
    """
    fs = 20 dimensione filtro
    fm = 4 numero di filtri
    init_sigma = 0.2 varianza distribuzione normale per l'inizializzazione
    deipesi del  modello
    init_mean = 0.0 media distribuzione normale per l'inizializzazione
    dei pesi del  modello
    """
    reg = l2(l=0.001)

    d = Sequential()
    d.add(Conv1D(fm//16, fs, strides=2, padding='same', kernel_regularizer=reg,
          bias_regularizer=reg, kernel_initializer=RandomNormal(init_mean,
          init_sigma), input_shape=(2048, 1))) #VAR
    # d.add(ELU())
    d.add(ReLU(negative_slope=alpha))
    d.add(Conv1D(fm//8, fs, strides=2, padding='same', kernel_regularizer=reg,
          bias_regularizer=reg,
          kernel_initializer=RandomNormal(init_mean, init_sigma)))
    #discriminator.add(ELU())
    d.add(ReLU(negative_slope=alpha))
    d.add(Conv1D(fm//4, fs, strides=2, padding='same', kernel_regularizer=reg,
          bias_regularizer=reg,
          kernel_initializer=RandomNormal(init_mean, init_sigma)))
    # d.add(ELU())
    d.add(ReLU(negative_slope=alpha))
    d.add(Conv1D(fm//2, fs, strides=2, padding='same', kernel_regularizer=reg,
          bias_regularizer=reg,
          kernel_initializer=RandomNormal(init_mean, init_sigma)))
    # d.add(ELU())
    d.add(ReLU(negative_slope=alpha))
    d.add(Conv1D(fm, fs, strides=2, padding='same', kernel_regularizer=reg,
          bias_regularizer=reg,
          kernel_initializer=RandomNormal(init_mean, init_sigma)))
    # d.add(ELU())
    d.add(ReLU(negative_slope=alpha))
    d.add(Flatten())
    d.add(Dense(1, kernel_regularizer=reg, bias_regularizer=reg))
    d.summary()

    return d




if __name__ == '__main__':
    fs = 100
    fm = 128
    init_sigma = 0.02
    init_mean = 0.01
    alpha = 0.3
    noise_dim = 100
    critic = build_critic(fs, fm, init_sigma, init_mean, alpha)
