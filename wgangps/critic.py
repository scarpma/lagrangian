#!/usr/bin/env python
# coding: utf-8

from db_utils import *

def build_critic(fs, fm, init_sigma, init_mean, alpha):
    """ 
    fs = 20 dimensione filtro
    fm = 4 numero di filtri
    init_sigma = 0.2 varianza distribuzione normale per l'inizializzazione dei pesi del  modello
    init_mean = 0.0 media distribuzione normale per l'inizializzazione dei pesi del  modello
    """
    reg = l2(l=0.001)
    
    discriminator = Sequential()
    #2000x1
    discriminator.add(Conv1D(fm//16, fs, strides=2, padding='same', kernel_regularizer=reg, bias_regularizer=reg, kernel_initializer=RandomNormal(init_mean, init_sigma), input_shape=(2000, 1)))
    #discriminator.add(ELU())
    discriminator.add(ReLU(negative_slope=alpha))
    #
    discriminator.add(Conv1D(fm//8, fs, strides=2, padding='same', kernel_regularizer=reg, bias_regularizer=reg, kernel_initializer=RandomNormal(init_mean, init_sigma)))    
    #discriminator.add(ELU())
    discriminator.add(ReLU(negative_slope=alpha))
    #
    discriminator.add(Conv1D(fm//4, fs, strides=2, padding='same', kernel_regularizer=reg, bias_regularizer=reg, kernel_initializer=RandomNormal(init_mean, init_sigma))) 
    #discriminator.add(ELU())
    discriminator.add(ReLU(negative_slope=alpha))
    #
    discriminator.add(Conv1D(fm//2, fs, strides=2, padding='same', kernel_regularizer=reg, bias_regularizer=reg, kernel_initializer=RandomNormal(init_mean, init_sigma))) 
    #discriminator.add(ELU())
    discriminator.add(ReLU(negative_slope=alpha))
    #
    discriminator.add(Conv1D(fm, fs, strides=5, padding='same', kernel_regularizer=reg, bias_regularizer=reg, kernel_initializer=RandomNormal(init_mean, init_sigma))) 
    #discriminator.add(ELU())
    discriminator.add(ReLU(negative_slope=alpha))
    #
    discriminator.add(Flatten())
    #
    discriminator.add(Dense(1, kernel_regularizer=reg, bias_regularizer=reg))
    #1x1
    #discriminator.summary()
    discriminator.summary()
    
    return discriminator




if __name__ == '__main__':
    fs = 20
    fm = 128
    init_sigma = 0.02
    init_mean = 0.01
    alpha = 0.3
    noise_dim = 100
    critic = build_critic(fs, fm, init_sigma, init_mean, alpha)
