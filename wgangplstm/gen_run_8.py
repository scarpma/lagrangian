#!/usr/bin/env python
# coding: utf-8

from db_utils import *

def build_generator(cells, fs, init_sigma, init_mean, noise_dim):

    reg = l2(l=0.001)
    generator = Sequential()

    generator.add(Bidirectional(LSTM(cells, return_sequences=True, kernel_initializer=RandomNormal(init_mean, init_sigma)), input_shape=(noise_dim,1), merge_mode='sum'))
    #generator.add(LSTM(cells, return_sequences=True, kernel_initializer=RandomNormal(init_mean, init_sigma), input_shape=(noise_dim,1)))
    generator.add(Dense(cells, kernel_regularizer=reg, bias_regularizer=reg, kernel_initializer=RandomNormal(init_mean, init_sigma), input_dim=noise_dim))
    generator.add(Conv1D(cells//2, fs, padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma), input_shape=(2000, cells), data_format='channels_last'))
    generator.add(Conv1D(cells//4, fs, padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)))
    generator.add(Conv1D(cells//8, fs, padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)))
    generator.add(Conv1D(1, fs, padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)))

    generator.add(Activation("tanh"))
    generator.summary()

    return generator



if __name__ == '__main__':
    cells = 16
    fs=100
    init_sigma = 0.02
    init_mean = 0.01
    noise_dim = 2000
    gen = build_generator(cells,fs,init_sigma, init_mean, noise_dim)
