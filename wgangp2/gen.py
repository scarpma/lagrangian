#!/usr/bin/env python
# coding: utf-8

from db_utils import *

def build_generator(fs, fm, init_sigma, init_mean, alpha, noise_dim):
    """ 
    fs = (20,1) dimensione filtro
    fm = 4 numero di filtri
    init_sigma = 0.2 varianza distribuzione normale per l'inizializzazione dei pesi del  modello
    init_mean = 0.03 media distribuzione normale per l'inizializzazione dei pesi del  modello
    alpha = 0.3 pendenza parte negativa del leaky relu
    """






    #x = K.reshape(x, (-1,1,2000,1))
    #x = image_gradients(x)[1]
    #x = K.reshape(x, (-1,2000,1))
    #x = np.gradient(sig_input, axis=1)
    #generator.add(Reshape((2000, 1)))

    
    #d_model = Model(sig_input, x)
    #d_model.summary()
    #return d_model


    sig_input = Input(shape=noise_dim)
    x = sig_input
    x = Dense(25*fm, kernel_initializer=RandomNormal(init_mean, init_sigma)) (x)
    x = ReLU() (x)
    x = Reshape((25, 1, fm)) (x)

    x = BatchNormalization(momentum=0.8) (x)
    x = Conv2DTranspose(fm//2, fs, strides=(5,1), padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)) (x)
    x = ReLU() (x)

    x = BatchNormalization(momentum=0.8) (x)
    x = Conv2DTranspose(fm//4, fs, strides=(2,1), padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)) (x)
    x = ReLU() (x)
    x = BatchNormalization(momentum=0.8) (x)
    x = Conv2DTranspose(fm//8, fs, strides=(2,1), padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)) (x)
    x = ReLU() (x)

    x = BatchNormalization(momentum=0.8) (x)
    x = Conv2DTranspose(fm//16, fs, strides=(2,1), padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)) (x)
    x = ReLU() (x)

    x = BatchNormalization(momentum=0.8) (x)
    x = Conv2DTranspose(1, fs, strides=(2,1), padding='same', kernel_initializer=RandomNormal(init_mean, init_sigma)) (x)
    x = Activation("tanh") (x)

    # Eseguo derivata del segnale
    x = Lambda(lambda x: image_gradients(x)[0]) (x)
    x = Reshape((2000,1)) (x)
    
    generator = Model(sig_input,x)
    generator.summary()


    return generator



if __name__ == '__main__':
    fs = (20,1)
    fm = 128
    init_sigma = 0.02
    init_mean = 0.01
    alpha = 0.3
    noise_dim = 100
    gen = build_generator(fs, fm, init_sigma, init_mean, alpha, noise_dim)
