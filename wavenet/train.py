#!/usr/bin/env python
# coding: utf-8

from db_utils import *
from model import build_wavenet_model
from preprocess import get_audio_sample_batches

def train_wavenet(db_train, db_valid, input_size,
                  num_filters, kernel_size, num_residual_blocks,
                  batch_size, num_epochs):
    """ A wrapper function that, given the path to audio files for training
        and validation, trains the WaveNet and returns a trained Keras model
        with its training history metadata.
        Args:
            path_to_audio_train (str): Path to the directory containin the
                audio files for training.
            path_to_audio_valid (str): Path to the directory containin the
                audio files for validation.
            input_size (int): The size of the input layer of the network,
                and the receptive field during the construction of the data
                samples.
            num_filters (int): Number of filters used for convolution in the
                causal and dilated convolution layers.
            kernel_size (int): Convolution window size for the causal and
                dilated convolution layers.
            num_residual_blocks (int): How many residual blocks to generate
                between input and output. Residual block i will have a dilation
                rate of 2^(i+1), i starting from zero.
            batch_size (int): Batch size for training.
            num_epochs (int): Number of training epochs.
    """
    wavenet = build_wavenet_model(
        input_size, num_filters, kernel_size, num_residual_blocks)
    print("Generating training data...")
    X_train, y_train = get_audio_sample_batches(
        db_train, input_size)
    print("Generating validation data...")
    X_test, y_test = get_audio_sample_batches(
        db_valid, input_size)
    print("Training model...")
    history = wavenet.fit(
        X_train, y_train, batch_size=batch_size, epochs=num_epochs,
        validation_data=(X_test, y_test), verbose=1)
    return wavenet, history




def main():
    # path_to_audio_train = "data/train/"
    # path_to_audio_valid = "data/valid/"
    db_train, db_valid = load_data(0.8)
    input_size = 256
    num_filters = 64
    kernel_size = 2
    num_residual_blocks = 23
    batch_size = 256
    num_epochs = 40

    distinct_filename = "wavenet_in{}_nf{}_k{}_nres{}_bat{}_e{}.h5".format(
        input_size,
        num_filters,
        kernel_size,
        num_residual_blocks,
        batch_size,
        num_epochs)

    wavenet, history = train_wavenet(
        db_train,
        db_valid,
        input_size,
        num_filters,
        kernel_size,
        num_residual_blocks,
        batch_size,
        num_epochs)

    # Persist model with a distinct name to remember the training parameters.
    print("Saving {}...".format(distinct_filename))
    wavenet.save(distinct_filename)


if __name__ == "__main__":
    main()

