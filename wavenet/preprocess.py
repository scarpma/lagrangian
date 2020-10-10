#!/usr/bin/env python
# coding: utf-8

from db_utils import *

def scale_audio_uint8_to_float64(arr):
    """ Scales an array of 8-bit unsigned integers [0-255] to [-1, 1].
    """
    vmax = np.iinfo(np.uint8).max
    vmin = np.iinfo(np.uint8).min
    arr = arr.astype(np.float64)
    return 2 * ((arr - vmin) / (vmax - vmin)) - 1


def scale_audio_float64_to_uint8(arr):
    """ Scales an array of float values between [-1, 1] to an 8-bit
        unsigned integer value [0, 255].
        Inverse of `scale_audio_uint8_to_float64`.
    """
    vmax = np.iinfo(np.uint8).max
    arr = ((arr + 1) / 2) * vmax
    arr = arr.astype(np.uint8)
    return arr

def to_one_hot(xt):
    """ Converts an integer value between 0 and 255 to its one-hot
        representation.
    """
    return np.eye(256, dtype="uint8")[xt]


def get_audio_sample_batches(db, receptive_field_size,
                             stride_step=32):
    """ Provides the audio data in batches for training and validation.
        Note: This function used to be a generator, but when experimenting
        with little data, a function makes more sense.
        Args:
            path_to_audio_train (str): Path to the directory containin the
                audio files for training.
            receptive_field_size (int): The size of the sliding window that
                passes over the data and collects training samples.
            stride_step (int, default:32): The step by which the window slides.
    """
    X = []
    y = []
    for traj in db:
        offset = 0
        # while offset + receptive_field_size - 1 < len(traj):
        while offset + receptive_field_size < len(traj):
            X.append(traj[offset:offset + receptive_field_size])
            y_cur = traj[receptive_field_size,COMPONENT]
            y_cur = scale_audio_float64_to_uint8(y_cur)
            y.append(to_one_hot(y_cur))
            offset += stride_step
    return np.array(X), np.array(y)


def prediction_to_waveform_value(probability_distribution, random=False):
    """ Accepts the output of the WaveNet as input (a probability vector of
        size 256) and outputs a 16-bit integer that that corresponds to the
        position selected in the expanded space.
        Args:
            probability_distribution (np.array(256)): An 1-dimensional vector
                that represents the probability distribution over the next
                value of the generated waveform.
            random (bool, default:False): If true, a random value between 0
                and 256 will be used to reconstruct the signal, drawn according
                to the provided distribution. Otherwise, the most probable
                value will be selected.
    """
    if random:
        choice = np.random.choice(range(256), p=probability_distribution)
    else:
        choice = np.argmax(probability_distribution)
    # Project the predicted [0, 255] integer value back to [-2^15, 2^15 - 1].
    y_cur = scale_audio_uint8_to_float64(choice)
    return y_cur


def generate_audio_from_model_output(
        path_to_model, path_to_output_wav, input_audio_size, generated_frames,
        sample_rate):
    wavenet = load_model(path_to_model)
    # We initialize with zeros, can also use a proper seed.
    generated_audio = np.zeros(input_audio_size, dtype=np.int16)
    cur_frame = 0
    while cur_frame < generated_frames:
        # Frame is always shifting by `cur_frame`, so that we can always
        # get the last `input_audio_size` values.
        probability_distribution = wavenet.predict(
            generated_audio[cur_frame:].reshape(
                1, input_audio_size, 1)).flatten()
        cur_sample = prediction_to_waveform_value(probability_distribution)
        generated_audio = np.append(generated_audio, cur_sample)
        cur_frame += 1
    return generated_audio


if __name__ == '__main__':
    import sys

    if len(sys.argv)==2 and sys.argv[1]=='-h':
        print('usage: <>')
        exit()
    else:
        input_size = int(sys.argv[1])

