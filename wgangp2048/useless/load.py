#!/usr/bin/env python
# coding: utf-8

import sys
import matplotlib.pyplot as plt
import numpy as np


def std_hist(bins, hist, linear=False):
    """

    Standardizes a pdf or an histogram.

    Parameters
    ----------
    bins : array_like
        Bins median point.
    hist : array_like
        Counts for each bin. Can be normalized.

    Returns
    -------
    bins : array_like
        Bins median point, standardized.
    hist : array_like
        Probability density function for each bin, standardized.

    Notes
    -----
    Works only with linearly spaced bins.

    """
    binw = bins[1]-bins[0]
    norm = np.sum(hist)*binw
    hist = hist / norm
    mean = np.sum(hist*bins*binw)
    std = np.sqrt(np.sum(binw*hist*(bins-mean)**2.))
    hist = hist * std
    bins = (bins - mean) / std
    return bins, hist


read_path =  sys.argv[1]
pdf = np.loadtxt(read_path)
plt.plot(*pdf.T)
plt.savefig("prova",dpi=60,fmt="png")
