#!/usr/bin/env python
# coding: utf-8

def save_pdf(bin_edges, hist, name):
    import pickle
    try:
        with open(name+'.pickle', 'wb') as f:
            pickle.dump((hist,bin_edges), f)
        return 1
    except:
        return 0
#print(save_pdf(bin_edges, hist, 'data/pdf_ax'))

def load_pdf(name):
    import pickle
    with open(name+'.pickle', 'rb') as f:
        (hist, bin_edges) = pickle.load(f)
    for i in range(len(hist)):
        bin_edges[i] = (bin_edges[i]+bin_edges[i+1])/2.
        
    return hist, bin_edges[:-1]

#hist1, bin_edges1 = load_pdf('data/pdf_ax')


def create_log_bins(xmin,xmax,nbin,eps):
    import numpy as np
    if xmin*xmax<0.:
        n1 = int(nbin*abs(xmin)/(xmax-xmin))
        n2 = int(nbin*xmax/(xmax-xmin))
        bins1 = np.logspace(np.log10(eps),np.log10(abs(xmin)), n1)
        bins2 = np.logspace(np.log10(eps),np.log10(xmax), n2)
        bins = np.concatenate((-bins1[::-1],bins2))
        return bins
    elif xmax>0. and xmin>0.:
        bins = np.logspace(np.log10(xmin),np.log10(xmax), nbin)
        return bins
    else:
        print("problem")
        return 0