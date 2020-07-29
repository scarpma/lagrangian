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
        estremo = np.max([xmax,-xmin])
        bins1 = np.logspace(np.log10(eps),np.log10(estremo), nbin//2)
        bins = np.r_[-bins1[::-1],[0.],bins1]
        return bins
    elif xmax>0. and xmin>0.:
        bins = np.logspace(np.log10(xmin),np.log10(xmax), nbin)
        return bins
    else:
        print("problem")
        return 0
    

def kurtosis(x,p):
    import numpy as np
    if len(x) == len(p):
        dx = []
        for i in range(len(p)-1):
            dx.append(x[i+1] - x[i])
        dx.append(dx[-1])
        dx = np.array(dx)
    elif len(x) == len(p)+1:
        dx = []
        for i in range(len(p)):
            dx.append(x[i+1] - x[i])
            x[i] = ( x[i] + x[i+1] ) / 2.
        x = x[:-1]
        dx = np.array(dx)
    else: 
        print('problem')
        return 0
    
    mean = np.sum(x*p*dx)
    std = np.sqrt(np.sum((x-mean)**2.*p*dx))
    x = (x-mean)/std
    p = p * std
    
    dx /= std
    mom4 = np.sum((x-mean)**4.*p*dx)
    return mom4