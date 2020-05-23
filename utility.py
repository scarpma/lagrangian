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
    return hist, bin_edges

#hist1, bin_edges1 = load_pdf('data/pdf_ax')
