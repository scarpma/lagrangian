#!/usr/bin/env python
# coding: utf-8

def histogram(array,nbins):
    hist = np.zeros(nbins,dtype=np.int32, order='F')
    bins = np.zeros(nbins,dtype=np.float64, order='F')
    dbn = np.asfortranarray(array.flatten().astype('float64'))
    ou.histogram(dbn, hist, bins)
    del dbn
    hist = np.ascontiguousarray(hist)
    bins = np.ascontiguousarray(bins)
    hist = hist.astype('float')
    # normalization
    binw = bins[1]-bins[0]
    hist = hist / (np.sum(hist)*binw)
    return hist, bins




if __name__ == '__main__' :


    import sys
    import os
    import os.path
    import numpy as np
    import ou

    # ARGUMENTS AND OPTIONS PARSING

    option_npart = False
    option_npart_arg = 0
    option_derivative = False

    if len(sys.argv)<3:
        print("usage: compute_pdf.py read_path write_path [-npart <number>] [-derivative]")
        exit()

    read_path = sys.argv[1]
    write_path = sys.argv[2]
    sys.argv.pop(1)
    sys.argv.pop(1)

    print("path read: ", read_path)
    if not os.path.isfile(read_path):
        print("Invalid read_path: file does not exist. Exiting.")
        exit()
    print("path write: ", write_path)
    if os.path.isfile(write_path):
        print("Write path already exists. Continuing.")
        #exit()
    if not os.path.isdir(os.path.split(write_path)[0]):
        print("Invalid write_path: dir does not exist. Exiting.")
        exit()

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "-npart":
            option_npart = True
            sys.argv.pop(i)
            option_npart_arg = int(sys.argv.pop(i))
        if sys.argv[i] == "-derivative":
            option_derivative = True
            sys.argv.pop(i)
        elif sys.argv[i] == "-p":
            print("Unknown argument, continuing ...")
        else:
            i += 1




    # DATABASE IMPORT

    db = np.load(read_path)
    if not option_npart:
        option_npart_arg = db.shape[0]
    else:
        db = db[np.random.randint(0,db.shape[0],option_npart_arg)]

    if option_derivative:
        db = np.gradient(db, axis=1)


    hist, bins = histogram(db,db.size//200000)
    temp = np.zeros(shape=(len(hist),2))
    temp[:,0] = bins
    temp[:,1] = hist
    print("Saving ...")
    np.savetxt(write_path,temp)
    print("Done!")
