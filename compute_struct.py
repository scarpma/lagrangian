#!/usr/bin/env python
# coding: utf-8

def compute_structure_function(db,npart=None):
    import ou
    import numpy as np
    struct = np.zeros(shape=(34,4),order='f')
    if len(db.shape)==2:
        print("Database shape ok, continuing...")
        if npart != None:
            print(f"Database larger than npart. Taken {npart} samples randomly.")
            idx = np.random.randint(0,db.shape[0],npart)
            print("Converting database to fortran order")
            dbn = np.asfortranarray(db[idx].T)
            ou.compute_struct(struct,dbn)
            return np.ascontiguousarray(struct)
        elif npart == None:
            print(f"Taking entire dataset, {npart} samples")
            print("Converting database to fortran order")
            dbn = np.asfortranarray(db.T)
            ou.compute_struct(struct,dbn)
            return np.ascontiguousarray(struct)
    else:
        print("Database with more than one component. Only x taken")
        if npart != None:
            print(f"Database larger than npart. Taken {npart} samples randomly.")
            idx = np.random.randint(0,db.shape[0],npart)
            print("Converting database to fortran order")
            dbn = np.asfortranarray(db[idx,:,0].T)
            ou.compute_struct(struct,dbn)
            return np.ascontiguousarray(struct)
        elif npart == None:
            print(f"Taking entire dataset, {npart} samples")
            print("Converting database to fortran order")
            dbn = np.asfortranarray(db[:,:,0].T)
            print(dbn.shape)
            ou.compute_struct(struct,dbn)
            return np.ascontiguousarray(struct)




if __name__ == '__main__' :


    import sys
    import os
    import os.path
    import numpy as np

    # ARGUMENTS AND OPTIONS PARSING

    option_npart = False
    option_npart_arg = 0

    if len(sys.argv)<3:
        print("usage: compute_struct.py read_path write_path [-npart <number>]")
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
        elif sys.argv[i] == "-p":
            print("Unknown argument, continuing ...")
        else:
            i += 1




    # DATABASE IMPORT

    db = np.load(read_path)
    if not option_npart:
        option_npart_arg = db.shape[0]



    s = compute_structure_function(db,option_npart_arg)
    print("Saving ...")
    np.savetxt(write_path,s)
    print("Done!")
