#!/usr/bin/env python
# coding: utf-8


def create_log_bins(xmin,xmax,nbin,eps):
    import numpy as np
    if xmin*xmax<0.:
        estremo = np.max([xmax,-xmin])
        bins1 = np.logspace(np.log10(eps),np.log10(estremo), nbin//2)
        bins = np.r_[-bins1[::-1],[0.],bins1]
        assert (np.diff(bins) > 0).all()
        return bins
    elif xmax>0. and xmin>0.:
        bins = np.logspace(np.log10(xmin),np.log10(xmax), nbin)
        return bins
    else:
        print("problem")
        return 0


def make_hist(samples, bins='lin', std=False, out=False, hist=False, no_mean=None):
    import numpy as np

    std_done = False

    if type(samples)==type('str') and samples[-7:]==".pickle" :

        import pickle
        with open(samples, 'rb') as f:
            (hist, bins) = pickle.load(f)

        if std :
            std_done = True
            stad = 0.
            mean = 0.
            for i in range(len(hist)):
                mean += (bins[i+1] - bins[i])*hist[i]*(bins[i+1] + bins[i])/2
            for i in range(len(hist)):
                stad += (bins[i+1] - bins[i])*hist[i]*((bins[i+1] + bins[i])/2 - mean)**2.
            stad = np.sqrt(stad)
            hist = hist * stad
            bins = (bins - mean) / stad

    elif type(samples)==type('str') and samples[-4:]==".dat" :

        import numpy
        bh = np.loadtxt(samples)
        if std or out:
            std_done = True
            stad = 0.
            mean = 0.
            mean = mean + (bh[0,1]-bh[0,0])*bh[1,0]*bh[0,0]
            for i in range(1,bh.shape[1]-1):
                mean = mean + ((bh[0,i+1]-bh[0,i-1])/2.)*bh[1,i]*bh[0,i]
            mean = mean + (bh[0,-1]-bh[0,-2])*bh[1,-1]*bh[0,-1]
            stad = stad + (bh[0,1]-bh[0,0])*bh[1,0]*(bh[0,0]-mean)**2.
            for i in range(1,bh.shape[1]-1):
                stad = stad + ((bh[0,i+1]-bh[0,i-1])/2.)*bh[1,i]*(bh[0,i]-mean)**2.
            stad = stad + (bh[0,-1]-bh[0,-2])*bh[1,-1]*(bh[0,-1]-mean)**2.
            stad = np.sqrt(stad)
            if std:
                if no_mean is None:
                    bh[1,:] = bh[1,:] * stad
                    bh[0,:] = (bh[0,:] - mean) / stad
                elif no_mean:
                    bh[1,:] = bh[1,:] * stad
                    bh[0,:] = bh[0,:] / stad

        if out : return bh, mean, stad

        return bh

    elif type(samples) == type(np.ndarray([])) :
        ## CASO STANDARD IN CUI VIENE FORNITO UN DB IN FORMATO NUMPY ARRAY
        if hist:
            hist, bins = np.histogram(samples.flatten(), bins=bins)
        else:
            hist, bins = np.histogram(samples.flatten(), bins=bins, density=True)

    else: raise NameError("'samples' type not recognized")

    if (std or out) and not std_done :
        mean = samples.mean()
        stad = samples.std()
        if std:
            print(stad)
            hist = hist * stad
            bins = (bins - mean) / stad

    assert len(hist) == len(bins)-1

    for i in range(len(hist)):
        bins[i] = (bins[i]+bins[i+1])/2.
    bins = bins[:-1]

    binhist = np.zeros(shape=(2,len(hist)))
    binhist[0,:] = bins
    binhist[1,:] = hist

    if out : return binhist, mean, stad

    return binhist



def compute_sf(db,npart=None):
    import ou
    import numpy as np
    struct = np.zeros(shape=(34,7),order='f')
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

