#!/usr/bin/env python
# coding: utf-8

def compute_structure_function(db,npart=None):
    import ou
    import numpy as np
    struct = np.zeros(shape=(34,4),order='f')
    if db.shape[-1] == 2000:
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
            ou.compute_struct(struct,dbn)
            return np.ascontiguousarray(struct)




if __name__ == '__main__' :
    
    run=65
    number=8000
    npart=None
    media=0
    gan_type = 'wgangp'
    
    
    import numpy as np
    


    """
    DATABASE IMPORT
    """

    print("Database import: ", end="")
    if run==0:
        #path_v = f'../databases/gaussian_process.npy'
        path_v = f'../databases/velocities.npy'
        print(path_v)
    elif media==0:
        #path_v = f'wgangps/runs/{run}/gen_trajs_{number}.npy'
        path_v = "../databases/lagrangian/"+gan_type+f'/runs/{run}/gen_trajs_{number}.npy'
        print(path_v)
    else:
        #path_v = f'wgangps/runs/{run}/gen_trajs_{number}_media.npy'
        path_v = "../databases/lagrangian/"+gan_type+f'/runs/{run}/gen_trajs_{number}_media.npy'
        print(path_v)
    
    
    db = np.load(path_v)
    if npart == None:
        npart_save = db.shape[0]
    else :
        npart_save = npart
    
    """
    SAVING
    """

    if run==0: 
        save_path = f"data/real/struct_function_{npart_save}_part"
        print("save_path = ",save_path)
    #elif media == 0: 
    #    save_path = f"data/wgangps/struct_function_{npart_save}_part_gen_{run}_{number}"
    #    print("save_path = ",save_path)
    elif media == 0: 
        save_path = f"data/"+gan_type+f"/struct_function_{npart_save}_part_gen_{run}_{number}"
        print("save_path = ",save_path)
    #else: 
    #    save_path = f"data/"+gan_type+f"/struct_function_{npart_save}_part_gen_{run}_{number}_media"
    #    print("save_path = ",save_path)
    else: 
        save_path = f"data/"+gan_type+f"/struct_function_{npart_save}_part_gen_{run}_{number}_media"
        print("save_path = ",save_path)
    
    



    s = compute_structure_function(db,npart)
    print("Saving ...")
    np.save(save_path,s)
    print("Done!")
