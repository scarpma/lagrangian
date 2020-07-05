/************************************************************
 *
 *   This example shows how to read and write data to a compact
 *     dataset.  The program first writes integers to a compact
 *       dataset with dataspace dimensions of DIM0xDIM1, then
 *         closes the file.  Next, it reopens the file, reads back
 *           the data, and outputs it to the screen.
 *
 *             This file is intended for use with HDF5 Library version 1.8
 *
 *              ************************************************************/

#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>

#define FILE            "/scratch/bonaccorso/LagrangianTracers/velocities.h5"
#define DATASET         "vel3d"
#define DIM0            327680
#define DIM1            2000

#define DIM2            3





int read_dataset(filename, rdata) {
  
    /*
 *      * Open file and dataset using the default properties.
 *           */
    file = H5Fopen (FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen (file, DATASET, H5P_DEFAULT);


 
//   /*
// *      * Retrieve the dataset creation property list, and print the
// *           * storage layout.
// *                */
//    dcpl = H5Dget_create_plist (dset);
//    layout = H5Pget_layout (dcpl);
//    printf ("Storage layout for %s is: ", DATASET);
//    switch (layout) {
//        case H5D_COMPACT:
//            printf ("H5D_COMPACT\n");
//            break;
//        case H5D_CONTIGUOUS:
//            printf ("H5D_CONTIGUOUS\n");
//            break;
//        case H5D_CHUNKED:
//            printf ("H5D_CHUNKED\n");
//    }



    /*
 *      * Read the data using the default properties.
 *           */
    status = H5Dread (dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                rdata[0]);

    /*
 *      * Output the data to the screen.
 *           */
    printf ("%s:\n", DATASET);
    for (i=0; i<DIM0; i++) {
        printf (" [");
        for (j=0; j<DIM1; j++)
            printf (" [");
            for (k=0; k<DIM2; k++)
                printf (" %10.5g", rdata[i][j][k]);
            printf ("]\n");
        printf ("]\n");
        
    }

    /*
 *      * Close and release resources.
 *           */
    status = H5Pclose (dcpl);
    status = H5Dclose (dset);
    status = H5Fclose (file);

    return 0; 
    
}
