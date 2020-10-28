// Compilation: cmd='cc [-DTIMEOUT_S=30] -O3 -DPOSIX -DREADWRITE io_perf.c'; echo $cmd > build.log; $cmd
//              cmd='cc [-DTIMEOUT_S=30] -O3 -DADIOS2 -DREADWRITE io_perf.c'; echo $cmd > build.log; $cmd
// Submit perf test: sbatch dom_slurm.sbatch
//
#ifdef ADIOS2_HDF5
#define ADIOS2
#endif

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "errno.h"
#include "math.h"
#include "mpi.h"
#if defined ADIOS2
#include <adios2_c.h>
#endif

#ifdef POSIX
#define FILE_NAME "data.posix"
#elif defined ADIOS2_HDF5
#define FILE_NAME "data.h5"
#elif defined ADIOS2
#define FILE_NAME "data.bp4"
#endif

#define NDIMS 3
#define NX 512
#define NY 512
#define NZ 1024
#define NT 200
#ifndef NB_FILES
    #define NB_FILES 1
#endif
// The timeout is applied separately for read and write, i.e. a complete run can take maximally 2*TIMEOUT_S seconds + initialization + warmup iterations (it0).
#ifndef TIMEOUT_S
    #define TIMEOUT_S 600
#endif

// Define data type
#define USE_SINGLE_PRECISION

#if defined USE_SINGLE_PRECISION
#define DAT              float
#define ADIOS2_DAT       adios2_type_float
#define PRECIS           4
#elif defined USE_DOUBLE_PRECISION
#define DAT              double
#define ADIOS2_DAT       adios2_type_double
#define PRECIS           8
#elif defined USE_INT
#define DAT              int
#define ADIOS2_DAT       adios2_type_int
#define PRECIS           4
#endif

#if (defined POSIX) || (defined ASCII) || (defined ADIOS2)
#define ERR()    ({ if (status != 0)         ABORT(errno); })
#endif
#define ABORT(e) ({ fflush(stdout); perror("ERROR"); MPI_Abort(MPI_COMM_WORLD, e); })

// Macros/Functions to measure time.
#define tic() ({MPI_Barrier(MPI_COMM_WORLD); time_tic = get_time();           })
#define toc() ({MPI_Barrier(MPI_COMM_WORLD); time_toc = get_time() - time_tic;})

double get_time(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


int main(int argc, char *argv[]){
    double time_tic=0.0, time_toc=0.0, time_s=0.0, time_it=-1.0, time_it_min=-1, time_it_max=-1, GB=-1, GBs=-1, GBs_min=-1, GBs_max=-1, GBs_it=-1;
    double* times_it;
    int it0=5, nb_arrays=-1, nb_ioops=-1, e=0, nit=0, nit_timeout=NT;    
    DAT *data_out, *data_in;
    char* fname;
    FILE* perf_file_id;
    int nprocs, nprocs_world, me, me_world, ierr, color, comm_size=-1, reorder = 0;
    // NOTE: we set all to 3 instead of NDIMS in the next two lines for simpler handling - should not matter if longer, we give just a pointer to the functions, so they only need to read what needed.
    int dims[] = {0,0,0}, coords[]={0,0,0}, periods[] = {0,0,0};
    size_t nxyz[3], nxyz_global[3], offset[3];
    MPI_Offset disp;
    MPI_Comm comm=MPI_COMM_NULL, topo_comm=MPI_COMM_NULL;
    MPI_Info info=MPI_INFO_NULL;
    #if (defined POSIX)
        FILE* file_id; 
        int status;
        size_t nb_processed;
    #elif defined ADIOS2
        adios2_adios* adios;
        adios2_io* io;
        adios2_engine* engine;
        adios2_error status;
        adios2_variable* dset_id; //TODO: change this to var_id probably for all. ADIOS2 and NetCDF call it variable, only HDF5 calls is dataset.
        const float timeout_seconds = 10.0;
        adios2_step_status* step_status;
        step_status = malloc(sizeof(adios2_step_status));
    #endif
    times_it = malloc(NT*sizeof(double));

    // Init MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_world);  if (NB_FILES>nprocs_world) { printf("ERROR: NB_FILES > nprocs_world"); fflush(stdout); MPI_Abort(MPI_COMM_WORLD, 0);}
    MPI_Comm_rank(MPI_COMM_WORLD, &me_world);
    comm_size = nprocs_world/NB_FILES;
    #if (defined POSIX)
        comm_size = 1;
    #endif
    color = floor(me_world/comm_size); // to make equally sized groups with contigous ranks //me_world % NB_FILES;  //TODO: replace probably rather with color that keeps ranks grouped
    MPI_Comm_split(MPI_COMM_WORLD, color, me_world, &comm);
    MPI_Comm_size(comm, &nprocs);
    MPI_Dims_create(nprocs, NDIMS, dims);
    MPI_Cart_create(comm, NDIMS, dims, periods, reorder, &topo_comm);
    MPI_Comm_rank(topo_comm, &me);
    MPI_Cart_coords(topo_comm, me, NDIMS, coords);
    if (me==0) printf("nprocs=%d(nprocs_world=%d,dims(1)=%d,dims(2)=%d,dims(3)=%d\n", nprocs,nprocs_world, dims[0],dims[1],dims[2]);

    // Set file name per communicator process group
    fname = malloc(32*sizeof(char));
    sprintf(fname, "%d_%s", color, FILE_NAME);

    // Init ADIOS2
    #if defined ADIOS2
        //adios = adios2_init_config("adios2.xml", topo_comm, adios2_debug_mode_on);  if (adios==NULL) ABORT(errno);
        adios = adios2_init_config("adios2.xml", topo_comm, adios2_debug_mode_off);  if (adios==NULL) ABORT(errno);
    #endif

    // Size and offset of the local and global problem (in ADIOS2 slang: nxyz: count; nxyz_global: shape; offset: start)
    nxyz[0]        = (size_t) NZ;
    nxyz[1]        = (size_t) NY;
    nxyz[2]        = (size_t) NX;
    nxyz_global[0] = (size_t) dims[0]*nxyz[0]; // Note: in a real-world app, the halo must be handled in addition.
    nxyz_global[1] = (size_t) dims[1]*nxyz[1]; // debug: removed: if (NDIMS>1) 
    nxyz_global[2] = (size_t) dims[2]*nxyz[2]; // debug: removed: if (NDIMS>1)
    offset[0]      = (size_t) coords[0]*nxyz[0]; // Casting required to avoid integer overflow for large NT! IMPORTANT: HDF5: nb elems; MPIIO: nb bytes
    offset[1]      = (size_t) coords[1]*nxyz[1]; // Casting required to avoid integer overflow for large NT! IMPORTANT: HDF5: nb elems; MPIIO: nb bytes
    offset[2]      = (size_t) coords[2]*nxyz[2]; // Casting required to avoid integer overflow for large NT! IMPORTANT: HDF5: nb elems; MPIIO: nb bytes
    
    // Create the data (allocate dynamically to avoid stack overflow)
    data_out = malloc(NX*NY*NZ*sizeof(DAT));
    for (int z=0; z<NZ; z++){
        for (int y=0; y<NY; y++){
            for (int x=0; x<NX; x++){
                data_out[z*NY*NX+y*NX+x] = (DAT)((coords[0]*NZ + z)*dims[1]*NY*dims[2]*NX +  (coords[1]*NY + y)*dims[2]*NX + coords[2]*NX + x);
            }
        }
    }
    
//------------------------------------------------------------------------
// Write the data and measure performance.
#if (defined WRITE) || (defined READWRITE)
    #if defined ADIOS2
        char* cmd;
        cmd = malloc(32*sizeof(char));
        sprintf(cmd, "rm -rf %s", fname);
        printf("%s\n", cmd);
        system(cmd);
    #else
        remove(fname);
    #endif
    MPI_Barrier(MPI_COMM_WORLD); 

    // Create file
    #if defined POSIX
        file_id = fopen(fname, "ab");  if (file_id==NULL) ABORT(errno);
    #endif

    // ADIOS2: the file will only be created once everything is defined and the engine invoked. Here we only define the IO component which allows to specify how things should be done later. It is here and not at the beginning of main() as we typically want a different IO for read and write - to specify different things.
    #if defined ADIOS2_HDF5
        io = adios2_declare_io(adios, "H5File_Write");  if (io==NULL) ABORT(errno);
    #elif defined ADIOS2
        io = adios2_declare_io(adios, "BPFile_Write");  if (io==NULL) ABORT(errno);
    #endif

    // Define Metadata:
    #if defined ADIOS2
        dset_id = adios2_define_variable(io, "pressure", ADIOS2_DAT, NDIMS, nxyz_global, offset, nxyz, adios2_constant_dims_true);  if (dset_id == NULL) ABORT(errno);
    #endif       

    // Create an engine for ADIOS2 - file name and access mode are defined here. (could also be done right after declaring the io - in symmetry with read...)
    #if defined ADIOS2
        engine = adios2_open(io, fname, adios2_mode_write);
    #endif 

    time_s=0.0; nit=0;
    for (int it=0; it<NT; it++){
        if (it >= it0) tic();
        #if defined ADIOS2
            status = adios2_begin_step(engine, adios2_step_mode_append, timeout_seconds, step_status); ERR();
        #endif

        // Write.
        #if defined POSIX
            nb_processed = fwrite(data_out, PRECIS, NX*NY*NZ, file_id);  if (nb_processed!=NX*NY*NZ) ABORT(errno);
        #elif defined ADIOS2
           status = adios2_put(engine, dset_id, data_out, adios2_mode_deferred);  ERR(); 
           status = adios2_end_step(engine); ERR();
        #endif

        //MPI_Barrier(topo_comm);  // Might improve perf (more regular IO requests).
        if (it >= it0){
            time_it = toc();
            time_s += time_it;
            if (time_it < time_it_min || it == it0) time_it_min = time_it;
            if (time_it > time_it_max || it == it0) time_it_max = time_it;
            times_it[it] = time_it;
            nit++;
            if (time_s > TIMEOUT_S){ nit_timeout = nit; break; }
        }
    }
    nb_arrays = 1;
    nb_ioops  = 1; // here 1 operation: write
    GB        = 1/1e9*NX*NY*NZ*PRECIS*nb_arrays;
    GBs       = 1/1e9*NX*NY*NZ*PRECIS*nb_ioops/time_s*nit*nprocs_world;
    GBs_max   = 1/1e9*NX*NY*NZ*PRECIS*nb_ioops/time_it_min*nprocs_world;
    GBs_min   = 1/1e9*NX*NY*NZ*PRECIS*nb_ioops/time_it_max*nprocs_world;
    if (me_world == 0){
        printf("Write performance:\n");
        printf("array size [GB]: %.4f    time [s]: %.10f    nit: %d\n"
               "sustained perf [GB/s]: %.4f    max perf [GB/s]: %.4f    min perf [GB/s]: %.4f\n", GB, time_s, nit, GBs, GBs_max, GBs_min);
        fflush(stdout);

        perf_file_id = fopen("io_perf_write.txt", "w");  if (perf_file_id==NULL) ABORT(errno);
        for (int it=it0; it<it0+nit; it++){
            GBs_it = 1/1e9*NX*NY*NZ*PRECIS*nb_ioops/times_it[it]*nprocs_world;
            status = fprintf(perf_file_id, "%.8f\n", GBs_it);  if (status<0) ABORT(errno);
        }
        fflush(perf_file_id);
        status = fclose(perf_file_id); ERR();
    }

    #if (defined POSIX)
        status = fclose(file_id); ERR();
    #elif defined ADIOS2
        status = adios2_close(engine);  ERR();
    #endif
#endif

//-------------------------------------------------------------------------
// Read the data and measure performance.
#if (defined READ) || (defined READWRITE)
    data_in = malloc(NX*NY*NZ*sizeof(DAT));

    // Open the file
    #if defined POSIX
        file_id = fopen(fname, "rb");  if (file_id==NULL) ABORT(errno);
    #endif

    // ADIOS2: the file will only be created once everything is defined and the engine invoked. Here we only define the IO component which allows to specify how things should be done later. It is here and not at the beginning of main() as we typically want a different IO for read and write - to specify different things.
    #if defined ADIOS2_HDF5
        io = adios2_declare_io(adios, "H5File_Read");  if (io==NULL) ABORT(errno);
    #elif defined ADIOS2
        io = adios2_declare_io(adios, "BPFile_Read");  if (io==NULL) ABORT(errno);
    #endif

    // Create an engine for ADIOS2 - file name and access mode are defined here.
    #if defined ADIOS2
        engine = adios2_open(io, fname, adios2_mode_read);
    #endif 

    #if defined ADIOS2
        dset_id = adios2_inquire_variable(io, "pressure");  if (dset_id == NULL) ABORT(errno);
        adios2_set_selection(dset_id, NDIMS, offset, nxyz);  // NOTE: adios2_variable_ndims, adios2_variable_start and adios2_variable_count let inquire these values from the file (adios2_variable_shape lets inquire nxyz_global)
    #endif   

    time_s=0.0;  nit=0;
    for (int it=0; it<NT; it++){
        if (it >= it0) tic();
        #if defined ADIOS2
            status = adios2_begin_step(engine, adios2_step_mode_read, timeout_seconds, step_status); ERR();
        #endif
        
        // Read.
        #if defined POSIX
            nb_processed = fread(data_in, PRECIS, NX*NY*NZ, file_id);  if (nb_processed!=NX*NY*NZ) ABORT(errno);
        #elif defined ADIOS2
           status = adios2_get(engine, dset_id, data_in, adios2_mode_deferred);  ERR(); 
           status = adios2_end_step(engine); ERR();
        #endif

        if (it >= it0){
            time_it = toc();
            time_s += time_it;
            if (time_it < time_it_min || it == it0) time_it_min = time_it;
            if (time_it > time_it_max || it == it0) time_it_max = time_it;
            times_it[it] = time_it;
            nit++;
            if (time_s > TIMEOUT_S || nit==nit_timeout) break;
        }
    }

    nb_arrays = 1;
    nb_ioops  = 1; // here 1 operation: read
    GB        = 1/1e9*NX*NY*NZ*PRECIS*nb_arrays;
    GBs       = 1/1e9*NX*NY*NZ*PRECIS*nb_ioops/time_s*nit*nprocs_world;
    GBs_max   = 1/1e9*NX*NY*NZ*PRECIS*nb_ioops/time_it_min*nprocs_world;
    GBs_min   = 1/1e9*NX*NY*NZ*PRECIS*nb_ioops/time_it_max*nprocs_world;
    if (me_world == 0){
        printf("Read performance:\n");
        printf("array size [GB]: %.4f    time [s]: %.10f    nit: %d\n"
               "sustained perf [GB/s]: %.4f    max perf [GB/s]: %.4f    min perf [GB/s]: %.4f\n", GB, time_s, nit, GBs, GBs_max, GBs_min);
        fflush(stdout);

        perf_file_id = fopen("io_perf_read.txt", "w");  if (perf_file_id==NULL) ABORT(errno);
        for (int it=it0; it<it0+nit; it++){
            GBs_it = 1/1e9*NX*NY*NZ*PRECIS*nb_ioops/times_it[it]*nprocs_world;
            status = fprintf(perf_file_id, "%.8f\n", GBs_it);  if (status<0) ABORT(errno);
        }
        fflush(perf_file_id);
        status = fclose(perf_file_id); ERR();
    }

    #if defined POSIX
        status = fclose(file_id); ERR();
    #elif defined ADIOS2
        status = adios2_close(engine);  ERR();
    #endif

    //Check the data (if < 2G numbers as then integer overflow)
    if ((MPI_Offset) nprocs*NX*NX*NZ < (MPI_Offset) 2147483648){
        for (int z=0; z<NZ; z++){
            for (int y=0; y<NY; y++){
                for (int x=0; x<NX; x++){
                    if (data_in[z*NY*NX+y*NX+x] != (DAT)((coords[0]*NZ + z)*dims[1]*NY*dims[2]*NX +  (coords[1]*NY + y)*dims[2]*NX + coords[2]*NX + x)){ 
                        printf("FAILED reading (expected: %f; obtained: %f)\n", 
                                                   (DAT)((coords[0]*NZ + z)*dims[1]*NY*dims[2]*NX +  (coords[1]*NY + y)*dims[2]*NX + coords[2]*NX + x), data_in[z*NY*NX+y*NX+x]); ABORT(0); 
                    }
                }
            }
        }
        printf("SUCCESS reading\n"); // TODO: in check, count nb of success.
    }

    free(data_in);
#endif
    free(data_out);

    #if defined ADIOS2
        status = adios2_finalize(adios);  ERR();
    #endif

    MPI_Finalize();

    return 0;
}
