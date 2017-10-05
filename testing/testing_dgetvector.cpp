/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from testing_zgetvector.cpp normal z -> d, Mon Feb 13 19:04:28 2017
       @author Mark Gates
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "cuda_runtime_api.h"

// includes, project
#include "testings.h"  // before magma.h, to include cublas_v2
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"  // internal header
#endif

void test1(magma_opts *opts);
void test2(magma_opts *opts);
void test3(magma_opts *opts);
void test4(magma_opts *opts);
void test5(magma_opts *opts);
void test6(magma_opts *opts);

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgemm
*/
int main( int argc, char** argv) {
    int iam_mpi, num_mpi;
    #if 1
    MPI_Init( &argc, &argv );
    #else
    int provided = MPI_THREAD_FUNNELED;
    MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
    MPI_Comm_rank( MPI_COMM_WORLD, &iam_mpi );
    MPI_Comm_size( MPI_COMM_WORLD, &num_mpi );
    if (provided != MPI_THREAD_FUNNELED && iam_mpi == 0) {
        switch(provided) {
            case MPI_THREAD_SINGLE:     printf( "\n ** Only provided SINGLE **\n" ); break;
            case MPI_THREAD_FUNNELED:   printf( "\n ** Only provided FUNNELED **\n" ); break;
            case MPI_THREAD_SERIALIZED: printf( "\n ** Only provided SERIALIZED **\n" ); break;
            case MPI_THREAD_MULTIPLE:   printf( "\n ** Only provided MULTIPLE **\n" ); break;
            default: printf( "??\n"); break;
        }
    }
    #endif
    TESTING_CHECK( magma_init() );

    int name_len;
    char proc_name[300];
    MPI_Get_processor_name( proc_name, &name_len );
    printf( "processor %d on %s\n",iam_mpi,proc_name );

    if (iam_mpi == 0) {
        printf( "%% Usage: %s [options] [-h|--help]\n\n", argv[0] );
        magma_print_environment();
    }

    magma_opts opts;
    opts.parse_opts( argc, argv );

    switch (opts.version) {
    case 1: // CPU <-> GPU (set/get)
        test1(&opts);
        break;
    case 2: // CPU <-> GPUs (set/get)
        test2(&opts);
        break;
    case 3: // GPU <-> GPU (on node)
        test3(&opts);
        break;
    case 4: // CPU <-> CPU (MPI)
        test4(&opts);
        break;
    case 5: // GPU <-> GPU (MPI)
        test5(&opts);
        break;
    case 6: // GPU <-> GPU (MPI)
        test6(&opts);
        break;
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    MPI_Finalize();
    return 0;
}

// get/set CPU/GPU
void test1(magma_opts *opts) {
    magma_int_t ione = 1;
    magma_int_t incx  = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    for (int d=0; d<opts->ngpu; d++) {
        double *X;
        magmaDouble_ptr dX;

        magma_setdevice(d);
        magma_queue_t queue;
        magma_queue_create( &queue );

        printf( "\n set/get CPU <-> GPU%d\n\n",d );
        printf( "%% M       MB         Set GB/s (ms)  Get GB/s (ms)\n" );
        printf( "%%================================================\n" );
        fflush( stdout );
        for( int itest = 0; itest < opts->ntest; ++itest ) {
            magma_int_t Xm = opts->msize[itest];
            magma_int_t sizeX = incx*Xm;

            //TESTING_MALLOC_CPU( X,  double, sizeX );
            //TESTING_MALLOC_PIN( X,  double, sizeX );
            //TESTING_MALLOC_DEV( dX, double, sizeX );
            TESTING_CHECK( magma_dmalloc_pinned( &X, sizeX ) );
            TESTING_CHECK( magma_dmalloc( &dX, sizeX ) );

            /* Initialize the matrices */
            lapackf77_dlarnv( &ione, ISEED, &sizeX, X );

            double mbytes = ((double)sizeof(double)*Xm)/1e6;
            printf( "%6d %7.2f ", Xm, mbytes );
            real_Double_t magma_time = magma_sync_wtime( NULL );
            for( int iter = 0; iter < opts->niter; ++iter ) {
                magma_dsetvector_async( Xm, X, incx, dX, incx, queue );
            }
            magma_time = magma_sync_wtime( NULL ) - magma_time;
            printf( "    %7.2f ", ((double)opts->niter * mbytes)/(1000.*magma_time) );
            printf( "(%.2f)", (1000.*magma_time)/((double)opts->niter) );

            magma_time = magma_sync_wtime( NULL );
            for( int iter = 0; iter < opts->niter; ++iter ) {
                magma_dgetvector_async( Xm, dX, incx, X, incx, queue );
            }
            magma_time = magma_sync_wtime( NULL ) - magma_time;
            printf( "  %7.2f", ((double)opts->niter * mbytes)/(1000.*magma_time) );
            printf( "(%.2f)\n", (1000.*magma_time)/((double)opts->niter) );
        }

        magma_queue_destroy( queue );
        //TESTING_FREE_CPU( X );
        magma_free_pinned( X );
        magma_free( dX );
        fflush( stdout );
    }
}

// get/set between CPU and multiple GPUs
void test2(magma_opts *opts) {
    real_Double_t  magma_time;
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    int iam_mpi, num_mpi;
    MPI_Comm_rank( MPI_COMM_WORLD, &iam_mpi );
    MPI_Comm_size( MPI_COMM_WORLD, &num_mpi );

    int num_gpu = opts->ngpu;
    int num_proc_node = opts->nsub;
    if (num_proc_node <= 0) num_proc_node = 4;
    int iam_gpu = (opts->offset)+num_gpu*(iam_mpi%num_proc_node);

    int num_gpu_node = 4;
    magma_device_t devices[ MagmaMaxGPUs ];
    magma_getdevices( devices, MagmaMaxGPUs, &num_gpu_node );

    magma_queue_t queue[MagmaMaxGPUs];
    printf( "\n %d: set/gpu CPU <-> GPU(%d:%d) (%d procs/node, %d gpus/node)",iam_mpi,iam_gpu,iam_gpu+num_gpu-1,num_proc_node,num_gpu_node );
    for (int d=iam_gpu; d<iam_gpu+num_gpu; d++) {
        int dd=d%num_gpu_node;
        magma_setdevice(dd);
        magma_queue_create( &queue[dd] );
    }

    if (iam_mpi == 0) {
        printf( "%% M       MB         Set GB/s (ms)  Get GB/s (ms)\n" );
        printf( "%%================================================\n" );
        fflush( stdout );
    }
    for( int itest = 0; itest < opts->ntest; ++itest ) {
        double *X[MagmaMaxGPUs];
        magmaDouble_ptr dX[MagmaMaxGPUs];

        magma_int_t Xm = opts->msize[itest];
        magma_int_t sizeX = Xm;

        for (int d=iam_gpu; d<iam_gpu+num_gpu; d++) {
            /* Initialize the matrices */
            int dd=d%num_gpu_node;
            magma_setdevice(dd);
            TESTING_CHECK( magma_dmalloc_pinned( &X[dd],  sizeX ) );
            TESTING_CHECK( magma_dmalloc( &dX[dd], sizeX ) );
            lapackf77_dlarnv( &ione, ISEED, &sizeX, X[dd] );
        }

        double mbytes = ((double)sizeof(double)*Xm)/1e6;
        if (iam_mpi == 0) {
            printf( "%6d %7.2f ", Xm, mbytes );
        }
        /* synch all GPUs */
        for (int d=iam_gpu; d<iam_gpu+num_gpu; d++) {
            int dd=d%num_gpu_node;
            magma_queue_sync( queue[dd] );
        }
        /* do Set */
        MPI_Barrier(MPI_COMM_WORLD);
        magma_time = magma_wtime();
        for( int iter = 0; iter < opts->niter; ++iter ) {
            for (int d=iam_gpu; d<iam_gpu+num_gpu; d++) {
                int dd=d%num_gpu_node;
                magma_setdevice(dd);
                magma_dsetvector_async( Xm, X[dd], 1, dX[dd], 1, queue[dd] );
            }
        }
        /* synch all GPUs */
        for (int d=iam_gpu; d<iam_gpu+num_gpu; d++) {
            int dd=d%num_gpu_node;
            magma_queue_sync( queue[dd] );
        }
        MPI_Barrier(MPI_COMM_WORLD);
        magma_time = magma_wtime() - magma_time;
        if (iam_mpi == 0) {
            printf( "    %7.2f ", ((double)opts->niter * mbytes)/(1000.*magma_time) );
            printf( "(%.2f)", (1000.*magma_time)/((double)opts->niter) );
        }


        /* synch all GPUs */
        for (int d=iam_gpu; d<iam_gpu+num_gpu; d++) {
            int dd=d%num_gpu_node;
            magma_queue_sync( queue[dd] );
        }
        /* do Get */
        MPI_Barrier(MPI_COMM_WORLD);
        magma_time = magma_wtime();
        for( int iter = 0; iter < opts->niter; ++iter ) {
            #define DISABLE_PARCPU
            #if !defined (DISABLE_PARCPU) && defined(_OPENMP)
            magma_int_t nthreads = magma_get_lapack_numthreads();
            magma_set_lapack_numthreads(1);
            magma_set_omp_numthreads(num_gpu);
            #pragma omp parallel for schedule(dynamic)
            #endif
            for (int d=iam_gpu; d<iam_gpu+num_gpu; d++) {
                int dd=d%num_gpu_node;
                magma_setdevice(dd);
                magma_dgetvector_async( Xm, dX[dd], 1, X[dd], 1, queue[dd] );
            }
            #if !defined (DISABLE_PARCPU) && defined(_OPENMP)
            magma_set_lapack_numthreads(nthreads);
            #endif
        }
        /* synch all GPUs */
        for (int d=iam_gpu; d<iam_gpu+num_gpu; d++) {
            int dd=d%num_gpu_node;
            magma_queue_sync( queue[dd] );
        }
        MPI_Barrier(MPI_COMM_WORLD);
        magma_time = magma_wtime() - magma_time;
        if (iam_mpi == 0) {
            printf( "  %7.2f ", ((double)opts->niter * mbytes)/(1000.*magma_time) );
            printf( "(%.2f)\n", (1000.*magma_time)/((double)opts->niter) );
        }
        for (int d=iam_gpu; d<iam_gpu+num_gpu; d++) {
            int dd=d%num_gpu_node;
            magma_setdevice(dd);
            magma_free_pinned( X[dd] );
            magma_free( dX[dd] );
        }
        fflush( stdout );
    }
    for (int d=iam_gpu; d<iam_gpu+num_gpu; d++) {
        int dd=d%num_gpu_node;
        magma_setdevice(dd);
        magma_queue_destroy( queue[dd] );
    }
}

// copy between multiple GPUs on a node
void test3(magma_opts *opts) {
    real_Double_t  magma_time;
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    double c_mone = -1.0;

    int num_gpus = opts->ngpu;
    magma_queue_t queue[MagmaMaxGPUs];
    for (int d=0; d<num_gpus; d++) {
        magma_setdevice(d);
        magma_queue_create( &queue[d] );
        printf( ",%d",d );
    }
    printf( "\n\n" );

    for (int source=0; source<num_gpus; source++) {
        for (int dest=0; dest<num_gpus; dest++) {
            printf( "\n set GPU[%d] -> GPU[%d] on node\n",source,dest );
            printf( "%% M       MB         Set GB/s (ms)  Get GB/s (ms)\n" );
            printf( "%%================================================\n" );
            fflush( stdout );
            for( int itest = 0; itest < opts->ntest; ++itest ) {
                double *X, *Y;
                magmaDouble_ptr dX[MagmaMaxGPUs];

                magma_int_t Xm = opts->msize[itest];
                magma_int_t sizeX = Xm;

                /* Initialize the matrices */
                TESTING_CHECK( magma_dmalloc_pinned( &X,  sizeX ) );
                TESTING_CHECK( magma_dmalloc_pinned( &Y,  sizeX ) );
                lapackf77_dlarnv( &ione, ISEED, &sizeX, X );

                magma_setdevice(dest);
                TESTING_CHECK( magma_dmalloc( &dX[dest],   sizeX ) );

                magma_setdevice(source);
                TESTING_CHECK( magma_dmalloc( &dX[source], sizeX ) );
                magma_dsetvector_async( Xm, X, 1, dX[source], 1, queue[source] );

                double mbytes = ((double)sizeof(double)*Xm)/1e6;
                printf( "%6d %7.2f ", Xm, mbytes );

                // get/set
                magma_time = magma_sync_wtime( NULL );
                for( int iter = 0; iter < opts->niter; ++iter ) {
                    magma_setdevice(source);
                    magma_dgetvector_async( Xm, dX[source], 1, Y, 1, queue[source] );
                    magma_queue_sync( queue[source] );

                    magma_setdevice(dest);
                    magma_dsetvector_async( Xm, Y, 1, dX[dest], 1, queue[dest] );
                }
                magma_time = magma_sync_wtime( NULL ) - magma_time;
                printf( "  %7.2f ", ((double)opts->niter * mbytes)/(1000.*magma_time) );
                printf( "(%.2f)", (1000.*magma_time)/((double)opts->niter) );
                if (opts->check != 0) {
                    magma_setdevice(dest);
                    magma_dgetvector( Xm, dX[dest], 1, Y, 1 );

                    double work[1];
                    blasf77_daxpy( &Xm, &c_mone, X, &ione, Y, &ione );
                    double magma_error = lapackf77_dlange( "F", &Xm, &ione, Y, &Xm, work );
                    printf( ", error=%.2e", magma_error );
                }

                // copy
                if (cudaSuccess == cudaMemcpy( dX[dest], dX[source], Xm*sizeof(double), cudaMemcpyDeviceToDevice )) {
                    magma_time = magma_sync_wtime( NULL );
                    for( int iter = 0; iter < opts->niter; ++iter ) {
                        magma_setdevice(source);
                        cudaMemcpyAsync( dX[dest], dX[source], Xm*sizeof(double), cudaMemcpyDeviceToDevice,
                                         magma_queue_get_cuda_stream( queue[source] ) );
                    }
                    magma_time = magma_sync_wtime( NULL ) - magma_time;
                    printf( "  %7.2f ", ((double)opts->niter * mbytes)/(1000.*magma_time) );
                    printf( "(%.2f)", (1000.*magma_time)/((double)opts->niter) );
                    if (opts->check != 0) {
                        magma_setdevice(dest);
                        magma_dgetvector( Xm, dX[dest], 1, Y, 1 );

                        double work[1];
                        blasf77_daxpy( &Xm, &c_mone, X, &ione, Y, &ione );
                        double magma_error = lapackf77_dlange( "F", &Xm, &ione, Y, &Xm, work );
                        printf( ", error=%.2e", magma_error );
                    }
                } else {
                    printf( " cudaMemcpy(failed)" );
                }

                // peer copy
                if (cudaDeviceEnablePeerAccess( dest, 0 ) == cudaSuccess) {
                    int canAccessPeer;
                    cudaDeviceCanAccessPeer(&canAccessPeer, source, dest);
                    if (canAccessPeer == cudaSuccess) {
                        magma_time = magma_sync_wtime( NULL );
                        for( int iter = 0; iter < opts->niter; ++iter ) {
                            magma_setdevice(source);
                            cudaMemcpyPeerAsync( dX[dest], dest, dX[source], source, Xm*sizeof(double),
                                                 magma_queue_get_cuda_stream( queue[source] ) );
                        }
                        magma_time = magma_sync_wtime( NULL ) - magma_time;
                        printf( "  %7.2f ", ((double)opts->niter * mbytes)/(1000.*magma_time) );
                        printf( "(%.2f)\n", (1000.*magma_time)/((double)opts->niter) );
                    } else {
                        printf( " cudaCanAccessPeer(failed)\n" );
                    }
                    cudaDeviceDisablePeerAccess( dest );
                } else {
                    printf( " cudaDeviceEnablePeerAccess(failed)\n" );
                }
                fflush( stdout );

                magma_free_pinned( Y );
                magma_free_pinned( X );
                magma_free( dX[source] );
                magma_free( dX[dest] );
            }
        }
    }
    for (int d=0; d<num_gpus; d++) {
        magma_setdevice(d);
        magma_queue_destroy( queue[d] );
    }
}

// MPI-comm between CPUs
void test4(magma_opts *opts) {
    real_Double_t  magma_time;
    magma_int_t ISEED[4] = {0,0,0,1};

    double c_mone = -1.0;
    magma_int_t ione = 1;

    int iam_mpi, num_mpi;
    MPI_Comm_rank( MPI_COMM_WORLD, &iam_mpi );
    MPI_Comm_size( MPI_COMM_WORLD, &num_mpi );

    if (iam_mpi == 0) {
        printf( " Allgatherv(CPU <-> CPU)\n\n" );
        printf( "%% M       MB         Set GB/s (ms)  Get GB/s (ms)\n" );
        printf( "%%================================================\n" );
        fflush( stdout );
    }
    for( int itest = 0; itest < opts->ntest; ++itest ) {
        double *X, *Xg, *Xloc;

        magma_int_t m = opts->msize[itest];
        magma_int_t mloc = (m+num_mpi-1)/num_mpi;
        
        int *recvcounts = (int*)malloc(num_mpi * sizeof(int));
        int *displs = (int*)malloc((1+num_mpi) * sizeof(int));

        displs[0] = 0;
        for( int p=0; p<num_mpi; p++) {
            recvcounts[p] = (p < num_mpi-1 ? mloc : m-p*mloc);
            displs[p+1] = displs[p] + recvcounts[p];
        }
        mloc = recvcounts[iam_mpi];

        /* Initialize the matrices */
        TESTING_CHECK( magma_dmalloc_cpu( &X, m ) );
        TESTING_CHECK( magma_dmalloc_cpu( &Xg, m ) );
        lapackf77_dlarnv( &ione, ISEED, &m, Xg );
        Xloc = &Xg[displs[iam_mpi]];

        double mbytes = ((double)sizeof(double)*m)/1e6;
        if (iam_mpi == 0) {
            printf( "%6d %7.2f ", m, mbytes );
        }
        magma_time = magma_sync_wtime( NULL );
        for( int iter = 0; iter < opts->niter; ++iter ) {
            MPI_Allgatherv( Xloc, mloc, MPI_DOUBLE, X, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD );
        }
        magma_time = magma_sync_wtime( NULL ) - magma_time;
        if (iam_mpi == 0) {
            printf( "    %7.2f ", ((double)opts->niter * mbytes)/(1000.*magma_time) );
            printf( "(%.2f)", (1000.*magma_time)/((double)opts->niter) );
            if (opts->check != 0) {
                double work[1];
                blasf77_daxpy( &m, &c_mone, X, &ione, Xg, &ione );
                double magma_error = lapackf77_dlange( "F", &m, &ione, Xg, &m, work );
                printf( ", error=%.2e", magma_error );
            }
            printf( "\n" );
            fflush( stdout );
        }
        free(displs); free(recvcounts);
        magma_free_cpu( X );
        magma_free_cpu( Xg );
    }
}

// MPI-comm between GPUs
void test5(magma_opts *opts) {
    real_Double_t  magma_time;
    magma_int_t ISEED[4] = {0,0,0,1};

    double c_mone = -1.0;
    magma_int_t ione = 1;

    int iam_mpi, num_mpi;
    int num_gpu = opts->ngpu;
    int num_proc_node = opts->nsub;
    MPI_Comm_rank( MPI_COMM_WORLD, &iam_mpi );
    MPI_Comm_size( MPI_COMM_WORLD, &num_mpi );

    printf( " processor %d uses GPU %d:%d\n",iam_mpi,num_gpu*(iam_mpi%num_proc_node),num_gpu*(1+(iam_mpi%num_proc_node))-1 );
    if (iam_mpi == 0) {
        printf( " Allgatherv(GPU <-> GPU), %d procs/node\n\n",num_proc_node );
        printf( "%% M       MB         Set GB/s (ms)  Get GB/s (ms)\n" );
        printf( "%%================================================\n" );
        fflush( stdout );
    }
    for( int itest = 0; itest < opts->ntest; ++itest ) {
        double *gX, *hX, *hXloc;
        double *dX[MagmaMaxGPUs], *dXloc[MagmaMaxGPUs];

        magma_int_t m = opts->msize[itest];
        magma_int_t mloc = (m+num_mpi-1)/num_mpi;
        magma_int_t mloc_d, disp_d;
        
        int *recvcounts = (int*)malloc(num_mpi * sizeof(int));
        int *displs = (int*)malloc((1+num_mpi) * sizeof(int));

        displs[0] = 0;
        for( int p=0; p<num_mpi; p++) {
            recvcounts[p] = (p < num_mpi-1 ? mloc : m-p*mloc);
            displs[p+1] = displs[p] + recvcounts[p];
        }
        mloc = recvcounts[iam_mpi];
        mloc_d = (mloc+num_gpu-1)/num_gpu;

        /* Initialize the matrices */
        TESTING_CHECK( magma_dmalloc_pinned( &gX, m ) );
        TESTING_CHECK( magma_dmalloc_pinned( &hX, m ) );
        lapackf77_dlarnv( &ione, ISEED, &m, gX );
        hXloc = &gX[displs[iam_mpi]];

        disp_d = 0;
        magma_queue_t queue[MagmaMaxGPUs];
        for (int d=0; d<num_gpu; d++) {
            magma_setdevice((num_gpu*(iam_mpi%num_proc_node) + d));
            magma_queue_create( &queue[d] );

            int mloc_l = (mloc-d*mloc_d < mloc_d ? mloc_d*mloc_d : mloc_d);
            TESTING_CHECK( magma_dmalloc( &dX[d], m ) );
            TESTING_CHECK( magma_dmalloc( &dXloc[d], mloc_l ) );
            magma_dsetvector_async( mloc_l, &hXloc[disp_d], 1, dXloc[d], 1, queue[d] );
            disp_d += mloc_l;
        }

        double mbytes = ((double)sizeof(double)*m)/1e6;
        if (iam_mpi == 0) {
            printf( "%6d %7.2f ", m, mbytes );
        }
        magma_time = magma_sync_wtime( NULL );
        for( int iter = 0; iter < opts->niter; ++iter ) {
            disp_d = 0;
            for (int d=0; d<num_gpu; d++) {
                magma_setdevice((num_gpu*(iam_mpi%num_proc_node) + d));

                int mloc_l = (mloc-d*mloc_d < mloc_d ? mloc_d*mloc_d : mloc_d);
                magma_dgetvector_async( mloc_l, dXloc[d], 1, &hXloc[disp_d], 1, queue[d] );
                disp_d += mloc_l;
            }
            for (int d=0; d<num_gpu; d++) {
                magma_setdevice((num_gpu*(iam_mpi%num_proc_node) + d));
                magma_queue_sync( queue[d] );
            }
            MPI_Allgatherv( hXloc, mloc, MPI_DOUBLE, hX, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD );
            for (int d=0; d<num_gpu; d++) {
                magma_setdevice((num_gpu*(iam_mpi%num_proc_node) + d));
                magma_dsetvector_async( m, hX, 1, dX[d], 1, queue[d] );
            }
            for (int d=0; d<num_gpu; d++) {
                magma_setdevice((num_gpu*(iam_mpi%num_proc_node) + d));
                magma_queue_sync( queue[d] );
            }
        }
        magma_time = magma_sync_wtime( NULL ) - magma_time;

        if (iam_mpi == 0) {
            printf( "    %7.2f ", ((double)opts->niter * mbytes)/(1000.*magma_time) );
            printf( "(%.2f)", (1000.*magma_time)/((double)opts->niter) );
            if (opts->check != 0) {
                double work[1];
                blasf77_daxpy( &m, &c_mone, hX, &ione, gX, &ione );
                double magma_error = lapackf77_dlange( "F", &m, &ione, gX, &m, work );
                printf( ", error=%.2e", magma_error );
            }
            printf( "\n" );
            fflush( stdout );

        }
        for (int d=0; d<num_gpu; d++) {
            magma_setdevice((num_gpu*(iam_mpi%num_proc_node) + d));
            magma_queue_destroy( queue[d] );
            magma_free( dX[d] );
            magma_free( dXloc[d] );
        }
        free(displs); free(recvcounts);
        magma_free( dX );
        magma_free( dXloc );
        magma_free_pinned( hX );
        magma_free_pinned( gX );
    }
}

// MPI-comm between GPUs
void test6(magma_opts *opts) {
    real_Double_t  magma_time;
    magma_int_t ISEED[4] = {0,0,0,1};

    double c_one  =  1.0;
    magma_int_t ione = 1;

    int iam_mpi, num_mpi;
    int num_gpu = opts->ngpu;
    int num_proc_node = opts->nsub;
    MPI_Comm_rank( MPI_COMM_WORLD, &iam_mpi );
    MPI_Comm_size( MPI_COMM_WORLD, &num_mpi );
    int iam_gpu = num_gpu*(iam_mpi%num_proc_node);

    printf( " processor %d uses GPU %d:%d\n",iam_mpi,iam_gpu,iam_gpu+num_gpu-1 );
    if (iam_mpi == 0) {
        printf( " Allgatherv(GPU <-> GPU), %d procs/node\n\n",num_proc_node );
        printf( "%% M       MB         Set GB/s (ms)  Get GB/s (ms)\n" );
        printf( "%%================================================\n" );
        fflush( stdout );
    }
    for( int itest = 0; itest < opts->ntest; ++itest ) {
        double *gX, *hX, *hXloc;
        double *dX[MagmaMaxGPUs], *dXloc[MagmaMaxGPUs], *dBuffer[MagmaMaxGPUs];

        magma_int_t m = opts->msize[itest];
        magma_int_t mloc = (m+num_mpi-1)/num_mpi;
        
        int *recvcounts = (int*)malloc(num_mpi * sizeof(int));
        int *displs = (int*)malloc((1+num_mpi) * sizeof(int));

        displs[0] = 0;
        for( int p=0; p<num_mpi; p++) {
            recvcounts[p] = (p < num_mpi-1 ? mloc : m-p*mloc);
            displs[p+1] = displs[p] + recvcounts[p];
        }
        mloc = recvcounts[iam_mpi];

        /* Initialize the matrices */
        magma_setdevice(iam_gpu);
        TESTING_CHECK( magma_dmalloc_pinned( &gX, m ) );
        TESTING_CHECK( magma_dmalloc_pinned( &hX, m ) );
        TESTING_CHECK( magma_dmalloc_pinned( &hXloc, mloc ) );
        lapackf77_dlarnv( &ione, ISEED, &m, gX );

        magma_queue_t queue[MagmaMaxGPUs];
        magma_event_t event[MagmaMaxGPUs];
        for (int d=0; d<num_gpu; d++) {
            magma_setdevice(iam_gpu + d);
            magma_queue_create( &queue[d] );
            magma_event_create( &event[d] );

            TESTING_CHECK( magma_dmalloc( &dX[d], m ) );
            TESTING_CHECK( magma_dmalloc( &dXloc[d], mloc ) );
            //magma_dsetvector( mloc, &gX[displs[iam_mpi]], 1, dXloc[d], 1 );
            magma_dsetvector_async( mloc, &gX[displs[iam_mpi]], 1, dXloc[d], 1, queue[d] );
            magma_queue_sync( queue[d] );
        }
        magma_setdevice(iam_gpu);
        for (int d=0; d<num_gpu; d++) {
            TESTING_CHECK( magma_dmalloc( &dBuffer[d], mloc ) );
        }

        double mbytes = ((double)sizeof(double)*m)/1e6;
        if (iam_mpi == 0) {
            printf( "%6d %7.2f ", m, mbytes );
        }
        MPI_Barrier(MPI_COMM_WORLD);
        magma_time = magma_sync_wtime( NULL );
        for( int iter = 0; iter < opts->niter; ++iter ) {
            // all-reduce to GPU0
            for (int d=1; d<num_gpu; d++) {
                magma_setdevice(iam_gpu + d);
                cudaMemcpyAsync( dBuffer[d], dXloc[d], mloc*sizeof(double), cudaMemcpyDeviceToDevice,
                                 magma_queue_get_cuda_stream( queue[d] ) );
                magma_event_record( event[d], queue[d] );
            }
            // sum them on GPU0
            magma_setdevice(iam_gpu);
            magmablasSetKernelStream( queue[0] );
            for (int d=1; d<num_gpu; d++) {
                magma_queue_wait_event( queue[0], event[d] );
                magma_daxpy( mloc, c_one,  dBuffer[d], 1, dXloc[0], 1 );
            }
            // copy to CPU
            magma_dgetvector( mloc, dXloc[0], 1, hXloc, 1 );
            // MPI
            MPI_Allgatherv( hXloc, mloc, MPI_DOUBLE, hX, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD );
            // copy to local GPUs
            for (int d=0; d<num_gpu; d++) {
                magma_setdevice(iam_gpu + d);
                magma_dsetvector_async( m, hX, 1, dX[d], 1, queue[d] );
            }
            for (int d=0; d<num_gpu; d++) {
                magma_setdevice(iam_gpu + d);
                magma_queue_sync( queue[d] );
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        magma_time = magma_sync_wtime( NULL ) - magma_time;

        if (iam_mpi == 0) {
            printf( "    %7.2f ", ((double)opts->niter * mbytes)/(1000.*magma_time) );
            printf( "(%.2f)", (1000.*magma_time)/((double)opts->niter) );
        }
        if (opts->check != 0) {
            double work[1], beta = -(double)num_gpu;
            blasf77_daxpy( &m, &beta, gX, &ione, hX, &ione );
            double local_error = lapackf77_dlange( "F", &m, &ione, hX, &m, work );
            double error;
            MPI_Allreduce(&local_error, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (iam_mpi == 0) {
                printf( ", error=%.2e", error );
            }
        }
        if (iam_mpi == 0) {
            printf( "\n" );
            fflush( stdout );
        }
        for (int d=0; d<num_gpu; d++) {
            magma_setdevice(iam_gpu + d);
            magma_queue_destroy( queue[d] );
            magma_event_destroy( event[d] );
            magma_free( dX[d] );
            magma_free( dXloc[d] );
        }
        magma_setdevice(iam_gpu);
        for (int d=0; d<num_gpu; d++) {
            magma_free( dBuffer[d] );
        }
        free(displs); free(recvcounts);
        //magma_free( dX );
        //magma_free( dXloc );
        magma_free_pinned( hX );
        magma_free_pinned( gX );
    }
}
