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

// includes, project
#include "testings.h"  // before magma.h, to include cublas_v2
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"

void test1(magma_opts *opts);
void test2(magma_opts *opts);
void test3(magma_opts *opts);

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgemm
*/
int main( int argc, char** argv) {
    MPI_Init( &argc, &argv );
    TESTING_INIT();

    int iam_mpi, num_mpi;
    MPI_Comm_rank( MPI_COMM_WORLD, &iam_mpi );
    MPI_Comm_size( MPI_COMM_WORLD, &num_mpi );
    if (iam_mpi == 0) {
        printf( "%% Usage: %s [options] [-h|--help]\n\n", argv[0] );
        magma_print_environment();
    }

    magma_opts opts;
    parse_opts( argc, argv, &opts );

    switch (opts.version) {
    case 1: // CPU <-> GPU (set/get)
        test1(&opts);
        break;
    case 2: // CPU <-> GPUs (set/get)
        test2(&opts);
        break;
    case 3: // CPU <-> CPU (MPI)
        test3(&opts);
        break;
    }

    TESTING_FINALIZE();
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

        printf( "\n GPU%d\n",d );
        printf( "%% M       MB         Set GB/s (ms)  Get GB/s (ms)\n" );
        printf( "%%================================================\n" );
        fflush( stdout );
        for( int itest = 0; itest < opts->ntest; ++itest ) {
            magma_int_t Xm = opts->msize[itest];
            magma_int_t sizeX = incx*Xm;

            //TESTING_MALLOC_CPU( X,  double, sizeX );
            TESTING_MALLOC_PIN( X,  double, sizeX );
            TESTING_MALLOC_DEV( dX, double, sizeX );

            /* Initialize the matrices */
            lapackf77_dlarnv( &ione, ISEED, &sizeX, X );

            double mbytes = (double)sizeof(double)*Xm/1e6;
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
        TESTING_FREE_PIN( X );
        TESTING_FREE_DEV( dX );
        fflush( stdout );
    }
}

// get/set between CPU and multiple GPUs
void test2(magma_opts *opts) {
    real_Double_t  magma_time;
    magma_int_t ione = 1;
    magma_int_t incx  = 1;
    magma_int_t ISEED[4] = {0,0,0,1};


    int num_gpus = opts->ngpu;
    magma_queue_t queue[MagmaMaxGPUs];
    printf( "\n GPU" );
    for (int d=0; d<num_gpus; d++) {
        magma_setdevice(d);
        magma_queue_create( &queue[d] );
        printf( ",%d",d );
    }
    printf( "\n" );

    printf( "%% M       MB         Set GB/s (ms)  Get GB/s (ms)\n" );
    printf( "%%================================================\n" );
    fflush( stdout );
    for( int itest = 0; itest < opts->ntest; ++itest ) {
        double *X[MagmaMaxGPUs];
        magmaDouble_ptr dX[MagmaMaxGPUs];

        magma_int_t Xm = opts->msize[itest];
        magma_int_t sizeX = incx*Xm;

        for (int d=0; d<num_gpus; d++) {
            TESTING_MALLOC_PIN( X[d],  double, sizeX );
            TESTING_MALLOC_DEV( dX[d], double, sizeX );

            /* Initialize the matrices */
            lapackf77_dlarnv( &ione, ISEED, &sizeX, X[d] );
        }

        double mbytes = (double)sizeof(double)*Xm/1e6;
        printf( "%6d %7.2f ", Xm, mbytes );
        magma_time = magma_sync_wtime( NULL );
        for( int iter = 0; iter < opts->niter; ++iter ) {
            for (int d=0; d<num_gpus; d++) {
                magma_setdevice(d);
                magma_dsetvector_async( Xm, X[d], incx, dX[d], incx, queue[d] );
            }
        }
        magma_time = magma_sync_wtime( NULL ) - magma_time;
        printf( "    %7.2f ", ((double)opts->niter * mbytes)/(1000.*magma_time) );
        printf( "(%.2f)", (1000.*magma_time)/((double)opts->niter) );

        magma_time = magma_sync_wtime( NULL );
        for( int iter = 0; iter < opts->niter; ++iter ) {
            for (int d=0; d<num_gpus; d++) {
                magma_setdevice(d);
                magma_dgetvector_async( Xm, dX[d], incx, X[d], incx, queue[d] );
            }
        }
        magma_time = magma_sync_wtime( NULL ) - magma_time;
        printf( "  %7.2f ", ((double)opts->niter * mbytes)/(1000.*magma_time) );
        printf( "(%.2f)\n", (1000.*magma_time)/((double)opts->niter) );

        for (int d=0; d<num_gpus; d++) {
            TESTING_FREE_PIN( X[d] );
            TESTING_FREE_DEV( dX[d] );
        }
        fflush( stdout );
    }
    for (int d=0; d<num_gpus; d++) {
        magma_setdevice(d);
        magma_queue_destroy( queue[d] );
    }
}

// MPI-comm between CPUs
void test3(magma_opts *opts) {
    real_Double_t  magma_time;
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    int iam_mpi, num_mpi;
    MPI_Comm_rank( MPI_COMM_WORLD, &iam_mpi );
    MPI_Comm_size( MPI_COMM_WORLD, &num_mpi );

    if (iam_mpi == 0) {
        printf( "%% M       MB         Set GB/s (ms)  Get GB/s (ms)\n" );
        printf( "%%================================================\n" );
        fflush( stdout );
    }
    for( int itest = 0; itest < opts->ntest; ++itest ) {
        double *X, *Xloc;

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
        TESTING_MALLOC_CPU( X,  double, m );
        TESTING_MALLOC_CPU( Xloc,  double, mloc );
        lapackf77_dlarnv( &ione, ISEED, &mloc, Xloc );

        double mbytes = (double)sizeof(double)*m/1e6;
        if (iam_mpi == 0) {
            printf( "%6d %7.2f ", m, mbytes );
        }
        magma_time = magma_sync_wtime( NULL );
        MPI_Allgatherv( Xloc, mloc, MPI_DOUBLE, X, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD );
        magma_time = magma_sync_wtime( NULL ) - magma_time;
        if (iam_mpi == 0) {
            printf( "    %7.2f ", ((double)opts->niter * mbytes)/(1000.*magma_time) );
            printf( "(%.2f)\n", (1000.*magma_time)/((double)opts->niter) );
            fflush( stdout );
        }
        free(displs); free(recvcounts);
        TESTING_FREE_CPU( X );
        TESTING_FREE_CPU( Xloc );
    }
}
