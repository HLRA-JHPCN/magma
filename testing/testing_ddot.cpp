/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zgemv.cpp, normal z -> d, Sun Sep 10 23:29:50 2017
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgemv
*/
int main(int argc, char **argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, dev_perf, dev_time;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t N, n;
    double c_zero = MAGMA_D_ZERO;
    double c_one  = MAGMA_D_ONE;
    double *X, *Y, *Alpha[ MagmaMaxGPUs ];
    magmaDouble_ptr dX[ MagmaMaxGPUs ], dY[ MagmaMaxGPUs ], dAlpha[ MagmaMaxGPUs ];
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    int check = opts.check;
    int ngpu  = opts.ngpu;
    magma_queue_t queue[ MagmaMaxGPUs ];
    for( int d = 0; d < ngpu; d++ ) {
        magma_queue_create( d, &queue[d] );
    }
    switch (opts.version) {
    case 1: printf( " DGEMM" ); break;
    case 2: printf( " DGEMV" ); break;
    case 3: printf( " Ddot" );  break;
    }
    printf( " with %d GPUs\n",ngpu );
    printf("%%   N            Gflop/s (ms)\n");
    printf("%%============================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            n = (N+ngpu-1)/ngpu;

            TESTING_CHECK( magma_dmalloc_cpu( &X, N ));
            TESTING_CHECK( magma_dmalloc_cpu( &Y, N ));
            
            for (int d=0; d<ngpu; d++) {
                magma_setdevice(d);
                TESTING_CHECK( magma_dmalloc( &dX[d], N ));
                TESTING_CHECK( magma_dmalloc( &dY[d], N ));
                TESTING_CHECK( magma_dmalloc( &dAlpha[d], 1 ));

                TESTING_CHECK( magma_dmalloc_cpu( &Alpha[d], 1 ));
            }

            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &N, X );
            lapackf77_dlarnv( &ione, ISEED, &N, Y );

            gflops = FLOPS_DGEMV( 1, N ) / 1e9;
            /* =====================================================================
               Performs operation using cuBLAS / clBLAS
               =================================================================== */
            for (int d=0; d<ngpu; d++) {
                magma_setdevice(d);
                magma_dsetvector( N, X, ione, dX[d], ione, queue[d] );
                magma_dsetvector( N, Y, ione, dY[d], ione, queue[d] );
            }

            dev_time = magma_wtime();
            int offset = 0;
            for (int d=0; d<ngpu; d++) {
                int nloc = (d < ngpu-1 ? n : N-d*n);
                magma_setdevice(d);

                switch (opts.version) {
                case 1:
                    magma_dgemm( MagmaTrans, MagmaNoTrans,
                                 1, 1, nloc,
                                 c_one,  &(dX[d][offset]), N,
                                         &(dY[d][offset]), N,
                                 c_zero, dAlpha[d], ione,
                                 queue[d] );
                    if (check != 0) {
                        magma_dgetvector( ione, dAlpha[d], ione, Alpha[d], ione, queue[d] );
                    }
                    break;
                case 2:
                    #if 0
                    magma_dgemv( MagmaTrans, nloc, 1,
                                 c_one,  &(dX[d][offset]), N,
                                         &(dY[d][offset]), ione,
                                 c_zero, dAlpha[d], ione,
                                 queue[d] );
                    #else
                    magmablas_dgemv( MagmaTrans, nloc, 1,
                                     c_one,  &(dX[d][offset]), N,
                                             &(dY[d][offset]), ione,
                                     c_zero, dAlpha[d], ione,
                                     queue[d] );
                    #endif
                    if (check != 0) {
                        magma_dgetvector( ione, dAlpha[d], ione, Alpha[d], ione, queue[d] );
                    }
                    break;
                case 3:
                    cublasHandle_t handle = magma_queue_get_cublas_handle( queue[d] );
                    if (check != 0) {
                        cublasDdot( handle, nloc, 
                                    &(dX[d][offset]), ione, &(dY[d][offset]), ione, Alpha[d] );
                    } else {
                        cublasSetPointerMode( handle, CUBLAS_POINTER_MODE_DEVICE );
                        cublasDdot( handle, nloc, 
                                    &(dX[d][offset]), ione, &(dY[d][offset]), ione, dAlpha[d] );
                        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
                    }
                    break;
                }

                offset += nloc;
            }
            for (int d=0; d<ngpu; d++) {
                magma_setdevice(d);
                magma_queue_sync(queue[d]);
            }
            dev_time = magma_wtime() - dev_time;
            dev_perf = gflops / dev_time;
            
            printf("%5lld (%d)  %7.2f (%7.2f)",
                   (long long) N, n, dev_perf, 1000.*dev_time);
            if (check != 0) {
                double alpha;
                blasf77_dgemv("T", &N, &ione,
                              &c_one,  X, &N,
                                       Y, &ione,
                              &c_zero, &alpha, &ione);
                for (int d=0; d<ngpu; d++) {
                    alpha -= Alpha[d][0];
                }
                printf( " error=%.2e",alpha );
            }
            printf( "\n" );
            
            magma_free_cpu( X );
            magma_free_cpu( Y );
            
            for (int d = 0; d <ngpu; d++) {
                magma_setdevice(d);
                magma_free( dX[d] );
                magma_free( dY[d] );
                magma_free( dAlpha[d] );
                magma_free_cpu( Alpha[d] );
            }
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    for( int d = 0; d < ngpu; ++d ) {
        magma_queue_destroy( queue[d] );
    }    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
