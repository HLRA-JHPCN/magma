/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zgemv_batched.cpp, normal z -> d, Sun Nov 20 20:20:38 2016
       @author Mark Gates
       @author Azzam Haidar
       @author Tingxing Dong
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
#include "testings.h"

#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"  // internal header
#endif

#define imax(a, b) ((a) > (b) ? (a) : (b))

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgemv_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf, cpu_time;
    double          error, magma_error, normalize, work[1];
    magma_int_t M, N, Xm, Ym, lda, ldda;
    magma_int_t sizeA, sizeX, sizeY;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;

    double *h_A, *h_X, *h_Y, *h_Ymagma;
    double *d_A, *d_X, *d_Y;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double alpha = MAGMA_D_MAKE(  0.29, -0.86 );
    double beta  = MAGMA_D_MAKE( -0.48,  0.38 );
    double **d_A_array = NULL;
    double **d_X_array = NULL;
    double **d_Y_array = NULL;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;
    batchCount = opts.batchcount;
    
    double *Anorm, *Xnorm, *Ynorm;
    TESTING_CHECK( magma_dmalloc_cpu( &Anorm, batchCount ));
    TESTING_CHECK( magma_dmalloc_cpu( &Xnorm, batchCount ));
    TESTING_CHECK( magma_dmalloc_cpu( &Ynorm, batchCount ));
    
    // See testing_dgemm about tolerance.
    double eps = lapackf77_dlamch("E");
    double tol = 3*eps;
    
    printf("%% trans = %s\n", lapack_trans_const(opts.transA) );
    printf("%% BatchCount     M     N   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error\n");
    printf("%%=============================================================================\n");
#define FROM_SORTED_FILE
#if defined(FROM_SORTED_FILE)
    int num_sizes, nlf;
    FILE *fp = fopen("sizes_sorted.dat","r");
    //FILE *fp = fopen("sizes.dat","r");
    fscanf(fp, "%d %d\n",&num_sizes,&nlf);
    opts.ntest = num_sizes;
    opts.niter = 1;

    real_Double_t tot_flops  = 0;
    real_Double_t avg_gflops = 0;
    real_Double_t tot_times  = 0;
    double start;
    double time0 = magma_sync_wtime( opts.queue );
    double time1 = 0.0;
    double time2 = 0.0;
    double time3 = 0.0;
    double time4 = 0.0;
#endif

#define MALLOC_ONCE
#if defined(MALLOC_ONCE)
    magma_int_t sizeAd_max = 0;
    magma_int_t sizeA_max = 0;
    magma_int_t sizeX_max = 0;
    magma_int_t sizeY_max = 0;
    for ( int itest = 0; itest < opts.ntest; ++itest ) {
        int id;
        fscanf(fp, "%d %d %d\n",&id,&N,&M);
        if ( opts.transA == MagmaTrans ) {
            int tmp = M;
            M = N;
            N = tmp;
        }
        if ( opts.transA == MagmaNoTrans ) {
            Xm = N;
            Ym = M;
        }
        else {
            Xm = M;
            Ym = N;
        }
        lda = M;
        ldda = magma_roundup( M, opts.align );  // multiple of 32 by default
        sizeAd_max = imax(sizeAd_max,ldda*N*batchCount);
        sizeA_max  = imax(sizeA_max, lda*N*batchCount);
        sizeX_max  = imax(sizeX_max, incx*Xm*batchCount);
        sizeY_max  = imax(sizeY_max, incy*Ym*batchCount);
    }

    TESTING_CHECK( magma_dmalloc_cpu( &h_A,  sizeA_max ));
    TESTING_CHECK( magma_dmalloc_cpu( &h_X,  sizeX_max ));
    TESTING_CHECK( magma_dmalloc_cpu( &h_Y,  sizeY_max  ));
    TESTING_CHECK( magma_dmalloc_cpu( &h_Ymagma,  sizeY_max  ));

    TESTING_CHECK( magma_dmalloc( &d_A, sizeAd_max ));
    TESTING_CHECK( magma_dmalloc( &d_X, sizeX_max ));
    TESTING_CHECK( magma_dmalloc( &d_Y, sizeY_max ));

    TESTING_CHECK( magma_malloc( (void**) &d_A_array, batchCount * sizeof(double*) ));
    TESTING_CHECK( magma_malloc( (void**) &d_X_array, batchCount * sizeof(double*) ));
    TESTING_CHECK( magma_malloc( (void**) &d_Y_array, batchCount * sizeof(double*) ));

    /* Initialize the matrices */
    lapackf77_dlarnv( &ione, ISEED, &sizeA_max, h_A );
    lapackf77_dlarnv( &ione, ISEED, &sizeX_max, h_X );
    lapackf77_dlarnv( &ione, ISEED, &sizeY_max, h_Y );
    fclose(fp);

    fp = fopen("sizes_sorted.dat","r");
    //fp = fopen("sizes.dat","r");
    fscanf(fp, "%d %d\n",&num_sizes,&nlf);
#endif
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            start = magma_sync_wtime( opts.queue );
            #if defined(FROM_SORTED_FILE)
            int id;
            fscanf(fp, "%d %d %d\n",&id,&N,&M);
            if ( opts.transA == MagmaTrans ) {
               int tmp = M;
                M = N;
                N = tmp;
            }
            #else
            M = opts.msize[itest];
            N = opts.nsize[itest];
            #endif
            lda    = M;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            gflops = FLOPS_DGEMV( M, N ) / 1e9 * batchCount;

            if ( opts.transA == MagmaNoTrans ) {
                Xm = N;
                Ym = M;
            }
            else {
                Xm = M;
                Ym = N;
            }

            sizeA = lda*N*batchCount;
            sizeX = incx*Xm*batchCount;
            sizeY = incy*Ym*batchCount;

            #if defined(MALLOC_ONCE)
            assert(sizeA <= sizeA_max);
            assert(sizeX <= sizeX_max);
            assert(sizeY <= sizeY_max);
            assert(ldda*N*batchCount <= sizeAd_max);
            /* Initialize the matrices */
            //lapackf77_dlarnv( &ione, ISEED, &sizeA, h_A );
            //lapackf77_dlarnv( &ione, ISEED, &sizeX, h_X );
            lapackf77_dlarnv( &ione, ISEED, &sizeY, h_Y );
            #else
            TESTING_CHECK( magma_dmalloc_cpu( &h_A,  sizeA ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_X,  sizeX ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_Y,  sizeY  ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_Ymagma,  sizeY  ));

            TESTING_CHECK( magma_dmalloc( &d_A, ldda*N*batchCount ));
            TESTING_CHECK( magma_dmalloc( &d_X, sizeX ));
            TESTING_CHECK( magma_dmalloc( &d_Y, sizeY ));

            TESTING_CHECK( magma_malloc( (void**) &d_A_array, batchCount * sizeof(double*) ));
            TESTING_CHECK( magma_malloc( (void**) &d_X_array, batchCount * sizeof(double*) ));
            TESTING_CHECK( magma_malloc( (void**) &d_Y_array, batchCount * sizeof(double*) ));

            /* Initialize the matrices */
            lapackf77_dlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_dlarnv( &ione, ISEED, &sizeX, h_X );
            lapackf77_dlarnv( &ione, ISEED, &sizeY, h_Y );
            #endif
            
            // Compute norms for error
            if ( opts.lapack ) {
                for (int s = 0; s < batchCount; ++s) {
                    Anorm[s] = lapackf77_dlange( "F", &M, &N,     &h_A[s*lda*N],   &lda,  work );
                    Xnorm[s] = lapackf77_dlange( "F", &ione, &Xm, &h_X[s*Xm*incx], &incx, work );
                    Ynorm[s] = lapackf77_dlange( "F", &ione, &Ym, &h_Y[s*Ym*incy], &incy, work );
                }
            }
            time1 += (magma_sync_wtime( opts.queue )-start);
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            start = magma_sync_wtime( opts.queue );
            magma_dsetmatrix( M, N*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_dsetvector( Xm*batchCount, h_X, incx, d_X, incx, opts.queue );
            magma_dsetvector( Ym*batchCount, h_Y, incy, d_Y, incy, opts.queue );
            
            magma_dset_pointer( d_A_array, d_A, ldda, 0, 0, ldda*N, batchCount, opts.queue );
            magma_dset_pointer( d_X_array, d_X, 1, 0, 0, incx*Xm, batchCount, opts.queue );
            magma_dset_pointer( d_Y_array, d_Y, 1, 0, 0, incy*Ym, batchCount, opts.queue );
            time2 += (magma_sync_wtime( opts.queue )-start);

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_dgemv_batched(opts.transA, M, N,
                             alpha, d_A_array, ldda,
                                    d_X_array, incx,
                             beta,  d_Y_array, incy, batchCount, opts.queue);
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            time3 += magma_time;

            start = magma_sync_wtime( opts.queue );
            if ( opts.lapack ) {
                magma_dgetvector( Ym*batchCount, d_Y, incy, h_Ymagma, incy, opts.queue );
            }
            time4 += (magma_sync_wtime( opts.queue )-start);
            #if defined(FROM_SORTED_FILE)
            avg_gflops+= magma_perf;
            tot_flops += gflops;
            tot_times += magma_time;
            #endif

            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (int i=0; i < batchCount; i++)
                {
                    blasf77_dgemv( lapack_trans_const(opts.transA),
                                   &M, &N,
                                   &alpha, h_A + i*lda*N, &lda,
                                           h_X + i*Xm*incx, &incx,
                                   &beta,  h_Y + i*Ym*incy, &incy );
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute error compared lapack
                // error = |dY - Y| / (gamma_{k+2}|A||X| + gamma_2|Yin|); k = Xn
                magma_error = 0;
                
                for (int s=0; s < batchCount; s++){
                    normalize = sqrt(double(Xm+2))*Anorm[s]*Xnorm[s] + 2*Ynorm[s];
                    if (normalize == 0)
                        normalize = 1;
                    blasf77_daxpy( &Ym, &c_neg_one, &h_Y[s*Ym*incy], &incy, &h_Ymagma[s*Ym*incy], &incy );
                    error = lapackf77_dlange( "F", &ione, &Ym, &h_Ymagma[s*Ym*incy], &incy, work )
                          / normalize;
                    magma_error = magma_max_nan( error, magma_error );
                }

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e  %s\n",
                       (long long) batchCount, (long long) M, (long long) N,
                       magma_perf,  1000.*magma_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)     ---   (  ---  )     ---\n",
                       (long long) batchCount, (long long) M, (long long) N,
                       magma_perf,  1000.*magma_time);
            }
            //printf( " %7.2f: %7.2f %7.2f %7.2f %7.2f\n",magma_sync_wtime( opts.queue )-time0,
            //        time1,time2,time3,time4);
            
            #if !defined(MALLOC_ONCE)
            magma_free_cpu( h_A );
            magma_free_cpu( h_X );
            magma_free_cpu( h_Y );
            magma_free_cpu( h_Ymagma );

            magma_free( d_A );
            magma_free( d_X );
            magma_free( d_Y );
            magma_free( d_A_array );
            magma_free( d_X_array );
            magma_free( d_Y_array );
            #endif

            fflush( stdout);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
            
#if defined(MALLOC_ONCE)
    magma_free_cpu( h_A );
    magma_free_cpu( h_X );
    magma_free_cpu( h_Y );
    magma_free_cpu( h_Ymagma );

    magma_free( d_A );
    magma_free( d_X );
    magma_free( d_Y );
    magma_free( d_A_array );
    magma_free( d_X_array );
    magma_free( d_Y_array );
#endif

#if defined(FROM_SORTED_FILE)
    printf( "\n tot_gflops/tot_time = %.2e/%.2e = %.2e, avg_gflop/s=%.2e\n\n",
            tot_flops, tot_times, tot_flops/tot_times, avg_gflops/opts.ntest );
    fclose(fp);
#endif
    magma_free_cpu( Anorm );
    magma_free_cpu( Xnorm );
    magma_free_cpu( Ynorm );

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
