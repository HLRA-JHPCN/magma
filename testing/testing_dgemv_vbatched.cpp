/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from testing/testing_zgemv_vbatched.cpp, normal z -> d, Wed Dec  7 22:45:47 2016
       @author Mark Gates
       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah
       
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

#ifdef SORT_SIZES
#define sort_array_size 4
#define sort_group_size 8

int hacapk_size_sorter(const void* arg1,const void* arg2) {
  const int *val1 = (const int*)arg1;
  const int *val2 = (const int*)arg2;

  //#define BY_GROUP
  #define BY_M
  //#define BY_N
  #if defined(BY_GROUP)
  // sort by n "group", whithin group, sort by m
  return (val2[3] == val1[3] ? (val2[1] < val1[1]) : val2[3] < val1[3]);
  #elif defined(BY_M)
  // sort by m
  return (val2[1] < val1[1]);
  #elif defined(BY_N)
  // sort by n
  return (val2[2] < val1[2]);
  #endif
}
#endif

#define imax(a, b) ((a) > (b) ? (a) : (b))

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgemv_vbatched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf, cpu_time;
    double          error, magma_error, normalize, work[1];
    magma_int_t M, N;
    magma_int_t *Xn, *Yn;
    magma_int_t total_size_A_cpu = 0, total_size_X = 0, total_size_Y = 0;
    magma_int_t total_size_A_dev = 0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;
    magma_int_t max_M, max_N;

    double *h_A, *h_X, *h_Y, *h_Ymagma;
    double *d_A, *d_X, *d_Y;
    double **h_A_array = NULL;
    double **h_X_array = NULL;
    double **h_Y_array = NULL;
    double **d_A_array = NULL;
    double **d_X_array = NULL;
    double **d_Y_array = NULL;
    double *h_A_tmp, *h_X_tmp, *h_Y_tmp, *h_Ymagma_tmp;
    magma_int_t *h_M, *h_N; // hold the sizes on cpu
    magma_int_t *d_M, *d_N; // hold the sizes on gpu
    magma_int_t *h_lda, *h_ldda, *d_ldda;
    magma_int_t *h_incx, *d_incx;
    magma_int_t *h_incy, *d_incy;
    magma_int_t max_inc = 1;
    
    double c_neg_one = MAGMA_D_NEG_ONE;
    double alpha = MAGMA_D_MAKE(  1.00, 0.00 ); //MAGMA_D_MAKE(  0.29, -0.86 );
    double beta  = MAGMA_D_MAKE(  1.00, 0.00 ); //MAGMA_D_MAKE( -0.48,  0.38 );
    
    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;
    batchCount = opts.batchcount;
    
    // allocate space for the sizes/leading dim.
    TESTING_CHECK( magma_imalloc_pinned(&h_M, batchCount) );
    TESTING_CHECK( magma_imalloc_pinned(&h_N, batchCount) );
    TESTING_CHECK( magma_imalloc_pinned(&h_ldda, batchCount) );
    TESTING_CHECK( magma_imalloc_pinned(&h_incx, batchCount) );
    TESTING_CHECK( magma_imalloc_pinned(&h_incy, batchCount) );
    
    TESTING_CHECK( magma_imalloc(&d_M, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_N, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_ldda, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_incx, batchCount+1) );
    TESTING_CHECK( magma_imalloc(&d_incy, batchCount+1) );
    
    double *Anorm, *Xnorm, *Ynorm;
    TESTING_CHECK( magma_dmalloc_cpu( &Anorm, batchCount ));
    TESTING_CHECK( magma_dmalloc_cpu( &Xnorm, batchCount ));
    TESTING_CHECK( magma_dmalloc_cpu( &Ynorm, batchCount ));
    
    TESTING_CHECK( magma_malloc_cpu((void**)&h_A_array, batchCount*sizeof(double*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&h_X_array, batchCount*sizeof(double*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&h_Y_array, batchCount*sizeof(double*)) );
    TESTING_CHECK( magma_malloc((void**)&d_A_array, batchCount*sizeof(double*)) );
    TESTING_CHECK( magma_malloc((void**)&d_X_array, batchCount*sizeof(double*)) );
    TESTING_CHECK( magma_malloc((void**)&d_Y_array, batchCount*sizeof(double*)) );
    
    // See testing_dgemm about tolerance.
    double eps = lapackf77_dlamch("E");
    double tol = 3*eps;
    
    // queues
    magma_device_t cdev;
    magma_getdevice( &cdev );

    int num_queues = opts.nqueue;
    magma_queue_t *queues = (magma_queue_t*)malloc(num_queues * sizeof(magma_queue_t));
    for (int i=0; i<num_queues; i++) {
        magma_queue_create( cdev, &queues[i] );
    }

    printf("%% If running lapack (option --lapack), MAGMA error is computed\n"
           "%% relative to CPU BLAS result.\n\n"
           "%% transA = %s\n",
           lapack_trans_const(opts.transA));
    
    printf("%%              max   max\n");
    printf("%% BatchCount     M     N   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error\n");
    printf("%%=============================================================================\n");
#define FROM_SORTED_FILE
#if defined(FROM_FILE)
    int batchCount_0 = batchCount;
    int num_sizes = 311808;
    FILE *fp = fopen("sizes.dat","r");
    #if defined(SORT_SIZES)
    int *sizes = (int*)malloc( sort_array_size*num_sizes * sizeof(int) );
    for (int i=0; i<num_sizes; i++) {
        sizes[sort_array_size * i] = i; // id

        int id, m, n;
        fscanf(fp, "%d %d %d\n",&id,&m,&n);
        if ( opts.transA == MagmaNoTrans ) {
            int tmp = m;
            m = n;
            n = tmp;
        }

        sizes[sort_array_size*i + 1] = m;
        sizes[sort_array_size*i + 2] = n;
        #if defined(BY_M)
        sizes[sort_array_size*i + 3] = (m-1) / sort_group_size;
        #else
        sizes[sort_array_size*i + 3] = (n-1) / sort_group_size;
        #endif
    }
    fclose(fp);
    qsort( sizes, num_sizes, sort_array_size*sizeof(int), hacapk_size_sorter );
    fp = fopen("sizes_sorted.dat","w");
    for (int i=0; i<num_sizes; i++) {
        fprintf(fp, "%d %d %d\n",sizes[sort_array_size*i], sizes[sort_array_size*i+1], sizes[sort_array_size*i+2] );
    }
    fclose(fp);
    #endif
#elif defined(FROM_SORTED_FILE)
    int batchCount_0 = batchCount;
    int num_sizes, nlf;
    FILE *fp = fopen("sizes_sorted.dat","r");
    //FILE *fp = fopen("sizes.dat","r");
    fscanf(fp, "%d %d\n",&num_sizes,&nlf);
    int ntest1 = magma_ceildiv(nlf, batchCount);
    int ntest2 = magma_ceildiv(num_sizes-nlf, batchCount);
    opts.ntest = ntest1+ntest2;
    opts.niter = 1;
printf( " ntest=%d+%d\n",ntest1,ntest2);

//#define MALLOC_ONCE
#if defined(MALLOC_ONCE)
    magma_int_t sizeAd_max = 0;
    magma_int_t sizeA_max = 0;
    magma_int_t sizeX_max = 0;
    magma_int_t sizeY_max = 0;
    for ( int itest = 0; itest < opts.ntest; ++itest ) {
        if (itest < ntest1) {
            batchCount = min(batchCount_0, nlf - batchCount_0*itest);
        } else {
            batchCount = min(batchCount_0, (num_sizes-nlf) - batchCount_0*(itest-ntest1));
        }
        int sizeAd_batch = 0;
        int sizeA_batch  = 0;
        int sizeX_batch  = 0;
        int sizeY_batch  = 0;
        for (int i = 0; i < batchCount; i++) {
            int id;
            fscanf(fp, "%d %d %d\n",&id,&N,&M);
            int lda = M;
            int ldda = magma_roundup( M, opts.align );  // multiple of 32 by default

            sizeA_batch  += N * lda;
            sizeAd_batch += N * ldda;
                
            sizeX_batch += N;
            sizeY_batch += M;
        }
        sizeAd_max = imax(sizeAd_max, sizeAd_batch);
        sizeA_max  = imax(sizeA_max, sizeA_batch);
        sizeX_max  = imax(sizeX_max, sizeX_batch);
        sizeY_max  = imax(sizeY_max, sizeY_batch);
    }

    TESTING_CHECK( magma_dmalloc_cpu( &h_A,  sizeA_max ));
    TESTING_CHECK( magma_dmalloc_cpu( &h_X,  sizeX_max ));
    TESTING_CHECK( magma_dmalloc_cpu( &h_Y,  sizeY_max  ));
    TESTING_CHECK( magma_dmalloc_cpu( &h_Ymagma,  sizeY_max  ));

    TESTING_CHECK( magma_dmalloc( &d_A, sizeAd_max ));
    TESTING_CHECK( magma_dmalloc( &d_X, sizeX_max ));
    TESTING_CHECK( magma_dmalloc( &d_Y, sizeY_max ));

    TESTING_CHECK( magma_malloc( (void**) &d_A_array, batchCount_0 * sizeof(double*) ));
    TESTING_CHECK( magma_malloc( (void**) &d_X_array, batchCount_0 * sizeof(double*) ));
    TESTING_CHECK( magma_malloc( (void**) &d_Y_array, batchCount_0 * sizeof(double*) ));

    /* Initialize the matrices */
    lapackf77_dlarnv( &ione, ISEED, &sizeA_max, h_A );
    lapackf77_dlarnv( &ione, ISEED, &sizeX_max, h_X );
    lapackf77_dlarnv( &ione, ISEED, &sizeY_max, h_Y );
    fclose(fp);

    fp = fopen("sizes_sorted.dat","r");
    //fp = fopen("sizes.dat","r");
    fscanf(fp, "%d %d\n",&num_sizes,&nlf);
#endif
#endif
    double total_gpu = 0.0, total_cpu = 0.0;
    double total_flop = 0.0;
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            
            if ( opts.transA == MagmaNoTrans ) {
                Xn = h_N;
                Yn = h_M;
            }
            else {
                Xn = h_M;
                Yn = h_N;
            }
            h_lda = h_M;
            
            // guarantee reproducible sizes
            srand(1000);
            
            gflops = 0;
            max_M = max_N = 0;
            total_size_A_cpu = total_size_A_dev = 0;
            total_size_X = total_size_Y = 0;
#if defined(FROM_FILE) | defined(FROM_SORTED_FILE)
            if (itest < ntest1) {
                batchCount = min(batchCount_0, nlf - batchCount_0*itest);
            } else {
                batchCount = min(batchCount_0, (num_sizes-nlf) - batchCount_0*(itest-ntest1));
            }
#endif
            #define VERSION 3 // 1 streamed, 2 batched nocheck, 3 batched check
            for (int i = 0; i < batchCount; i++) {
#if defined(FROM_FILE) | defined(FROM_SORTED_FILE)
                #if defined(SORT_SIZES)
                h_M[i] = sizes[sort_array_size*(batchCount_0*iter+i) + 1];
                h_N[i] = sizes[sort_array_size*(batchCount_0*iter+i) + 2];
                #elif defined(FROM_SORTED_FILE)
                int id, m, n;
                fscanf(fp, "%d %d %d\n",&id,&n,&m);
                if ( opts.transA == MagmaTrans ) {
                   int tmp = m;
                    m = n;
                    n = tmp;
                }
                h_M[i] = m;
                h_N[i] = n;
                #else
                int id, m, n;
                fscanf(fp, "%d %d %d\n",&id,&m,&n);
                if ( opts.transA == MagmaNoTrans ) {
                   int tmp = m;
                    m = n;
                    n = tmp;
                }
                h_M[i] = m;
                h_N[i] = n;
                #endif
#else
                h_M[i] = 1 + (rand() % M);
                h_N[i] = 1 + (rand() % N);
#endif

                h_incx[i] = 1 + (rand() % max_inc);
                h_incy[i] = 1 + (rand() % max_inc);
                
                max_M = max( max_M, h_M[i] );
                max_N = max( max_N, h_N[i] );
                
                h_ldda[i] = magma_roundup( h_lda[i], opts.align );  // multiple of 32 by default
                
                #if VERSION!=3
                total_size_A_cpu += h_N[i] * h_lda[i];
                total_size_A_dev += h_N[i] * h_ldda[i];
                
                total_size_X += Xn[i] * h_incx[i];
                total_size_Y += Yn[i] * h_incy[i];
                
                gflops += FLOPS_DGEMV( h_M[i], h_N[i]) / 1e9;
                #endif
            }
            
            #if VERSION==3
            for (int i = 0; i < batchCount; i++) {
                h_M[i] = max_M;
                h_N[i] = max_N;
                h_ldda[i] = magma_roundup( h_lda[i], opts.align );  // multiple of 32 by default

                total_size_A_cpu += h_N[i] * h_lda[i];
                total_size_A_dev += h_N[i] * h_ldda[i];

                total_size_X += Xn[i] * h_incx[i];
                total_size_Y += Yn[i] * h_incy[i];

                gflops += FLOPS_DGEMV( h_M[i], h_N[i]) / 1e9;
            }
            #endif

            #if defined(MALLOC_ONCE)
            assert(total_size_A_dev <= sizeAd_max);
            assert(total_size_A_cpu <= sizeA_max);
            assert(total_size_X <= sizeX_max);
            assert(total_size_Y <= sizeY_max);
            /* Initialize the matrices */
            lapackf77_dlarnv( &ione, ISEED, &total_size_Y, h_Y );
            #else
            TESTING_CHECK( magma_dmalloc_pinned(&h_A, total_size_A_cpu) );
            TESTING_CHECK( magma_dmalloc_pinned(&h_X,   total_size_X) );
            TESTING_CHECK( magma_dmalloc_pinned(&h_Y,   total_size_Y) );
            TESTING_CHECK( magma_dmalloc_pinned(&h_Ymagma, total_size_Y) );
            
            TESTING_CHECK( magma_dmalloc(&d_A, total_size_A_dev) );
            TESTING_CHECK( magma_dmalloc(&d_X, total_size_X) );
            TESTING_CHECK( magma_dmalloc(&d_Y, total_size_Y) );

            /* Initialize the matrices */
            lapackf77_dlarnv( &ione, ISEED, &total_size_A_cpu, h_A );
            lapackf77_dlarnv( &ione, ISEED, &total_size_X, h_X );
            lapackf77_dlarnv( &ione, ISEED, &total_size_Y, h_Y );
            #endif            

            // Compute norms for error
            h_A_tmp = h_A;
            h_X_tmp = h_X;
            h_Y_tmp = h_Y;
            for (int s = 0; s < batchCount; ++s) {
                Anorm[s] = lapackf77_dlange( "F", &h_M[s], &h_N[s], h_A_tmp, &h_lda[s],  work );
                Xnorm[s] = lapackf77_dlange( "F", &ione,   &Xn[s],  h_X_tmp, &h_incx[s], work );
                Ynorm[s] = lapackf77_dlange( "F", &ione,   &Yn[s],  h_Y_tmp, &h_incy[s], work );
                h_A_tmp += h_N[s] * h_lda[s];
                h_X_tmp += Xn[s] * h_incx[s];
                h_Y_tmp += Yn[s] * h_incy[s];
            }
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_setvector(batchCount, sizeof(magma_int_t), h_M, 1, d_M, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), h_N, 1, d_N, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), h_ldda, 1, d_ldda, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), h_incx, 1, d_incx, 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), h_incy, 1, d_incy, 1, opts.queue );
            
            h_A_array[0] = d_A;
            h_X_array[0] = d_X;
            h_Y_array[0] = d_Y;
            for (int i = 1; i < batchCount; i++) {
                h_A_array[i] = h_A_array[i-1] + h_N[i-1] * h_ldda[i-1];
                h_X_array[i] = h_X_array[i-1] + Xn[i-1] * h_incx[i-1];
                h_Y_array[i] = h_Y_array[i-1] + Yn[i-1] * h_incy[i-1];
            }
            magma_setvector(batchCount, sizeof(double*), h_A_array, 1, d_A_array, 1, opts.queue );
            magma_setvector(batchCount, sizeof(double*), h_X_array, 1, d_X_array, 1, opts.queue );
            magma_setvector(batchCount, sizeof(double*), h_Y_array, 1, d_Y_array, 1, opts.queue );
            
            h_A_tmp = h_A;
            for (int i = 0; i < batchCount; i++) {
                magma_dsetmatrix( h_M[i], h_N[i], h_A_tmp, h_lda[i], h_A_array[i], h_ldda[i], opts.queue );
                h_A_tmp += h_N[i] * h_lda[i];
            }
            magma_dsetvector( total_size_X, h_X, 1, d_X, 1, opts.queue );
            magma_dsetvector( total_size_Y, h_Y, 1, d_Y, 1, opts.queue );
            
            magma_time = magma_sync_wtime( opts.queue );
            #if VERSION==1
            for (int i = 0; i < batchCount; i++) {
                magma_dgemv( opts.transA, h_M[i], h_N[i],
                             alpha, h_A_array[i], h_ldda[i],
                                    h_X_array[i], h_incx[i],
                             beta,  h_Y_array[i], h_incy[i], queues[i%num_queues] );
            }
            #elif VERSION==2
            magmablas_dgemv_vbatched_max_nocheck(opts.transA, d_M, d_N,
                             alpha, d_A_array, d_ldda,
                                    d_X_array, d_incx,
                             beta,  d_Y_array, d_incy,
                             batchCount,
                             max_M, max_N, opts.queue);
            #elif VERSION==3
            magmablas_dgemv_batched(opts.transA, 
                             max_M, max_N,
                             alpha, d_A_array, h_ldda[0],
                                    d_X_array, h_incx[0],
                             beta,  d_Y_array, h_incy[0],
                             batchCount, opts.queue);
            #else
            magmablas_dgemv_vbatched(opts.transA, d_M, d_N,
                             alpha, d_A_array, d_ldda,
                                    d_X_array, d_incx,
                             beta,  d_Y_array, d_incy,
                             batchCount, opts.queue);
            #endif
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            total_gpu += magma_time;
            total_flop += gflops;
            
            if ( opts.lapack ) {
                magma_dgetvector(total_size_Y, d_Y, 1, h_Ymagma, 1, opts.queue );
            }

            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                // displace pointers for the cpu, reuse h_A_array
                h_A_array[0] = h_A;
                h_X_array[0] = h_X;
                h_Y_array[0] = h_Y;
                for (int i = 1; i < batchCount; i++) {
                    h_A_array[i] = h_A_array[i-1] + h_N[i-1] * h_lda[i-1];
                    h_X_array[i] = h_X_array[i-1] + Xn[i-1] * h_incx[i-1];
                    h_Y_array[i] = h_Y_array[i-1] + Yn[i-1] * h_incy[i-1];
                }
                cpu_time = magma_wtime();
                //#define BATCHED_DISABLE_PARCPU
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (magma_int_t s=0; s < batchCount; s++)
                {
                    blasf77_dgemv( lapack_trans_const(opts.transA),
                                   &h_M[s], &h_N[s],
                                   &alpha, h_A_array[s], &h_lda[s],
                                           h_X_array[s], &h_incx[s],
                                   &beta,  h_Y_array[s], &h_incy[s] );
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                total_cpu += cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute error compared lapack
                // error = |dY - Y| / (gamma_{k+2}|A||X| + gamma_2|Yin|); k = Xn
                magma_error = 0;
                
                h_Y_tmp = h_Y;
                h_Ymagma_tmp = h_Ymagma;
                for (int s=0; s < batchCount; s++){
                    normalize = sqrt(double(Xn[s]+2))*Anorm[s]*Xnorm[s] + 2*Ynorm[s];
                    if (normalize == 0)
                        normalize = 1;
                    blasf77_daxpy( &Yn[s], &c_neg_one, h_Y_tmp, &h_incy[s], h_Ymagma_tmp, &h_incy[s] );
                    error = lapackf77_dlange( "F", &ione, &Yn[s], h_Ymagma_tmp, &h_incy[s], work )
                          / normalize;
                    magma_error = magma_max_nan( error, magma_error );

                    h_Y_tmp      += Yn[s] * h_incy[s];
                    h_Ymagma_tmp += Yn[s] * h_incy[s];
                }

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e  %s\n",
                       (long long) batchCount, (long long) max_M, (long long) max_N,
                       magma_perf,  1000.*magma_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)     ---   (  ---  )     ---\n",
                       (long long) batchCount, (long long) max_M, (long long) max_N,
                       magma_perf,  1000.*magma_time);
            }
            
            #if !defined(MALLOC_ONCE)
            magma_free_pinned( h_A );
            magma_free_pinned( h_X );
            magma_free_pinned( h_Y );
            magma_free_pinned( h_Ymagma );

            magma_free( d_A );
            magma_free( d_X );
            magma_free( d_Y );
            #endif            

            fflush( stdout);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    printf( "\n Total time: %.2e seconds (%.2f Gflop/s) on a GPU, %.2e seconds (%.2f Gflop/s) with CPUs\n\n",
            total_gpu,total_flop/total_gpu, total_cpu,total_flop/total_cpu );
#if defined(FROM_FILE) | defined(FROM_SORTED_FILE)
    #if defined(SORT_SIZES)
    free(sizes);
    #else
    fclose(fp);
    #endif
#endif
    // free resources
    for (int i=0; i<num_queues; i++) {
        magma_queue_destroy( queues[i] );
    }
    free(queues);
    #if !defined(MALLOC_ONCE)
    magma_free_pinned( h_A );
    magma_free_pinned( h_X );
    magma_free_pinned( h_Y );
    magma_free_pinned( h_Ymagma );

    magma_free( d_A );
    magma_free( d_X );
    magma_free( d_Y );
    #endif            

    magma_free_pinned( h_M );
    magma_free_pinned( h_N );
    magma_free_pinned( h_ldda );
    magma_free_pinned( h_incx );
    magma_free_pinned( h_incy );

    magma_free_cpu( Anorm );
    magma_free_cpu( Xnorm );
    magma_free_cpu( Ynorm );

    magma_free_cpu( h_A_array );
    magma_free_cpu( h_X_array );
    magma_free_cpu( h_Y_array );
    
    magma_free( d_M );
    magma_free( d_N );
    magma_free( d_ldda );
    magma_free( d_incx );
    magma_free( d_incy );
    magma_free( d_A_array );
    magma_free( d_X_array );
    magma_free( d_Y_array );
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
