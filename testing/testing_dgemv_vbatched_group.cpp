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

#define imax(a, b) ((a) > (b) ? (a) : (b))
#define iabs(a) ((a) < 0.0 ? -(a) : (a))


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
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;
    magma_int_t *max_M, *max_N;
    magma_int_t *min_M, *min_N;

    double *h_A, *h_X, *h_Y, *h_Ymagma;
    double *d_A, *d_X, *d_Y;
    double **h_A_array = NULL;
    double **h_X_array = NULL;
    double **h_Y_array = NULL;
    double **d_A_array = NULL;
    double **d_X_array = NULL;
    double **d_Y_array = NULL;
    double *h_A_tmp, *h_X_tmp, *h_Y_tmp, *h_Ymagma_tmp;

    magma_int_t *sizes_A, *sizes_X, *sizes_Y;
    magma_int_t *h_batch_count;
    magma_int_t *h_M, *h_N; // hold the sizes on cpu
    magma_int_t *d_M, *d_N; // hold the sizes on gpu
    magma_int_t *d_ldda;
    magma_int_t *h_incx, *d_incx;
    magma_int_t *h_incy, *d_incy;
    magma_int_t max_inc = 1;
    
    double zero = 0.0;
    double one  = 1.0;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double alpha = MAGMA_D_MAKE(  1.00, 0.00 ); //MAGMA_D_MAKE(  0.29, -0.86 );
    double beta  = MAGMA_D_MAKE(  1.00, 0.00 ); //MAGMA_D_MAKE( -0.48,  0.38 );
    
    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;
    batchCount = opts.batchcount;
    int batchCount_0 = batchCount;
    batchCount = 2*batchCount; // to adjust later..    

    int num_sizes, nlf;
    FILE *fp = fopen("sizes_sorted.dat","r");
    fscanf(fp, "%d %d\n",&num_sizes,&nlf);
    int ntest1 = magma_ceildiv(nlf, batchCount_0);
    int ntest2 = magma_ceildiv(num_sizes-nlf, batchCount_0);
    opts.ntest = ntest1+ntest2;
    printf( " ntest=%d+%d (%d+%d)\n",ntest1,ntest2,nlf,num_sizes);

    // ------------------------------ //
    // find more balanced batch sizes //
    // ------------------------------ //
    int max_M_global = 0;
    for ( int itest = 0; itest < num_sizes; itest++) {
        int id, m, n;
        fscanf(fp, "%d %d %d\n",&id,&n,&m);
        max_M_global = max(max_M_global, m);
    }
    int m_count = 1 + ((max_M_global+31) / 32);
    printf( " max_M_global=%d, m_count=%d\n",max_M_global,m_count );
    fclose(fp);

    // reopen file
    fp = fopen("sizes_sorted.dat","r");
    fscanf(fp, "%d %d\n",&num_sizes,&nlf);

    // count GEMVs in each interval (first type)
    int *batch_sizes1 = (int*)calloc(m_count, sizeof(int));
    for ( int itest = 0; itest < nlf; itest++) {
        int id, m, n;
        fscanf(fp, "%d %d %d\n",&id,&n,&m);
        if (m <= 8) {
            batch_sizes1[0] ++;
        } else {
            batch_sizes1[1+((m-1)/32)] ++;
        }
    }
    // adjust batch counts
    for (int i = 0; i < m_count; i++) {
        int count = (batch_sizes1[i]+batchCount_0-1)/batchCount_0;
        if (count > 1) {
            if (batch_sizes1[i]%batchCount_0 < batchCount_0/2) {
                batch_sizes1[i] = (batch_sizes1[i]+count-2)/(count-1);
            } else {
                batch_sizes1[i] = (batch_sizes1[i]+count-1)/count;
            }
        }
        if (batch_sizes1[i] > 0) printf( " batch_sizes1[%d]=%d\n",i,batch_sizes1[i]);
    }

    // count GEMVs in each interval (second type)
    int *batch_sizes2 = (int*)calloc(m_count, sizeof(int));
    for ( int itest = nlf; itest < num_sizes; itest++) {
        int id, m, n;
        fscanf(fp, "%d %d %d\n",&id,&n,&m);
        if (m <= 8) {
            batch_sizes2[0] ++;
        } else {
            batch_sizes2[1+((m-1)/32)] ++;
        }
    }
    printf( "\n" );
    // adjust batch counts
    for (int i = 0; i < m_count; i++) {
        int count = (batch_sizes2[i]+batchCount_0-1)/batchCount_0;
        if (count > 1) {
            if (batch_sizes2[i]%batchCount_0 < batchCount_0/2) {
                batch_sizes2[i] = (batch_sizes2[i]+count-2)/(count-1);
            } else {
                batch_sizes2[i] = (batch_sizes2[i]+count-1)/count;
            }
        }
        if (batch_sizes2[i] > 0) printf( " batch_sizes2[%d]=%d\n",i,batch_sizes2[i]);
    }
    fclose(fp);

    // --------------------------------------------- //
    // find total batch count, required memory, etc //
    // -------------------------------------------- //
    TESTING_CHECK( magma_imalloc_cpu(&h_M, batchCount+1) );
    TESTING_CHECK( magma_imalloc_cpu(&h_N, batchCount+1) );
    TESTING_CHECK( magma_imalloc_cpu(&h_incx, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_incy, batchCount) );
    for (int i=0; i<=batchCount; i++) {
        h_M[i] = 0;
        h_N[i] = 0;
    }
    h_M = h_M;

    // reopen file
    fp = fopen("sizes_sorted.dat","r");
    fscanf(fp, "%d %d\n",&num_sizes,&nlf);

    int m_id_j = 0;
    int m_id = 0;
    int m_upper = 8;
    int total_batch = 0;
    int batchCount_j = batch_sizes1[0];
    int total_size_A_cpu = 0, total_size_A_dev = 0;
    int total_size_X = 0, total_size_Y = 0;
    int num_batch = 0;
    for( int itest = 0; itest < num_sizes; itest+=batchCount ) {
        if (itest < nlf) {
            batchCount = min(batchCount_j, nlf - itest);
        } else {
            if (itest == nlf) {
                for (int i = 0; i < 4; i++) {
                    if (batch_sizes2[i] > 0) {
                        batchCount_j = batch_sizes2[i];
                        break;
                    }
                }
            }
            batchCount = min(batchCount_j, num_sizes - itest);
        }

        int i;
        int size_A_cpu = 0, size_A_dev = 0;
        int size_X = 0, size_Y = 0;
        for (i = 0; i < batchCount; i++) {
            if (h_M[i] == 0) {
                int id, m, n;
                fscanf(fp, "%d %d %d\n",&id,&n,&m);
                h_M[i] = m;
                h_N[i] = n;
            }

            if (itest+i == nlf) {
                if (h_M[i] <= 8) {
                    m_id = 0;
                    m_upper = 8;
                } else {
                    m_id = 1+((h_M[i]-1)/32);
                    m_upper = 32*m_id;
                }
            }
            if (h_M[i] > m_upper) {
                m_id = 1+((h_M[i]-1)/32);
                m_upper = 32*m_id;
                if (itest+i < nlf) {
                    batchCount_j = batch_sizes1[m_id];
                } else {
                    batchCount_j = batch_sizes2[m_id];
                }
                break;
            }
            h_incx[i] = 1;
            h_incy[i] = 1;
                
            h_M[i] = magma_roundup( h_M[i], opts.align );  // multiple of 32 by default
                
            size_A_cpu += h_N[i] * h_M[i];
            size_A_dev += h_N[i] * h_M[i];
                
            size_X += h_N[i] * h_incx[i];
            size_Y += h_M[i] * h_incy[i];
        }
        batchCount = imax(0, i);
        if (batchCount > 0) {
            for (i=0; i<batchCount; i++) {
                h_M[i] = h_N[i] = 0;
            }
            if (h_M[batchCount] != 0) {
                h_M[0] = h_M[batchCount];
                h_N[0] = h_N[batchCount];
                //printf( " save: h_M[0]=h_M[%d]=%d\n",batchCount,h_M[0]);
                h_M[batchCount] = 0;
                h_N[batchCount] = 0;
            }
        }
        total_batch += batchCount;

        total_size_A_dev += size_A_dev;
        total_size_A_cpu += size_A_cpu;
        total_size_X += size_X;
        total_size_Y += size_Y;

        num_batch ++;
    }
    magma_free_cpu( h_M );
    magma_free_cpu( h_N );
    magma_free_cpu( h_incx );
    magma_free_cpu( h_incy );
    fclose(fp);    

    // --------------- //
    // allocate memory //
    // --------------- //
    batchCount = 2*batchCount_0; // for now..
    // reopen file
    fp = fopen("sizes_sorted.dat","r");
    fscanf(fp, "%d %d\n",&num_sizes,&nlf);

    // allocate space for the sizes/leading dim.
    TESTING_CHECK( magma_imalloc_cpu(&h_M, total_batch+1) );
    TESTING_CHECK( magma_imalloc_cpu(&h_N, total_batch+1) );
    TESTING_CHECK( magma_imalloc_cpu(&h_incx, total_batch) );
    TESTING_CHECK( magma_imalloc_cpu(&h_incy, total_batch) );
    for (int i=0; i<=total_batch; i++) {
        h_M[i] = h_N[i] = 0;
    }
    h_M = h_M;
    
    TESTING_CHECK( magma_imalloc(&d_M, total_batch) );
    TESTING_CHECK( magma_imalloc(&d_N, total_batch) );
    TESTING_CHECK( magma_imalloc(&d_ldda, total_batch) );
    TESTING_CHECK( magma_imalloc(&d_incx, total_batch) );
    TESTING_CHECK( magma_imalloc(&d_incy, total_batch) );
    
    double *Anorm, *Xnorm, *Ynorm;
    TESTING_CHECK( magma_dmalloc_cpu( &Anorm, total_batch ));
    TESTING_CHECK( magma_dmalloc_cpu( &Xnorm, total_batch ));
    TESTING_CHECK( magma_dmalloc_cpu( &Ynorm, total_batch ));
    
    TESTING_CHECK( magma_malloc_cpu((void**)&sizes_A, num_batch*sizeof(int)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&sizes_X, num_batch*sizeof(int)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&sizes_Y, num_batch*sizeof(int)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&max_M, num_batch*sizeof(int)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&max_N, num_batch*sizeof(int)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&min_M, num_batch*sizeof(int)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&min_N, num_batch*sizeof(int)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&h_batch_count, num_batch*sizeof(int)) );

    TESTING_CHECK( magma_malloc_cpu((void**)&h_A_array, total_batch*sizeof(double*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&h_X_array, total_batch*sizeof(double*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&h_Y_array, total_batch*sizeof(double*)) );

    TESTING_CHECK( magma_malloc((void**)&d_A_array, total_batch*sizeof(double*)) );
    TESTING_CHECK( magma_malloc((void**)&d_X_array, total_batch*sizeof(double*)) );
    TESTING_CHECK( magma_malloc((void**)&d_Y_array, total_batch*sizeof(double*)) );
    
    TESTING_CHECK( magma_dmalloc_cpu(&h_A, total_size_A_dev) );
    TESTING_CHECK( magma_dmalloc_cpu(&h_X, total_size_X) );
    TESTING_CHECK( magma_dmalloc_cpu(&h_Y, total_size_Y) );
    TESTING_CHECK( magma_dmalloc_cpu(&h_Ymagma, total_size_Y) );

    TESTING_CHECK( magma_dmalloc(&d_A, total_size_A_dev) );
    TESTING_CHECK( magma_dmalloc(&d_X, total_size_X) );
    TESTING_CHECK( magma_dmalloc(&d_Y, total_size_Y) );

    // ------------- //
    // load matrices //
    // ------------- //
    
    // See testing_dgemm about tolerance.
    double eps = lapackf77_dlamch("E");
    double tol = 3*eps;
    
    double total_gpu = 0.0, total_cpu = 0.0;
    batchCount_0 = batch_sizes1[0];
    printf("%% If running lapack (option --lapack), MAGMA error is computed\n"
           "%% relative to CPU BLAS result.\n\n"
           "%% transA = %s\n",
           lapack_trans_const(MagmaNoTrans));
    
    gflops = 0;
    m_id_j = 0;
    m_id = 0;
    m_upper = 8;
    total_size_A_cpu = 0; total_size_A_dev = 0;
    total_size_X = 0;     total_size_Y = 0;
    batchCount_0 = batch_sizes1[0];
    total_batch = 0;
    num_batch = 0;
    for( int itest = 0; itest < num_sizes; itest+=batchCount ) {
        max_M[num_batch] = max_N[num_batch] = 0;
        int size_A_cpu = 0, size_A_dev = 0;
        int size_X = 0, size_Y = 0;
        if (itest < nlf) {
            batchCount = min(batchCount_0, nlf - itest);
        } else {
            if (itest == nlf) {
                for (int i = 0; i < 4; i++) {
                    if (batch_sizes2[i] > 0) {
                        batchCount_0 = batch_sizes2[i];
                        break;
                    }
                }
            }
            batchCount = min(batchCount_0, num_sizes - itest);
        }
        //if (m_id_j != m_id) {
        //    printf( "\n" );
        //}
        m_id_j = m_id;
        int i;
        for (i = 0; i < batchCount; i++) {
            if (h_M[total_batch+i] == 0) {
                int id, m, n;
                fscanf(fp, "%d %d %d\n",&id,&n,&m);
                h_M[total_batch+i] = m;
                h_N[total_batch+i] = n;
            } else {
                //printf( " >> reload(%d): %d %d\n",i,h_M[i],h_N[i] );
            }

            if (itest+i == nlf) {
                if (h_M[total_batch+i] <= 8) {
                    m_id = 0;
                    m_upper = 8;
                } else {
                    m_id = 1+((h_M[total_batch+i]-1)/32);
                    m_upper = 32*m_id;
                }
                //printf( "\n > batch-2\n" );
                m_id_j = m_id;
            }
            if (h_M[total_batch+i] > m_upper) {
                m_id = 1+((h_M[total_batch+i]-1)/32);
                m_upper = 32*m_id;
                if (itest+i < nlf) {
                    batchCount_0 = batch_sizes1[m_id];
                } else {
                    batchCount_0 = batch_sizes2[m_id];
                }
                break;
            }
            h_incx[total_batch+i] = 1;
            h_incy[total_batch+i] = 1;
                
            max_M[num_batch] = max( max_M[num_batch], h_M[total_batch+i] );
            max_N[num_batch] = max( max_N[num_batch], h_N[total_batch+i] );
            if (i == 0) {
                min_M[num_batch] = h_M[total_batch+i];
                min_N[num_batch] = h_N[total_batch+i];
            } else {
                min_M[num_batch] = min( min_M[num_batch], h_M[total_batch+i] );
                min_N[num_batch] = min( min_N[num_batch], h_N[total_batch+i] );
            }
                
            h_M[total_batch+i] = magma_roundup( h_M[total_batch+i], opts.align );  // multiple of 32 by default
            h_M[total_batch+i] = h_M[total_batch+i]; // make it the same for now
    
            size_A_cpu += h_N[total_batch+i] * h_M[total_batch+i];
            size_A_dev += h_N[total_batch+i] * h_M[total_batch+i];
                
            size_X += h_N[total_batch+i] * h_incx[total_batch+i];
            size_Y += h_M[total_batch+i] * h_incy[total_batch+i];
                
            gflops += FLOPS_DGEMV( h_M[total_batch+i], h_N[total_batch+i]) / 1e9;
        }
        batchCount = imax(0, i);
            
        if (batchCount > 0) {
            
            /* Initialize the matrices */
            lapackf77_dlarnv( &ione, ISEED, &size_A_cpu, &h_A[total_size_A_dev] );
            lapackf77_dlarnv( &ione, ISEED, &size_X, &h_X[total_size_X] );
            lapackf77_dlarnv( &ione, ISEED, &size_Y, &h_Y[total_size_Y] );

            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_setvector(batchCount, sizeof(magma_int_t), &h_M[total_batch], 1, &d_M[total_batch], 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), &h_N[total_batch], 1, &d_N[total_batch], 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), &h_M[total_batch], 1, &d_ldda[total_batch], 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), &h_incx[total_batch], 1, &d_incx[total_batch], 1, opts.queue );
            magma_setvector(batchCount, sizeof(magma_int_t), &h_incy[total_batch], 1, &d_incy[total_batch], 1, opts.queue );
            
            h_A_array[0] = &d_A[total_size_A_dev];
            h_X_array[0] = &d_X[total_size_X];
            h_Y_array[0] = &d_Y[total_size_Y];
            for (i = 1; i < batchCount; i++) {
                h_A_array[i] = h_A_array[i-1] + h_N[total_batch+i-1] * h_M[total_batch+i-1];
                h_X_array[i] = h_X_array[i-1] + h_N[total_batch+i-1] * h_incx[total_batch+i-1];
                h_Y_array[i] = h_Y_array[i-1] + h_M[total_batch+i-1] * h_incy[total_batch+i-1];
            }
            magma_setvector(batchCount, sizeof(double*), h_A_array, 1, &d_A_array[total_batch], 1, opts.queue );
            magma_setvector(batchCount, sizeof(double*), h_X_array, 1, &d_X_array[total_batch], 1, opts.queue );
            magma_setvector(batchCount, sizeof(double*), h_Y_array, 1, &d_Y_array[total_batch], 1, opts.queue );
            
            h_A_tmp = &h_A[total_size_A_dev];
            for (i = 0; i < batchCount; i++) {
                magma_dsetmatrix( h_M[total_batch+i], h_N[total_batch+i], 
                                  h_A_tmp, h_M[total_batch+i], 
                                  h_A_array[i], h_M[total_batch+i], opts.queue );
                h_A_tmp += h_N[total_batch+i] * h_M[total_batch+i];
            }
            magma_dsetvector( size_X, &h_X[total_size_X], 1, &d_X[total_size_X], 1, opts.queue );
            magma_dsetvector( size_Y, &h_Y[total_size_Y], 1, &d_Y[total_size_Y], 1, opts.queue );

            total_size_A_dev += size_A_dev;
            total_size_X += size_X;
            total_size_Y += size_Y;

            total_batch += batchCount;

            h_batch_count[num_batch] = batchCount;
            sizes_A[num_batch] = size_A_dev;
            sizes_X[num_batch] = size_X;
            sizes_Y[num_batch] = size_Y;
            num_batch ++;
        }
    }

    // -------------------- //
    // perform batched Gemv //
    // -------------------- //
    total_size_A_dev = 0;
    total_size_X = 0;     total_size_Y = 0;
    total_batch = 0;
//num_batch = 2;
    int num_queues = 5;
    magma_queue_t *queues = (magma_queue_t*)malloc(num_queues * sizeof(magma_queue_t));
    magma_device_t cdev;
    magma_getdevice( &cdev );
    for (int q=0; q<num_queues; q++) {
        magma_queue_create( cdev, &queues[q] );
    }
    magma_time = magma_sync_wtime( opts.queue );
    for (int iid = 0; iid < num_batch; iid ++) {
        batchCount = h_batch_count[iid];
         
        #if 1
/*h_A_array[0] = &d_A[total_size_A_dev];
h_X_array[0] = &d_X[total_size_X];
h_Y_array[0] = &d_Y[total_size_Y];
for (int i = 0; i < min(batchCount,10); i++) {
    magma_dprint_gpu(h_M[total_batch+i],h_N[total_batch+i],h_A_array[i],h_M[total_batch+i], opts.queue);
    magma_dprint_gpu(h_N[total_batch+i],1,h_X_array[i],h_N[total_batch+i], opts.queue);
    magma_dprint_gpu(h_M[total_batch+i],1,h_Y_array[i],h_M[total_batch+i], opts.queue);

    if (i < batchCount-1) {
        h_A_array[i+1] = h_A_array[i] + h_N[total_batch+i] * h_M[total_batch+i];
        h_X_array[i+1] = h_X_array[i] + h_N[total_batch+i] * h_incx[total_batch+i];
        h_Y_array[i+1] = h_Y_array[i] + h_M[total_batch+i] * h_incy[total_batch+i];
    }
}*/
        magmablas_dgemv_vbatched_max_nocheck(
                         MagmaNoTrans, &d_M[total_batch], &d_N[total_batch],
                         alpha, &d_A_array[total_batch], &d_ldda[total_batch],
                                &d_X_array[total_batch], &d_incx[total_batch],
                         beta,  &d_Y_array[total_batch], &d_incy[total_batch],
                         batchCount,
                         max_M[iid], max_N[iid], queues[iid%num_queues]);
        #else
        magmablas_dgemv_vbatched(
                         MagmaNoTrans, d_M, d_N,
                         alpha, d_A_array, d_ldda,
                                d_X_array, d_incx,
                         beta,  d_Y_array, d_incy,
                         batchCount, opts.queue);
        #endif
//for (int i = 0; i < min(batchCount,10); i++) {
//    magma_dprint_gpu(h_M[total_batch+i],1,h_Y_array[i],h_M[total_batch+i], opts.queue);
//}
        total_size_A_dev += sizes_A[iid];
        total_size_X += sizes_X[iid];
        total_size_Y += sizes_Y[iid];

        total_batch += batchCount;
    }
    for (int q=0; q<num_queues; q++) {
        magma_queue_sync( queues[q] );
        magma_queue_destroy( queues[q] );
    }
    magma_time = magma_sync_wtime( opts.queue ) - magma_time;
    magma_perf = gflops / magma_time;
    total_gpu  = magma_time;
    printf( "\n Total time: %.2e seconds on a GPU, %.2e seconds with CPUs (total count=%d)\n\n",
                total_gpu,total_cpu,total_batch );
 
    // ----------- //
    // Check error //
    // ----------- //
    printf("%%              min,max   min,max\n");
    printf("%% BatchCount         M         N   MAGMA error\n");
    printf("%%=============================================\n");
    if ( opts.lapack ) {
        total_batch = 0;
        total_size_A_cpu = 0; total_size_A_dev = 0;
        total_size_X = 0;     total_size_Y = 0;
        for (int iid = 0; iid < num_batch; iid ++) {

            int size_A = sizes_A[iid];
            int size_X = sizes_X[iid];
            int size_Y = sizes_Y[iid];
            magma_dgetvector(size_Y, &d_Y[total_size_Y], 1, h_Ymagma, 1, opts.queue );

            // Compute norms for error
            batchCount = h_batch_count[iid];
            h_A_tmp = &h_A[total_size_A_dev];
            h_X_tmp = &h_X[total_size_X];
            h_Y_tmp = &h_Y[total_size_Y];
            for (int s = 0; s < batchCount; ++s) {
                Anorm[s] = lapackf77_dlange( "F", &h_M[total_batch+s], &h_N[total_batch+s], h_A_tmp, &h_M[total_batch+s],  work );
                Xnorm[s] = lapackf77_dlange( "F", &ione,   &h_N[total_batch+s],  h_X_tmp, &h_incx[total_batch+s], work );
                Ynorm[s] = lapackf77_dlange( "F", &ione,   &h_M[total_batch+s],  h_Y_tmp, &h_incy[total_batch+s], work );

                h_A_tmp += h_N[total_batch+s] * h_M[total_batch+s];
                h_X_tmp += h_N[total_batch+s] * h_incx[total_batch+s];
                h_Y_tmp += h_M[total_batch+s] * h_incy[total_batch+s];
            }
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            // displace pointers for the cpu, reuse h_A_array
            h_A_array[0] = &h_A[total_size_A_dev];
            h_X_array[0] = &h_X[total_size_X];
            h_Y_array[0] = &h_Y[total_size_Y];
            for (int i = 1; i < batchCount; i++) {
                h_A_array[i] = h_A_array[i-1] + h_N[total_batch+i-1] * h_M[total_batch+i-1];
                h_X_array[i] = h_X_array[i-1] + h_N[total_batch+i-1] * h_incx[total_batch+i-1];
                h_Y_array[i] = h_Y_array[i-1] + h_M[total_batch+i-1] * h_incy[total_batch+i-1];
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
//printf( " -- %d (%dx%d)--\n",total_batch+s,h_M[total_batch+s],h_N[total_batch+s] );
//magma_dprint(h_M[total_batch+s],h_N[total_batch+s],h_A_array[s],h_M[total_batch+s]);
//magma_dprint(h_N[total_batch+s],1,h_X_array[s],h_N[total_batch+s]);
//magma_dprint(h_M[total_batch+s],1,h_Y_array[s],h_M[total_batch+s]);

                blasf77_dgemv( lapack_trans_const(MagmaNoTrans),
                               &h_M[total_batch+s], &h_N[total_batch+s],
                               &alpha, h_A_array[s], &h_M[total_batch+s],
                                       h_X_array[s], &h_incx[total_batch+s],
                               &beta,  h_Y_array[s], &h_incy[total_batch+s] );
//magma_dprint(h_M[total_batch+s],1,h_Y_array[s],h_M[total_batch+s]);
            }
            #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
            magma_set_lapack_numthreads(nthreads);
            #endif
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            total_cpu += cpu_time;
 
            /* =====================================================================
              Check the result
               =================================================================== */
            // compute error compared lapack
            // error = |dY - Y| / (gamma_{k+2}|A||X| + gamma_2|Yin|); k = Xn
            magma_error = 0;
                
            h_Y_tmp = &h_Y[total_size_Y];
            h_Ymagma_tmp = h_Ymagma;
            for (int s=0; s < batchCount; s++){
//printf( " -- error %d --\n",s );
//magma_dprint(h_M[total_batch+s],1,h_Y_tmp,h_M[total_batch+s]);
//magma_dprint(h_M[total_batch+s],1,h_Ymagma_tmp,h_M[total_batch+s]);
                normalize = sqrt(double(h_N[total_batch+s]+2))*Anorm[s]*Xnorm[s] + 2*Ynorm[s];
                if (normalize == 0)
                    normalize = 1;
                blasf77_daxpy( &h_M[total_batch+s], &c_neg_one, h_Y_tmp, &h_incy[total_batch+s], h_Ymagma_tmp, &h_incy[total_batch+s] );
                error = lapackf77_dlange( "F", &ione, &h_M[total_batch+s], h_Ymagma_tmp, &h_incy[total_batch+s], work )
                      / normalize;
                magma_error = magma_max_nan( error, magma_error );
             
                h_Y_tmp      += h_M[total_batch+s] * h_incy[total_batch+s];
                h_Ymagma_tmp += h_M[total_batch+s] * h_incy[total_batch+s];
            }

            bool okay = (magma_error < tol);
            status += ! okay;
            printf("  %10lld %5lld,%5lld %5lld,%5lld   %8.2e  %s\n",
                   (long long) batchCount, (long long)min_M[iid], (long long) max_M[iid],
                                           (long long)min_N[iid], (long long) max_N[iid],
                   magma_error, (okay ? "ok" : "failed"));

            total_size_A_dev += size_A;
            total_size_X += size_X;
            total_size_Y += size_Y;

            total_batch += batchCount;
            fflush( stdout);
        }
    }
    fclose(fp);

    // free resources
    magma_free_cpu( h_M );
    magma_free_cpu( h_N );
    magma_free_cpu( h_incx );
    magma_free_cpu( h_incy );

    magma_free_cpu( Anorm );
    magma_free_cpu( Xnorm );
    magma_free_cpu( Ynorm );

    magma_free_cpu( h_A_array );
    magma_free_cpu( h_X_array );
    magma_free_cpu( h_Y_array );
    
    magma_free_cpu( h_A );
    magma_free_cpu( h_X );
    magma_free_cpu( h_Y );
    magma_free_cpu( h_Ymagma );

    magma_free( d_M );
    magma_free( d_N );
    magma_free( d_ldda );
    magma_free( d_incx );
    magma_free( d_incy );
    magma_free( d_A_array );
    magma_free( d_X_array );
    magma_free( d_Y_array );
    
    magma_free( d_A );
    magma_free( d_X );
    magma_free( d_Y );

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
