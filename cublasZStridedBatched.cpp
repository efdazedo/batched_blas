#include <stdlib.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
static
void synchronize()
{
               cudaError_t istat = cudaDeviceSynchronize();
               assert( istat == cudaSuccess );
}

static
void *gpu_alloc(  size_t nbytes )
{
        void *devPtr = nullptr;

        cudaError_t istat = cudaMalloc( &devPtr, nbytes );
        assert( istat == cudaSuccess );

        return( devPtr );

}

static
void gpu_free( void *devPtr )
{
        cudaError_t istat = cudaFree( devPtr );
        assert( istat == cudaSuccess );
}

static
void host2gpu( void *dest, void *src, size_t nbytes )
{
        cudaError_t istat = cudaMemcpy( dest, src, nbytes, cudaMemcpyHostToDevice);
        assert( istat == cudaSuccess );

        synchronize();
}

static 
void gpu2host( void *dest, void *src, size_t nbytes )
{
        cudaError_t istat = cudaMemcpy( dest, src, nbytes, cudaMemcpyDeviceToHost);
        assert( istat == cudaSuccess );

        synchronize();
}

static cublasHandle_t handle;

extern "C"
void cublasCreate_()
{
       cublasStatus_t istat = cublasCreate( &handle );
       assert( istat == CUBLAS_STATUS_SUCCESS );
}

extern "C"
void cublasDestroy_()
{
        cublasStatus_t istat = cublasDestroy( handle );
        assert( istat == CUBLAS_STATUS_SUCCESS );
}



extern "C"
void cublasZgetrfStridedBatched_( 
        int n,
        cuDoubleComplex *dA_,
        int lda,
        size_t strideA,
        int *dpivot_,
        int *dinfo_,
        int batchCount )
{
   
   // ------------------------
   // setup  array of pointers
   // ------------------------
        size_t nbytes = sizeof(cuDoubleComplex *) * (batchCount + 1);
        cuDoubleComplex **hAarray = (cuDoubleComplex **) malloc( nbytes );
        assert( hAarray != nullptr );


        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                size_t ip = ibatch * strideA;
                hAarray[ibatch] = &(dA_[ip]);
        };

        cuDoubleComplex **dAarray = (cuDoubleComplex **)  gpu_alloc( nbytes );
        assert( dAarray != nullptr );

        host2gpu( dAarray, hAarray, nbytes );

       cublasStatus_t istat = cublasZgetrfBatched( handle,
                                                   n,
                                                   dAarray,
                                                   lda,
                                                   dpivot_,
                                                   dinfo_,
                                                   batchCount );
       assert( istat == CUBLAS_STATUS_SUCCESS);

       synchronize();

       free( hAarray );
       gpu_free( dAarray );
}


extern "C"
void cublasZgetrsStridedBatched_(
                       int n,
                       int nrhs,
                       char c_trans,
                       cuDoubleComplex *dA_,
                       int lda,
                       size_t strideA,
                       int *dpivot_,
                       cuDoubleComplex *dB_,
                       int ldb,
                       size_t strideB,
                       int *dinfo_,
                       int batchCount )
{
        size_t nbytes = sizeof(cuDoubleComplex *) * (batchCount + 1);
        cuDoubleComplex **hAarray = (cuDoubleComplex **) malloc( nbytes );
        cuDoubleComplex **hBarray = (cuDoubleComplex **) malloc( nbytes );

        assert( hAarray != nullptr );
        assert( hBarray != nullptr );

        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                size_t ip_A = ibatch * strideA;
                size_t ip_B = ibatch * strideB;

                hAarray[ ibatch ] = &(dA_[ip_A]);
                hBarray[ ibatch ] = &(dB_[ip_B]);
        };

        cuDoubleComplex **dAarray = (cuDoubleComplex **) gpu_alloc( nbytes );
        cuDoubleComplex **dBarray = (cuDoubleComplex **) gpu_alloc( nbytes );

        host2gpu( dAarray, hAarray, nbytes );
        host2gpu( dBarray, hBarray, nbytes );

        {
        cublasOperation_t trans = ((c_trans == 'N') || (c_trans == 'n')) ? CUBLAS_OP_N :
                                  ((c_trans == 'T') || (c_trans == 't')) ? CUBLAS_OP_T :
                                                                           CUBLAS_OP_C ;
        cublasStatus_t istat = cublasZgetrsBatched( 
                                         handle,
                                         trans,
                                         n,
                                         nrhs,
                                         dAarray,
                                         lda,
                                         dpivot_,
                                         dBarray,
                                         ldb,
                                         dinfo_,
                                         batchCount );
        assert( istat == CUBLAS_STATUS_SUCCESS );
        }

        synchronize();

        gpu_free( dAarray );
        gpu_free( dBarray );

        free( hAarray );
        free( hBarray );
}
