#include <stdlib.h>
#include <assert.h>

#include <hip/hip_runtime.h>
#include <hipblas.h>

typedef hipblasDoubleComplex hipDoubleComplex;
static
void synchronize()
{
               hipError_t istat = hipDeviceSynchronize();
               assert( istat == hipSuccess );
}

static
void *gpu_alloc(  size_t nbytes )
{
        void *devPtr = nullptr;

        hipError_t istat = hipMalloc( &devPtr, nbytes );
        assert( istat == hipSuccess );

        return( devPtr );

}

static
void gpu_free( void *devPtr )
{
        hipError_t istat = hipFree( devPtr );
        assert( istat == hipSuccess );
}

static
void host2gpu( void *dest, void *src, size_t nbytes )
{
        hipError_t istat = hipMemcpy( dest, src, nbytes, hipMemcpyHostToDevice);
        assert( istat == hipSuccess );

        synchronize();
}

static 
void gpu2host( void *dest, void *src, size_t nbytes )
{
        hipError_t istat = hipMemcpy( dest, src, nbytes, hipMemcpyDeviceToHost);
        assert( istat == hipSuccess );

        synchronize();
}

static hipblasHandle_t handle;

extern "C"
void hipblasCreate_()
{
       hipblasStatus_t istat = hipblasCreate( &handle );
       assert( istat == HIPBLAS_STATUS_SUCCESS );
}

extern "C"
void hipblasDestroy_()
{
        hipblasStatus_t istat = hipblasDestroy( handle );
        assert( istat == HIPBLAS_STATUS_SUCCESS );
}



extern "C"
void hipblasZgetrfStridedBatched_( 
        int n,
        hipDoubleComplex *dA_,
        int lda,
        size_t strideA,
        int *dpivot_,
        int *dinfo_,
        int batchCount )
{
   
   // ------------------------
   // setup  array of pointers
   // ------------------------
        size_t nbytes = sizeof(hipDoubleComplex *) * (batchCount + 1);
        hipDoubleComplex **hAarray = (hipDoubleComplex **) malloc( nbytes );
        assert( hAarray != nullptr );


        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                size_t ip = ibatch * strideA;
                hAarray[ibatch] = &(dA_[ip]);
        };

        hipDoubleComplex **dAarray = (hipDoubleComplex **)  gpu_alloc( nbytes );
        assert( dAarray != nullptr );

        host2gpu( dAarray, hAarray, nbytes );

       hipblasStatus_t istat = hipblasZgetrfBatched( handle,
                                                   n,
                                                   dAarray,
                                                   lda,
                                                   dpivot_,
                                                   dinfo_,
                                                   batchCount );
       assert( istat == HIPBLAS_STATUS_SUCCESS);

       synchronize();

       free( hAarray );
       gpu_free( dAarray );
}


extern "C"
void hipblasZgetrsStridedBatched_(
                       int n,
                       int nrhs,
                       char c_trans,
                       hipDoubleComplex *dA_,
                       int lda,
                       size_t strideA,
                       int *dpivot_,
                       hipDoubleComplex *dB_,
                       int ldb,
                       size_t strideB,
                       int *dinfo_,
                       int batchCount )
{
        size_t nbytes = sizeof(hipDoubleComplex *) * (batchCount + 1);
        hipDoubleComplex **hAarray = (hipDoubleComplex **) malloc( nbytes );
        hipDoubleComplex **hBarray = (hipDoubleComplex **) malloc( nbytes );

        assert( hAarray != nullptr );
        assert( hBarray != nullptr );

        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                size_t ip_A = ibatch * strideA;
                size_t ip_B = ibatch * strideB;

                hAarray[ ibatch ] = &(dA_[ip_A]);
                hBarray[ ibatch ] = &(dB_[ip_B]);
        };

        hipDoubleComplex **dAarray = (hipDoubleComplex **) gpu_alloc( nbytes );
        hipDoubleComplex **dBarray = (hipDoubleComplex **) gpu_alloc( nbytes );

        host2gpu( dAarray, hAarray, nbytes );
        host2gpu( dBarray, hBarray, nbytes );

        {
        hipblasOperation_t trans = ((c_trans == 'N') || (c_trans == 'n')) ? HIPBLAS_OP_N :
                                  ((c_trans == 'T') || (c_trans == 't')) ? HIPBLAS_OP_T :
                                                                           HIPBLAS_OP_C ;
        hipblasStatus_t istat = hipblasZgetrsBatched( 
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
        assert( istat == HIPBLAS_STATUS_SUCCESS );
        }

        synchronize();

        gpu_free( dAarray );
        gpu_free( dBarray );

        free( hAarray );
        free( hBarray );
}
