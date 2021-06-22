#include <stdlib.h>
#include <assert.h>

#include "magmablas.h"
#include "magma_v2.h"


static magma_queue_t queue;
static size_t dAarray_nbytes = 0;
static size_t dBarray_nbytes = 0;
static size_t dipiv_array_nbytes = 0;
static magmaDoubleComplex **dAarray_saved = NULL;
static magmaDoubleComplex **dBarray_saved = NULL;
static magma_int_t ** dipiv_array_saved = NULL;

static
void synchronize()
{
        magma_queue_sync( queue );
}


static
void *gpu_alloc(  size_t nbytes )
{
        void *devPtr = nullptr;


        magma_int_t istat = magma_malloc( &devPtr, nbytes );

        assert( devPtr != nullptr );
        assert( istat == MAGMA_SUCCESS );

        return( devPtr );

}

static
void gpu_free( void *devPtr )
{
  if (devPtr != nullptr) {
        magma_int_t istat = magma_free( devPtr );
        assert( istat == MAGMA_SUCCESS );
  };
}

static
void gpu2gpu( void *dest, void *src, size_t nbytes )
{
        magma_int_t const n = 1;
        magma_int_t const incx = 1;
        magma_int_t const incy = 1;
        magma_int_t const elemSize = nbytes;

       assert( elemSize >= 0 );
       assert( queue != 0);
       magma_copyvector( n, elemSize, src, incx, dest, incy, queue );

}

static
void host2gpu( void *dest, void *src, size_t nbytes )
{
        if (nbytes == 0) return;


        magma_int_t const n = 1;
        magma_int_t const incx = 1;
        magma_int_t const incy = 1;
        magma_int_t const elemSize = nbytes;
        if (nbytes == 0) return;



        assert( elemSize >= 0 );
        assert( queue != 0);
        assert( src != nullptr );
        assert( dest != nullptr );
        magma_setvector( n, elemSize, src, incx, dest, incy, queue );

        synchronize();
}

static 
void gpu2host( void *dest, void *src, size_t nbytes )
{
        if (nbytes == 0) return;

        magma_int_t const n = 1;
        magma_int_t const incx = 1;
        magma_int_t const incy = 1;
        magma_int_t const elemSize = nbytes;

        assert( elemSize >= 0 );
        assert( queue != 0);
        assert( src != nullptr );
        assert( dest != nullptr );

        magma_getvector(n, elemSize, src, incx, dest, incy, queue );

        synchronize();
}

extern "C"
void magmaCreate_()
{
        magma_int_t device = 0;
        queue = 0;
        magma_queue_create( device, &queue );
        assert( queue != 0);
}

extern "C"
void magmaDestroy_()
{
        assert( queue != 0);
        magma_queue_destroy( queue );


        dAarray_nbytes = 0;
        dBarray_nbytes = 0;
        dipiv_array_nbytes = 0;

        if (dAarray_saved != nullptr) { gpu_free( dAarray_saved ); };
        if (dBarray_saved != nullptr) { gpu_free( dBarray_saved ); };
        if (dipiv_array_saved != nullptr) { gpu_free( dipiv_array_saved ); };


        dAarray_saved = nullptr;
        dBarray_saved = nullptr;
        dipiv_array_saved = nullptr;

}



extern "C"
void magmaZgetrfStridedBatched_( 
        int n,
        magmaDoubleComplex *dA_,
        int lda,
        size_t strideA,
        int *dpivot_,
        int *dinfo_,
        int batchCount )
{
   
   // ------------------------
   // setup  array of pointers
   // ------------------------
        size_t nbytes = sizeof(magmaDoubleComplex *) * (batchCount + 1);
        magmaDoubleComplex **hAarray = (magmaDoubleComplex **) malloc( nbytes );
        assert( hAarray != nullptr );


        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                size_t ip = ibatch * strideA;
                hAarray[ibatch] = &(dA_[ip]);
        };


        if (dAarray_nbytes < nbytes) {
                gpu_free( dAarray_saved );
                dAarray_saved = (magmaDoubleComplex **) gpu_alloc(nbytes);
                assert( dAarray_saved != nullptr );
                dAarray_nbytes = nbytes;
        };

        magmaDoubleComplex **dAarray = dAarray_saved;
        assert( dAarray != nullptr );
        host2gpu( dAarray, hAarray, nbytes );

        //  magma need host array for pivot and info ?
        magma_int_t const m = n;

        magma_int_t **ipiv_array = (magma_int_t **) malloc( sizeof(magma_int_t *)*(batchCount+1));

        assert( ipiv_array != 0);

        magma_int_t minMN = (m <= n) ? m : n;
        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                size_t ip = ibatch * minMN;
                ipiv_array[ibatch] = &(dpivot_[ ip ]);
        };

        magma_int_t **dipiv_array = (magma_int_t **) gpu_alloc( sizeof(magma_int_t *) *
                                                                (batchCount + 1) );
        assert( dipiv_array != 0);
        host2gpu( dipiv_array, ipiv_array, sizeof(magma_int_t *)*batchCount );
                

       magma_int_t istat = magma_zgetrf_batched(   m,
                                                   n,
                                                   dAarray,
                                                   lda,
                                                   dipiv_array,
                                                   dinfo_,
                                                   batchCount,
                                                   queue );
       synchronize();
       assert( istat == MAGMA_SUCCESS);


       synchronize();

       free( ipiv_array );
       free( hAarray );
}


extern "C"
void magmaZgetrsStridedBatched_(
                       int n,
                       int nrhs,
                       char c_trans,
                       magmaDoubleComplex *dA_,
                       int lda,
                       size_t strideA,
                       int *dpivot_,
                       magmaDoubleComplex *dB_,
                       int ldb,
                       size_t strideB,
                       int *dinfo_,
                       int batchCount )
{
        size_t nbytes = sizeof(magmaDoubleComplex *) * (batchCount + 1);
        magmaDoubleComplex **hAarray = (magmaDoubleComplex **) malloc( nbytes );
        magmaDoubleComplex **hBarray = (magmaDoubleComplex **) malloc( nbytes );

        assert( hAarray != nullptr );
        assert( hBarray != nullptr );

        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                size_t ip_A = ibatch * strideA;
                size_t ip_B = ibatch * strideB;

                hAarray[ ibatch ] = &(dA_[ip_A]);
                hBarray[ ibatch ] = &(dB_[ip_B]);
        };

        if (dAarray_nbytes < nbytes) {
                if (dAarray_saved != nullptr) { gpu_free( dAarray_saved ); };
                dAarray_saved = (magmaDoubleComplex **) gpu_alloc(nbytes );
                assert(dAarray_saved != nullptr );
                dAarray_nbytes = nbytes;
        };
        if (dBarray_nbytes < nbytes) {
                if (dBarray_saved != nullptr) { gpu_free( dBarray_saved ); };
                dBarray_saved = (magmaDoubleComplex **) gpu_alloc(nbytes);
                assert( dBarray_saved != nullptr );
                dBarray_nbytes = nbytes;
        };



        magmaDoubleComplex **dAarray = dAarray_saved;
        magmaDoubleComplex **dBarray = dBarray_saved;

        assert( dAarray != nullptr );
        assert( dBarray != nullptr );
        host2gpu( dAarray, hAarray, nbytes );
        host2gpu( dBarray, hBarray, nbytes );

        magma_int_t **hipiv_array = (magma_int_t **) malloc( sizeof(magma_int_t *) * (batchCount+1));
        assert( hipiv_array != 0 );

        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                hipiv_array[ibatch] = &(dpivot_[ ibatch * n ]);
        };

        size_t nbytes_dipiv = sizeof(magma_int_t *) * (batchCount + 1 );
        if (dipiv_array_nbytes  < nbytes_dipiv) {
                if (dipiv_array_saved != NULL) { gpu_free( dipiv_array_saved ); };
                dipiv_array_saved = (magma_int_t **) gpu_alloc( nbytes_dipiv );
                assert( dipiv_array_saved != NULL );
                dipiv_array_nbytes = nbytes_dipiv;
        };

        magma_int_t **dipiv_array = dipiv_array_saved;
        assert( dipiv_array != 0);

        host2gpu( dipiv_array, hipiv_array, sizeof(magma_int_t *)*batchCount );


        {
        magma_trans_t trans =     ((c_trans == 'N') || (c_trans == 'n')) ? MagmaNoTrans :
                                  ((c_trans == 'T') || (c_trans == 't')) ? MagmaTrans :
                                                                           MagmaConjTrans ;
        magma_int_t istat = magma_zgetrs_batched( 
                                         trans,
                                         n,
                                         nrhs,
                                         dAarray,
                                         lda,
                                         dipiv_array,
                                         dBarray,
                                         ldb,
                                         batchCount,
                                         queue );
        assert( istat == MAGMA_SUCCESS );
        }

        synchronize();


        free( hipiv_array );
        free( hAarray );
        free( hBarray );
}
