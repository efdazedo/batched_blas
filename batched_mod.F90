      module batched_mod
      use iso_c_binding

#ifdef __PGI
       use cudafor
#endif
      implicit none


      interface
         subroutine zgetrs( trans,                                       &
     &    n,nrhs,A,ldA,ipiv,B,ldB,info)
         implicit none
         character trans
         integer n,nrhs,ldA,ldB,info
         integer ipiv(*)
         complex*16 A(ldA,*), B(ldB,*)
         end subroutine zgetrs

         subroutine zgetrf( m,n,A,ldA,ipiv,info)
         implicit none
         integer m,n,ldA,info
         integer ipiv(*)
         complex*16 A(ldA,*)
         end subroutine zgetrf

      end interface

#ifdef USE_CUBLAS
#include "cublasZStridedBatched.h"
#endif

#ifdef USE_MAGMA
#include "magmaZStridedBatched.h"
#endif

        contains

        subroutine ZgetrfStridedBatchedF( n, dA, ldA, strideA, &
     &       dpivot, dinfo, batchCount )
        use iso_c_binding
        implicit none
        integer(kind=c_int), value :: n, batchCount, ldA
        integer(kind=c_size_t), value :: strideA

        integer(kind=c_int) :: dpivot(*)
        integer(kind=c_int) :: dinfo(batchCount)

        complex(kind=c_double_complex) :: dA(*)

        integer :: mm, nn, ibatch, ip_A,ip_pivot

#ifdef USE_CUBLAS

#ifdef _OPENACC
!$acc host_data use_device(dA,dpivot,dinfo)
#else
!$omp target data use_dev_ptr(dA,dpivot,dinfo)
#endif

            call cublasZgetrfStridedBatched(n,dA,ldA,strideA,            &
     &              dpivot, dinfo, batchCount )

#ifdef _OPENACC
!$acc end host_data
#else
!$omp end target data
#endif

#else

#ifdef USE_MAGMA

#ifdef _OPENACC
!$acc host_data use_device(dA,dpivot,dinfo)
#else
!$omp target data map(tofrom:dA,dpivot) map(from:dinfo)
#endif

            call magmaZgetrfStridedBatched(n,dA,ldA,strideA,             &
     &              dpivot, dinfo, batchCount )

#ifdef _OPENACC
!$acc end host_data
#else
!$omp end target data
#endif

#else

!$omp parallel do private(ibatch,ip_A,ip_pivot,mm,nn)
            do ibatch=1,batchCount
               ip_A = 1 + (ibatch-1)*strideA
               ip_pivot = 1 + (ibatch-1)*n
               mm = n
               nn = n
               call  zgetrf(mm,nn, dA(ip_A),                             &
     &                 ldA,dpivot(ip_pivot), dinfo(ibatch)) 
            enddo
#endif


#endif

         return
         end subroutine ZgetrfStridedBatchedF




        subroutine ZgetrsStridedBatchedF(c_trans,n,nrhs,                 &
     &      dA,ldA,strideA,  dpivot, dB, ldB, strideB,                   &
     &      dinfo, batchCount)                                           

        use iso_c_binding
        implicit none

        integer(kind=c_int), value :: n
        integer(kind=c_int), value :: nrhs
        integer(kind=c_int), value :: batchCount
        integer(kind=c_int), value :: ldA
        integer(kind=c_int), value :: ldB

        character(kind=c_char), value :: c_trans

        complex(kind=c_double_complex) :: dA(*)
        complex(kind=c_double_complex) :: dB(*)

        integer(kind=c_size_t), value :: strideA
        integer(kind=c_size_t), value :: strideB

        integer(kind=c_int) :: dinfo(batchCount)
        integer(kind=c_int) :: dpivot(*)



        character :: trans
        integer :: ibatch,ip_A, ip_B, ip_pivot

        
#ifdef USE_CUBLAS

#ifdef _OPENACC
!$acc host_data use_device(dA,dB,dpivot)
#else
!$omp target data map(to:dA,dpivot) map(tofrom:dB)
#endif

          call cublasZgetrsStridedBatched(n,nrhs,c_trans,                &
     &      dA,ldA,strideA,  dpivot, dB, ldB, strideB,                   &
     &      dinfo, batchCount)

#ifdef _OPENACC
!$acc end host_data
#else
!$omp end target data
#endif

#else

#ifdef USE_MAGMA

#ifdef _OPENACC
!$acc host_data use_device(dA,dB,dpivot)
#else
!$omp target data map(to:dA,dpivot) map(tofrom:dB)
#endif

          call magmaZgetrsStridedBatched(n,nrhs,c_trans,                 &
     &      dA,ldA,strideA,  dpivot, dB, ldB, strideB,                   &
     &      dinfo, batchCount)
#ifdef _OPENACC
!$acc end host_data
#else
!$omp end target data
#endif

#else

        trans = c_trans

!$omp parallel do private(ibatch,ip_A,ip_B,ip_pivot)
             do ibatch=1,batchCount
               ip_A = 1 + (ibatch-1)*strideA
               ip_B = 1 + (ibatch-1)*strideB
               ip_pivot = 1 + (ibatch-1)*n
               call zgetrs( trans, n,nrhs,                               &
     &                 dA(ip_A),ldA, dpivot(ip_pivot),                   &
     &                 dB(ip_B), ldB, dinfo(ibatch))
             enddo

#endif

#endif
          return
          end subroutine ZgetrsStridedBatchedF

          subroutine init_batched()
#ifdef USE_CUBLAS
          call cublasCreate()
#endif

#ifdef USE_MAGMA
          call magmaCreate()
#endif
          end subroutine init_batched

          subroutine finalize_batched()
#ifdef USE_CUBLAS
          call cublasDestroy()
#endif

#ifdef USE_MAGMA
         call magmaDestroy()
#endif
          end subroutine finalize_batched

       end module batched_mod
