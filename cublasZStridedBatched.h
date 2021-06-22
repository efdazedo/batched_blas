#ifndef CUBLAS_ZSTRIDEDBATCHED_H
#define CUBLAS_ZSTRIDEDBATCHED_H 1
      interface

        subroutine cublasCreate() bind(c,name='cublasCreate_')
        end subroutine cublasCreate

        subroutine cublasDestroy() bind(c,name='cublasDestroy_')
        end subroutine cublasDestroy

        subroutine cublasZgetrsStridedBatched(n,nrhs,c_trans,            &
     &      dA,ldA,strideA,  dpivot, dB, ldB, strideB,                   &
     &      dinfo, batchCount)                                           &
     &      bind(c,name='cublasZgetrsStridedBatched_')

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

#ifdef __PGI
        attributes(device) :: dA,dB,dpivot
#endif
        end subroutine cublasZgetrsStridedBatched


        subroutine cublasZgetrfStridedBatched( n, dA, ldA, strideA,      &
     &       dpivot, dinfo, batchCount)                                  &
     &       bind(c,name='cublasZgetrfStridedBatched_')
        use iso_c_binding
        implicit none

        integer(kind=c_int), value :: n
        integer(kind=c_int), value :: batchCount
        integer(kind=c_int), value :: ldA

        integer(kind=c_size_t), value :: strideA

        integer(kind=c_int) :: dpivot(*)
        integer(kind=c_int) :: dinfo(batchCount)

        complex(kind=c_double_complex) :: dA(*)

#ifdef __PGI
        attributes(device) :: dA,dinfo,dpivot
#endif
        end subroutine cublasZgetrfStridedBatched


        end interface
#endif
