      module test_batched_mod
      contains
      subroutine test_batched_omp(n,batchCount)
      use assert_mod
      use batched_mod
      use iso_c_binding
      implicit none


      interface
         subroutine zgemm( transA, transB, m,n,k,                        &
     &           alpha, A, ldA,  B, ldB, beta, C, ldC)
         character transA
         character transB
         integer m, n, k, ldA,ldB,ldC
         complex*16 alpha,beta
         complex*16 A(ldA,*), B(ldB,*), C(ldC,*)
         end subroutine zgemm
      end interface

      integer, intent(in) :: n, batchCount

      integer :: t1,t2,count_rate
      real*8 :: ttime
      real*8 :: gflops
      logical :: isok

      integer, parameter :: wp = selected_real_kind(15,100)

      integer :: ldA,ldB,ldX,nrhs
      complex(kind=wp), allocatable :: A(:,:,:)
      complex(kind=wp), allocatable :: Aorg(:,:,:)
      complex(kind=wp), allocatable :: B(:,:,:)
      complex(kind=wp), allocatable :: X(:,:,:)
      complex(kind=wp), allocatable :: Xnew(:,:,:)
      integer :: i,j,ibatch
      integer(kind=c_int), allocatable :: dpivot(:,:)
      integer(kind=c_int), allocatable :: dinfo(:)

      integer(kind=c_size_t) :: strideA,strideB
      character(kind=c_char) :: c_trans

      real*8, allocatable :: xre(:), xim(:)
      real*8 :: max_abserr, abserr, res, max_res
      complex(kind=wp)  :: alpha, beta
      integer :: ld1, ld2, ld3, mm,nn,kk


      nrhs = n
      ldA = n
      ldB = n
      ldX = n


      allocate( A(ldA,n,batchCount ) )
      allocate( Aorg(ldA,n,batchCount ) )
      allocate( B(ldB,nrhs,batchCount) )
      allocate( X(ldX,nrhs,batchCount) )
      allocate( Xnew(ldX,nrhs,batchCount) )

      allocate( dpivot(n,batchCount) )
      allocate( dinfo(batchCount) )

      allocate( xre(n), xim(n) )

      do ibatch=1,batchCount
      do j=1,n
        call random_number(xre(1:n))
        call random_number(xim(1:n))
        do i=1,n
          xre(i) = 2*xre(i)-1
          xim(i) = 2*xim(i)-1
        enddo
        do i=1,n
         A(i,j,ibatch) = dcmplx( xre(i), xim(i) )
        enddo
      enddo
      enddo

!$omp parallel do collapse(3) private(i,j,ibatch)
      do ibatch=1,batchCount
      do j=1,n
      do i=1,n
        Aorg(i,j,ibatch) = A(i,j,ibatch)
      enddo
      enddo
      enddo


      do ibatch=1,batchCount
      do j=1,nrhs
        call random_number(xre)
        call random_number(xim)
        do i=1,n
          xre(i) = 2*xre(i)-1
          xim(i) = 2*xim(i)-1
        enddo
        do i=1,n
         X(i,j,ibatch) = dcmplx(xre(i),xim(i))
        enddo
      enddo
      enddo

!$omp parallel do private(ibatch)
      do ibatch=1,batchCount
        mm = n
        nn = nrhs
        kk = n
        alpha = 1
        beta = 0
        ld1 = ldA
        ld2 = ldX
        ld3 = ldB
        call zgemm( 'N', 'N', mm,nn,kk,                                  &
     &        alpha, Aorg(1,1,ibatch), ld1,                              &
     &               X(1,1,ibatch), ld2,                                 &
     &        beta,  B(1,1,ibatch), ld3 )
      enddo

      call init_batched()

#ifdef _OPENACC
!$acc  enter data                                                        &
!$acc& create(dinfo,dpivot)                                              &
!$acc& copyin(A,B)
#else
!$omp  target enter data                                                       &
!$omp& map(alloc:dinfo,dpivot)                                           &
!$omp& map(to:A,B) 
#endif
      call system_clock(t1,count_rate)
      strideA = ldA*n
      call ZgetrfStridedBatchedF(                                   &
              n, A, ldA, strideA, dpivot,dinfo,batchCount )
      call system_clock(t2,count_rate)
      ttime = dble(t2-t1)/dble(count_rate)
      gflops = (batchCount * 2.67d0 * dble(n)**3) * 1.0d-9
      print 9010, ttime, gflops/ttime
 9010 format(' time for LU took ',f5.1, ' sec ',f8.1,' Gflops/sec') 

#ifdef _OPENACC
!$acc update host(dinfo)
#else
!$omp target update from(dinfo)
#endif
      isok = all( dinfo(1:batchCount) .eq. 0)
      if (.not.isok) then
         do ibatch=1,batchCount
           if (dinfo(ibatch).ne.0) then
               print*,'dinfo(',ibatch,') = ',dinfo(ibatch)
           endif
         enddo
      endif
      call assert(isok,                                                  &
     &   'ZgetrfStridedBatched return dinfo.ne.0',maxval(dinfo))



      strideB = ldB * nrhs
      c_trans ='N'

      call system_clock(t1,count_rate)

      call ZgetrsStridedBatchedF(                                        &
     &      c_trans,n,nrhs,A,ldA,strideA,dpivot,                         &
     &      B,ldB,strideB, dinfo, batchCount)
      call system_clock(t2,count_rate)
      ttime = dble(t2-t1)/dble(count_rate)
      gflops = batchCount * (4.0d0*n*n*nrhs) * 1.0d-9
      print 9020, ttime, gflops/ttime
 9020 format(' time for solve ', f5.1,' sec ', f8.1, ' Gflops/sec')

#ifdef _OPENACC
!$acc update host(B)
#else
!$omp target update from(B)
#endif

      isok = all( dinfo(1:batchCount).eq.0)
      if (.not.isok) then
         do ibatch=1,batchCount
           if (dinfo(ibatch).ne.0) then
              print*,'dinfo(',ibatch,')=',dinfo(ibatch)
           endif
         enddo
      endif
      call assert( isok,                                                 &
     &  'ZgetrsStridedBatched return non-zero', maxval(dinfo))

!$omp parallel do collapse(3) private(i,j,ibatch)
      do ibatch=1,batchCount
      do j=1,nrhs
      do i=1,n
        Xnew(i,j,ibatch) = B(i,j,ibatch)
      enddo
      enddo
      enddo

!    ---------------
!    check solution
!    ---------------
      max_abserr = 0
!$omp  parallel do collapse(3)                                           &
!$omp& private(i,j,ibatch,abserr)                                        &
!$omp& reduction(max:max_abserr)
      do ibatch=1,batchCount
        do j=1,n
        do i=1,n
          abserr = abs(Xnew(i,j,ibatch)-X(i,j,ibatch)) 
          max_abserr = max( max_abserr, abserr )
        enddo
        enddo
      enddo
      print*,'max_abserr = ', max_abserr

!     --------------
!     check residual
!     --------------
!$omp  parallel do                                                      &
!$omp& private(ibatch,alpha,beta,mm,nn,kk,ld1,ld2,ld3)
      do ibatch=1,batchCount
        alpha = 1
        beta = 0
        mm = n
        nn = nrhs
        kk = n
        ld1 = ldA
        ld2 = ldX
        ld3 = ldB
        call zgemm( 'N','N', mm,nn,kk,                                   &
     &         alpha, Aorg(1,1,ibatch), ld1,                             &
     &                X(1,1,ibatch), ld2,                                &
     &         beta,  B(1,1,ibatch), ld3 )

         alpha = -1
         beta = 1
        call zgemm( 'N','N', mm,nn,kk,                                   & 
     &          alpha, Aorg(1,1,ibatch), ld1,                            &
     &                 Xnew(1,1,ibatch), ld2,                            &
     &          beta,  B(1,1,ibatch), ld3 )
       enddo

       max_res = 0
!$omp  parallel do collapse(3)                                           &
!$omp& private(i,j,ibatch,res)                                           &
!$omp& reduction(max:max_res)
       do ibatch=1,batchCount
       do j=1,nrhs
       do i=1,nrhs
          res = abs( B(i,j,ibatch) )
          max_res = max( max_res, res )
       enddo
       enddo
       enddo

       print*,'max_res = ',  max_res




#ifdef _OPENACC
!$acc exit data delete(A,B,dpivot,dinfo)
#else
!$omp target exit data map(release:A,B,dpivot,dinfo)
#endif


      call finalize_batched()

      deallocate( A, Aorg, B, X, Xnew )
      deallocate( dpivot, dinfo )
      deallocate( xre, xim )

      return
      end subroutine test_batched_omp
      end module test_batched_mod

      program test_batch
      use test_batched_mod
      implicit none
      integer :: n, batchCount
      integer :: it, ntimes

      batchCount = 16*4;
      n = 32;
      ntimes = 5;
      print*,'n = ',n,' batchCount = ', batchCount
      print*,'ntimes = ', ntimes

      do it=1,ntimes
       call test_batched_omp(n,batchCount)
      enddo
      stop
      end program test_batch

