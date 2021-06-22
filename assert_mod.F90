      module assert_mod
      implicit none
      contains
      subroutine assert(lcond,msg,ival)
      implicit none
      logical, intent(in) :: lcond
      character*(*), intent(in) :: msg
      integer, intent(in) :: ival

      if (.not.lcond) then
          print*,msg, ival
          stop '** assertion error ** '
      endif
      return
      end subroutine assert
      end module assert_mod
