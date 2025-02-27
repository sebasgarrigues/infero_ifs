!
! (C) Copyright 1996- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.
!

program my_program
use inferof
use iso_c_binding, only : c_double, c_int, c_float, c_char, c_null_char, c_ptr
implicit none

real(c_float) :: tol = 1e-3;
integer, parameter :: n_inference_reps = 10
integer :: n_batches

! Command line arguments
character(1024) :: model_path
character(1024) :: model_type
character(1024) :: input_path
character(1024) :: n_batches_str
character(1024) :: ref_output_path
character(1024) :: yaml_config
character(1024) :: tol_str

! model of infero model
type(infero_model) :: model

! indexes and Tensor dimensions
integer :: ss, i, j, ch
integer :: argc

integer :: input_dim_0
integer :: input_dim_1
integer :: input_dim_2
integer :: input_dim_3
integer :: input_dim_flat

integer :: output_dim_0
integer :: output_dim_1
integer :: output_dim_2
integer :: output_dim_3

integer :: err

real*8 input_sum
real*8 tmp_input
real output_sum

! file IO
integer :: ios, fu
integer, parameter :: read_unit = 99

! input and output tensors
real(c_float), allocatable :: it2f(:,:,:,:)
real(c_float), allocatable :: ot2f(:,:,:,:)
real(c_float), allocatable :: ot2f_ref(:,:,:,:)

! Get CL arguments
CALL get_command_argument(1, model_path)
CALL get_command_argument(2, model_type)
CALL get_command_argument(3, input_path)
CALL get_command_argument(4, n_batches_str)
CALL get_command_argument(5, ref_output_path)

! read n batches
read(n_batches_str,*) n_batches
write(*,*) "N batches set to ", n_batches

argc = command_argument_count()
if (argc>5) then
   call get_command_argument(6, tol_str)
   read(tol_str,*) tol
   write(*,*) "Tolerance set to ", tol
endif

! tcyclone model input size [ 1, 200, 200, 17 ]
input_dim_0 = n_batches
input_dim_1 = 200
input_dim_2 = 200
input_dim_3 = 17
input_dim_flat =  input_dim_0 * input_dim_1 * input_dim_2 * input_dim_3

! tcyclone model output size [ 1, 200, 200, 1 ]
output_dim_0 = n_batches
output_dim_1 = 200
output_dim_2 = 200
output_dim_3 = 1


! 0) init infero
call infero_check(infero_initialise())


! Allocate tensors
allocate( it2f(input_dim_0,input_dim_1,input_dim_2,input_dim_3) )
allocate( ot2f(output_dim_0,output_dim_1,output_dim_2,output_dim_3) )
allocate( ot2f_ref(output_dim_0,output_dim_1,output_dim_2,output_dim_3) )

! Read 4D data from sequential CSV (CSV values are in Fortran order)
input_sum = 0
open (action='read', file=TRIM(input_path), iostat=ios, newunit=fu)
if (ios /= 0) stop
do ch = 1,input_dim_3
    do j = 1,input_dim_2
        do i = 1,input_dim_1
          read(fu, *) tmp_input
            do ss = 1,input_dim_0              
              ! write(*,*) "input_dim_0=", input_dim_0 , ", ss=", ss, "  -> idx=", (ss-1)*input_dim_flat + 1
              it2f(ss, i, j, ch) = tmp_input
              input_sum = input_sum + tmp_input
            end do
        end do
    end do
end do
close(fu)

! YAML config string
yaml_config = "---"//NEW_LINE('A') &
  //"  path: "//TRIM(model_path)//NEW_LINE('A') &
  //"  type: "//TRIM(model_type)//c_null_char

! get a infero model
  call infero_check(model%initialise_from_yaml_string(yaml_config))

! run inference
do i=1,n_inference_reps
  call infero_check(model%infer(it2f, ot2f ))
end do

! free the model
  call infero_check(model%free())

! finalise infero library
  call infero_check(infero_finalise())

! check element by element
open (action='read', file=TRIM(ref_output_path), iostat=ios, newunit=fu)
if (ios /= 0) stop
do ch = 1,output_dim_3
    do j = 1,output_dim_2
        do i = 1,output_dim_1
          read(fu, *) tmp_input
            do ss = 1,output_dim_0              
              ot2f_ref(ss, i, j, ch) = tmp_input
            end do
        end do
    end do
end do
close(fu)

do ch = 1,output_dim_3
    do j = 1,output_dim_2
        do i = 1,output_dim_1
            do ss = 1,output_dim_0
                if (abs(ot2f(ss, i, j, ch) - ot2f_ref(ss, i, j, ch)) .gt. tol) then
                    write(*,*) "ERROR: output element ", ss, i, j, ch, " (", ot2f(ss, i, j, ch) ,") ", &
                    "is different from expected value ", ot2f_ref(ss, i, j, ch)
                    stop 1
                end if
            end do
        end do
    end do
end do

deallocate(it2f)
deallocate(ot2f)

end program

