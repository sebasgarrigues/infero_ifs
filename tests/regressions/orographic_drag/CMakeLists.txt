# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ----------------- Runners ----------------
ecbuild_add_executable( TARGET runner_oro_drag
   SOURCES   runner_oro_drag.c
   CONDITION HAVE_TOOLS
   INCLUDES  ${eckit_INCLUDE_DIRS}
   LIBS      inferoapi
   NOINSTALL
)
set_target_properties( runner_oro_drag
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

if(infero_HAVE_FORTRAN_APIS)
    ecbuild_add_executable( TARGET runner_oro_drag_f90
       SOURCES   runner_oro_drag.F90
       CONDITION HAVE_TOOLS
       INCLUDES  ${eckit_INCLUDE_DIRS}
       LIBS      inferof
       NOINSTALL
    )
    set_target_properties( runner_oro_drag_f90
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# ----------------- tests ------------------
list(APPEND tests_types_ "")
list(APPEND tests_models_ "")

if (HAVE_ONNX)
    list(APPEND tests_types_ onnx)
    list(APPEND tests_models_ model.onnx)
endif()

if (HAVE_TF_C)
    list(APPEND tests_types_ tf_c)
    list(APPEND tests_models_ model_tf)
endif()

if (HAVE_TF_LITE)
    list(APPEND tests_types_ tflite)
    list(APPEND tests_models_ model.tflite)
endif()

if (HAVE_TENSORRT)
    list(APPEND tests_types_ tensorrt)
    list(APPEND tests_models_ model.trt)
endif()

message("ORO_DRAG test models: ${tests_models_}")

list( LENGTH tests_types_ _count )
math( EXPR _count "${_count}-1" )

if(NOT ${_count} EQUAL "-1")
    foreach( _i RANGE ${_count} )

        list( GET tests_types_ ${_i} type_ )
        list( GET tests_models_ ${_i} model_ )

        # C-runner
        SET(runner_exe "runner_oro_drag")
        ecbuild_configure_file(
            test_oro_drag_template.sh.in
            test_oro_drag_${type_}_c.sh @ONLY
        )

        ecbuild_add_test(
            TYPE     SCRIPT
            COMMAND  test_oro_drag_${type_}_c.sh
        )

        # F90-runner
        if(infero_HAVE_FORTRAN_APIS)
            SET(runner_exe "runner_oro_drag_f90")
            ecbuild_configure_file(
                test_oro_drag_template.sh.in
                test_oro_drag_${type_}_f90.sh @ONLY
            )

            ecbuild_add_test(
                TYPE     SCRIPT
                COMMAND  test_oro_drag_${type_}_f90.sh
            )
        endif()
    endforeach()
endif()








