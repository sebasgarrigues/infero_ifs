# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

########################################################################################################################

cmake_minimum_required( VERSION 3.16 FATAL_ERROR )

find_package( ecbuild 3.6 REQUIRED HINTS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../ecbuild)

project( infero LANGUAGES CXX )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

########################################################################################################################
### dependencies and options

set( PERSISTENT_NAMESPACE "eckit" CACHE INTERNAL "" ) # needed for generating .b files for persistent support

# eckit always required
ecbuild_find_package( NAME eckit  VERSION  1.19 REQUIRED )

# for cnpy
find_package(ZLIB REQUIRED) 


############## FORTRAN
ecbuild_add_option( FEATURE FORTRAN
                    DEFAULT ON
                    DESCRIPTION "Support for fckit Fortran wrappers"
                    REQUIRED_PACKAGES "NAME fckit VERSION 0.11" )

############## TFlite
ecbuild_add_option( FEATURE TF_LITE
                    DEFAULT OFF
                    DESCRIPTION "TensorflowLite engine"
                    REQUIRED_PACKAGES TensorflowLite )

############## TF_C
ecbuild_add_option( FEATURE TF_C
                    DEFAULT OFF
                    DESCRIPTION "Tensorflow C engine"
                    REQUIRED_PACKAGES TensorflowC )

############## ONNX
ecbuild_add_option( FEATURE ONNX
                    DEFAULT OFF
                    DESCRIPTION "ONNX engine"
                    REQUIRED_PACKAGES ONNX )

############## TensorRT
ecbuild_add_option( FEATURE TENSORRT
                    DEFAULT OFF
                    DESCRIPTION "TensorRT engine"
                    REQUIRED_PACKAGES TensorRT )

############## Option for CL tools
ecbuild_add_option( FEATURE TOOLS
                    DEFAULT ON
                    DESCRIPTION "Build the command line tools" )

############## Option for Examples
ecbuild_add_option( FEATURE EXAMPLES
                    DEFAULT ON
                    DESCRIPTION "Build the infero examples" )

############## Use infero-run for tests
ecbuild_add_option( FEATURE INFERO_RUN
                    DEFAULT ON
                    DESCRIPTION "Use infero/tools/infero-run to run tests" )


if( HAVE_INFERO_RUN )
  set( MPIEXEC_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/tools/infero-run )
  set( MPIEXEC_NUMPROC_FLAG "-n" )
  set( CMAKE_CROSSCOMPILING_EMULATOR ${CMAKE_CURRENT_SOURCE_DIR}/tools/infero-run )
endif()                    


if(HAVE_FORTRAN)
    ecbuild_enable_fortran( REQUIRED MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/module )
endif()


if (HAVE_FORTRAN)
    set(infero_HAVE_FINAL ${fckit_HAVE_FINAL})
    set(infero_HAVE_FORTRAN_APIS TRUE)
    set(infero_LIBRARIES "MLclustering inferoapi inferof")
else()
    set(infero_HAVE_FINAL 0)
    set(infero_HAVE_FORTRAN_APIS FALSE)
    set(infero_LIBRARIES "MLclustering inferoapi")
endif()

ecbuild_info("TENSORFLOWLITE_FOUND ${TENSORFLOWLITE_FOUND}")
ecbuild_info("TENSORFLOWLITE_LIBRARIES ${TENSORFLOWLITE_LIBRARIES}")
ecbuild_info("TENSORFLOWLITE_INCLUDE_DIRS ${TENSORFLOWLITE_INCLUDE_DIRS}")

ecbuild_info("TENSORFLOWC_FOUND ${TENSORFLOWC_FOUND}")
ecbuild_info("TENSORFLOWC_LIBRARIES ${TENSORFLOWC_LIBRARIES}")
ecbuild_info("TENSORFLOWC_INCLUDE_DIRS ${TENSORFLOWC_INCLUDE_DIRS}")

ecbuild_info("ONNX_FOUND ${ONNX_FOUND}")
ecbuild_info("ONNX_LIBRARIES ${ONNX_LIBRARIES}")
ecbuild_info("ONNX_INCLUDE_DIRS ${ONNX_INCLUDE_DIRS}")

ecbuild_info("TENSORRT_FOUND ${TENSORRT_FOUND}")
ecbuild_info("TENSORRT_LIBRARIES ${TENSORRT_LIBRARIES}")
ecbuild_info("TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIRS}")

ecbuild_info("FCKIT_FOUND ${fckit_FOUND}")
ecbuild_info("FCKIT_LIBRARIES ${FCKIT_LIBRARIES}")
ecbuild_info("FCKIT_INCLUDE_DIRS ${FCKIT_INCLUDE_DIRS}")
ecbuild_info("---> fckit_HAVE_FINAL ${HAVE_FINAL}")


### infero version header
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" INFERO_VERSION_STR)
    string(STRIP "${INFERO_VERSION_STR}" INFERO_VERSION_STR)
    string(REPLACE "." ";" INFERO_VERSION_FULL "${INFERO_VERSION_STR}")

    list( GET INFERO_VERSION_FULL 0 INFERO_VERSION_MAJOR )
    list( GET INFERO_VERSION_FULL 1 INFERO_VERSION_MINOR )
    list( GET INFERO_VERSION_FULL 2 INFERO_VERSION_PATCH )

    ecbuild_configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/infero_version.h.in"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/infero_version.h" @ONLY
    )

    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/infero_version.h"
        TYPE INCLUDE
    )

else()
    message(FATAL_ERROR "Version file ${CMAKE_CURRENT_SOURCE_DIR}/VERSION NOT FOUND!")
endif()


include(cmake/compiler_warnings.cmake) # optionally handle compiler specific warnings

add_subdirectory( contrib )
add_subdirectory( src )
add_subdirectory( tests )
add_subdirectory( examples )


############################################################################################
# finalize

ecbuild_install_project( NAME ${PROJECT_NAME} )

ecbuild_print_summary()
