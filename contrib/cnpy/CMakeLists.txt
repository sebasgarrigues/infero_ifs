find_package(ZLIB REQUIRED)
ecbuild_add_library(

    TARGET cnpy

    INSTALL_HEADERS LISTED

    HEADER_DESTINATION
        ${INSTALL_INCLUDE_DIR}/infero

    SOURCES
        cnpy.h
        cnpy.cpp

    PUBLIC_INCLUDES
       $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
       $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>

    PRIVATE_INCLUDES
        "${ZLIB_INCLUDE_DIRS}"

    PRIVATE_LIBS
        "${ZLIB_LIBRARIES}"
)


