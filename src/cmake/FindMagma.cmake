# Basic CMake script to find Magma
# Uses Magma_ROOT to find libraries 
# Sets Magma_INCLUDE_DIR Magma_LIBRARIES Magma_FOUND

if(NOT Magma_ROOT)
    message(WARNING "Magma_ROOT not set. Will attempt to find Magma anyway")
endif()

if(NOT Magma_LIBRARIES)
    find_library(Magma_LIBRARIES magma ${Magma_ROOT}/lib)
endif()
mark_as_advanced(Magma_LIBRARIES)

if(NOT Magma_INCLUDE_DIR)
    find_path(Magma_INCLUDE_DIR magma.h ${Magma_ROOT}/include)
endif()
mark_as_advanced(Magma_INCLUDE_DIR)

find_package_handle_standard_args(Magma DEFAULT_MSG Magma_INCLUDE_DIR Magma_LIBRARIES)

