# Basic CMake script to find ScaLAPACK
# Uses ScaLAPACK_ROOT to find libraries 
# Sets ScaLAPACK_LIBRARIES ScaLAPACK_FOUND

if(NOT ScaLAPACK_ROOT)
    message(WARNING "ScaLAPACK_ROOT not set. Will attempt to find ScaLAPACK anyway")
endif()

if(NOT ScaLAPACK_LIBRARIES)
    find_library(ScaLAPACK_LIBRARIES ScaLAPACK ${ScaLAPACK_ROOT}/lib)
endif()
mark_as_advanced(ScaLAPACK_LIBRARIES)


find_package_handle_standard_args(ScaLAPACK DEFAULT_MSG ScaLAPACK_LIBRARIES)


