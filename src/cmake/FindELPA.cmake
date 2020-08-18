# Basic CMake script to find ELPA
# Uses ELPA_ROOT to find libraries 
# Sets ELPA_INCLUDE_DIR ELPA_LIBRARIES ELPA_FOUND

if(NOT ELPA_ROOT)
    message(WARNING "ELPA_ROOT not set. Will attempt to find ELPA anyway")
endif()

if(NOT ELPA_LIBRARIES)
    find_library(ELPA_LIBRARIES elpa_openmp elpa ${ELPA_ROOT}/lib)
endif()
mark_as_advanced(ELPA_LIBRARIES)

if(NOT ELPA_INCLUDE_DIR)
    find_path(ELPA_INCLUDE_DIR elpa.h elpa/elpa.h ${ELPA_ROOT}/include)
endif()
mark_as_advanced(ELPA_INCLUDE_DIR)

find_package_handle_standard_args(ELPA DEFAULT_MSG ELPA_INCLUDE_DIR ELPA_LIBRARIES)

