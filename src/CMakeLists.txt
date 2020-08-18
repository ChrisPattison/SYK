cmake_minimum_required(VERSION 3.16)
project(DECODER LANGUAGES CXX C CUDA)
find_package(CUDA 10.2 REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

set(CMAKE_CXX_FLAGS "-march=${TARGET_ARCH} -mtune=${TARGET_ARCH} -pedantic-errors -fopenmp")
add_definitions(-DMKL_ILP64)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(FlatBuffers REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Magma REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(MPI REQUIRED COMPONENTS C)
find_package(ELPA REQUIRED)

message("${Magma_ROOT}")
message("${Magma_INCLUDE_DIR}")

message("${ELPA_ROOT}")
message("${ELPA_INCLUDE_DIR}")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU") 
    link_libraries("stdc++fs")
endif()

# ================= Precision =================

set(DIAG_SINGLE_PRECISION OFF CACHE BOOL "Use single precision in GPU diagonalization routine")
if(DIAG_SINGLE_PRECISION)
    add_definitions(-DDIAG_SINGLE_PRECISION)
endif()
message("* Using single precision GPU diagonalization : ${DIAG_SINGLE_PRECISION}")

# ================= Schemas =================

set(flatbuff_files input.fbs output.fbs checkpoint.fbs types.fbs)
set(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS "--python")
build_flatbuffers("${flatbuff_files}" "" gen_flatbuff "" "${CMAKE_CURRENT_BINARY_DIR}/flatbuff" "" "")

# ================= Bench =================

add_executable(bench bench.cpp cuda_diagonalize.cu)

if(OpenMP_CXX_FOUND)
    target_compile_options(bench PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(bench PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

target_include_directories(bench PRIVATE ${Magma_INCLUDE_DIR})
target_include_directories(bench PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(bench PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(bench PRIVATE ${Magma_LIBRARIES})
target_link_libraries(bench PRIVATE CUDA::cusolver)

# ================= Spectral =================

add_executable(spectral spectral.cpp cuda_diagonalize.cu)

if(OpenMP_CXX_FOUND)
    target_compile_options(spectral PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(spectral PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

target_include_directories(spectral PRIVATE ${Magma_INCLUDE_DIR})
target_include_directories(spectral PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(spectral PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(spectral PRIVATE ${Magma_LIBRARIES})
target_link_libraries(spectral PRIVATE CUDA::cusolver)

# ================= Mixer =================

add_executable(mixer mixer.cpp cuda_diagonalize.cu)

if(OpenMP_CXX_FOUND)
    target_compile_options(mixer PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(mixer PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

if(MPI_COMPILE_FLAGS)
  set_target_properties(mixer PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(mixer PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

add_dependencies(mixer gen_flatbuff)
target_include_directories(mixer PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(mixer PRIVATE ${BOOST_INCLUDE_DIR})
target_include_directories(mixer PRIVATE ${Magma_INCLUDE_DIR})
target_include_directories(mixer PRIVATE ${ELPA_INCLUDE_DIR})
target_include_directories(mixer PRIVATE ${FLATBUFFERS_INCLUDE_DIR})
target_include_directories(mixer PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(mixer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(mixer PRIVATE ${MPI_INCLUDE_PATH})

target_link_libraries(mixer PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(mixer PRIVATE ${Magma_LIBRARIES})
target_link_libraries(mixer PRIVATE ${ELPA_LIBRARIES})
target_link_libraries(mixer PRIVATE CUDA::cusolver)
target_link_libraries(mixer PRIVATE ${MPI_LIBRARIES})

# ================= Gen Ham =================

add_executable(gen_ham gen_ham.cpp)

if(OpenMP_CXX_FOUND)
    target_compile_options(gen_ham PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(gen_ham PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

add_dependencies(gen_ham gen_flatbuff)
target_include_directories(gen_ham PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(gen_ham PRIVATE ${FLATBUFFERS_INCLUDE_DIR})
target_include_directories(gen_ham PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(gen_ham PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(gen_ham PRIVATE ${HDF5_LIBRARIES})
# ================= Concat Ham =================

add_executable(concat_ham concat_ham.cpp)

add_dependencies(concat_ham gen_flatbuff)
target_include_directories(concat_ham PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(concat_ham PRIVATE ${FLATBUFFERS_INCLUDE_DIR})
target_include_directories(concat_ham PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(concat_ham PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(concat_ham PRIVATE ${HDF5_LIBRARIES})

# ================= Install ================

install(TARGETS bench spectral mixer gen_ham)
