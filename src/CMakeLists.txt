cmake_minimum_required(VERSION 3.16)
project(DECODER LANGUAGES CXX CUDA)
find_package(CUDA 10.2 REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

set(CMAKE_CXX_FLAGS "-march=native -mtune=native -pedantic-errors -fopenmp")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU") 
    link_libraries("stdc++fs")
endif()

# ================= Bench =================

add_executable(bench bench.cpp diagonalize.cu)

if(OpenMP_CXX_FOUND)
    target_compile_options(bench PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(bench PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

target_link_libraries(bench PUBLIC ${PROTOBUF_LIBRARIES} ${LAPACK_LIBRARIES})
target_include_directories(bench PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(bench PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(bench PRIVATE CUDA::cusolver)

# ================= Spectral =================

add_executable(spectral spectral.cpp diagonalize.cu)

if(OpenMP_CXX_FOUND)
    target_compile_options(spectral PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(spectral PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

target_link_libraries(spectral PUBLIC ${LAPACK_LIBRARIES})
target_include_directories(spectral PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(spectral PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(spectral PRIVATE CUDA::cusolver)

# ================= Mixer =================

add_executable(mixer mixer.cpp diagonalize.cu)

if(OpenMP_CXX_FOUND)
    target_compile_options(mixer PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(mixer PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

target_link_libraries(mixer PUBLIC ${LAPACK_LIBRARIES})
target_include_directories(mixer PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(mixer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(mixer PRIVATE CUDA::cusolver)

# ================= Install ================

install(TARGETS bench spectral mixer)