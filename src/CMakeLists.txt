cmake_minimum_required(VERSION 3.10)
project(DECODER LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP)
find_package(Protobuf REQUIRED)
#set(PROTO_FILES cluster_observables.proto)
#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
#protobuf_generate_python(PROTO_PY ${PROTO_FILES})

#find_package(LAPACK REQUIRED)
find_package(Eigen3 REQUIRED)

# ================= Bench =================

add_executable(bench bench.cpp ${PROTO_SRCS} ${PROTO_HDRS})
set_property(TARGET bench PROPERTY CXX_STANDARD 17)
target_compile_options(bench PUBLIC "-march=native")
target_compile_options(bench PUBLIC "-mtune=native")

if(OpenMP_CXX_FOUND)
    target_compile_options(bench PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(bench PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

target_link_libraries(bench PUBLIC ${PROTOBUF_LIBRARIES} ${LAPACK_LIBRARIES})
target_include_directories(bench PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_include_directories(bench PRIVATE ${PROTOBUF_INCLUDE_DIRS})
target_include_directories(bench PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# ================= Spectral =================

add_executable(spectral spectral.cpp ${PROTO_SRCS} ${PROTO_HDRS})
set_property(TARGET spectral PROPERTY CXX_STANDARD 17)
target_compile_options(spectral PUBLIC "-march=native")
target_compile_options(spectral PUBLIC "-mtune=native")
target_compile_options(spectral PUBLIC "-pedantic-errors")

if(OpenMP_CXX_FOUND)
    target_compile_options(spectral PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(spectral PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

target_link_libraries(spectral PUBLIC ${PROTOBUF_LIBRARIES} ${LAPACK_LIBRARIES})
target_include_directories(spectral PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_include_directories(spectral PRIVATE ${PROTOBUF_INCLUDE_DIRS})
target_include_directories(spectral PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# ================= Mixer =================

add_executable(mixer mixer.cpp ${PROTO_SRCS} ${PROTO_HDRS})
set_property(TARGET mixer PROPERTY CXX_STANDARD 17)
target_compile_options(mixer PUBLIC "-march=native")
target_compile_options(mixer PUBLIC "-mtune=native")
target_compile_options(mixer PUBLIC "-pedantic-errors")

if(OpenMP_CXX_FOUND)
    target_compile_options(mixer PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(mixer PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

target_link_libraries(mixer PUBLIC ${PROTOBUF_LIBRARIES} ${LAPACK_LIBRARIES})
target_include_directories(mixer PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_include_directories(mixer PRIVATE ${PROTOBUF_INCLUDE_DIRS})
target_include_directories(mixer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# ================= TESTS =================

# Build GoogleTest
# This is from the GoogleTest readme

configure_file(GTestCMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# Build Tests

include(GoogleTest)
add_executable(tests test_syk.cpp)
set_property(TARGET tests PROPERTY CXX_STANDARD 17)
gtest_discover_tests(tests)
target_link_libraries(tests gtest_main)
