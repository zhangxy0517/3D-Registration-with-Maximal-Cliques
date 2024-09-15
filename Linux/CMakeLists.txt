cmake_minimum_required(VERSION 3.22)
project(MAC)

set(CMAKE_CXX_STANDARD 14)
find_package(PCL REQUIRED)
find_package(igraph 0.10.6 EXACT)
if (NOT igraph_FOUND)
    message("igraph version is lower than 0.10.6, memory issues will happen!")
    find_package(igraph 0.9.9 EXACT)
endif()
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${PCL_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})
add_executable(MAC main.cpp desc_dec.cpp funcs.cpp PCR.cpp registration.cpp visualization.cpp Eva.h)

target_link_libraries(MAC ${PCL_LIBRARIES} igraph::igraph)
