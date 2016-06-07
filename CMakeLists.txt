cmake_minimum_required(VERSION 2.8.12.2)
project(mpnn)

include_directories(include)

set (CMAKE_CXX_FLAGS "-std=c++11")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
		"Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
		FORCE)
endif(NOT CMAKE_BUILD_TYPE)

add_library(libDNN src/ANN.C src/kd_tree.C src/kd_util.C src/kd_split.C src/kd_search.C src/nn.C src/multiann.C)

add_executable(test test/test.C)
target_link_libraries(test libDNN) 

