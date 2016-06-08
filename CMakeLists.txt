cmake_minimum_required(VERSION 2.8.12.2)
project(mpnn)

include_directories(include)

set (CMAKE_CXX_FLAGS "-std=c++11")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
		"Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
		FORCE)
endif(NOT CMAKE_BUILD_TYPE)

add_library(DNN src/ANN.cpp src/kd_tree.cpp src/kd_util.cpp src/kd_split.cpp src/kd_search.cpp src/nn.cpp src/multiann.cpp)

add_subdirectory(test)
