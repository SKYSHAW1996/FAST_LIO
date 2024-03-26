cmake_minimum_required(VERSION 2.8.3)
project(fast_lio)

# SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")

ADD_COMPILE_OPTIONS(-std=c++14 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set( CMAKE_CXX_FLAGS "-std=c++14 -O3" )

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions -Wl,-rpath=/usr/local/PCL19/lib")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(Eigen3 REQUIRED)
set(PCL_DIR "/usr/local/PCL19/share/pcl-1.9")
find_package(PCL 1.8 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  livox_ros_driver
  message_generation
  eigen_conversions
)

message(>>>>>>>>>>>>>>> EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR})
message(>>>>>>>>>>>>>>> PCL_INCLUDE_DIRS: ${PCL_INCLUDE_DIRS})
include_directories(
  /usr/local/PCL19/include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS} 
  ${PYTHON_INCLUDE_DIRS}
  include
)
link_directories(
  /usr/local/PCL19/lib
)

add_message_files(
  FILES
  Pose6D.msg
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

add_library(fast_lio SHARED
  src/laserMapping.cpp 
  include/ikd-Tree/ikd_Tree_impl.h 
  src/preprocess.cpp
  src/IMU_Processing.hpp
)
target_link_libraries(fast_lio PUBLIC
  ${PCL_LIBRARIES} 
  ${catkin_LIBRARIES} 	
  ${PYTHON_LIBRARIES}
)
add_executable(fastlio_mapping src/node_main.cpp)

target_link_libraries(fastlio_mapping 
	${PCL_LIBRARIES} 
	${catkin_LIBRARIES} 	
	${PYTHON_LIBRARIES}
  fast_lio
)
target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})








