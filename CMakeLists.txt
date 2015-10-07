cmake_minimum_required(VERSION 3.3)

project(pcl_qr_segmentation)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(PCL 1.7 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(SOURCE_FILES main.cpp QRDetector.h QRDetector.cpp)

add_executable(pcl_qr_segmentation ${SOURCE_FILES})

target_link_libraries(pcl_qr_segmentation ${PCL_LIBRARIES})
target_link_libraries(pcl_qr_segmentation ${OpenCV_LIBS})