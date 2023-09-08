cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( fundamental_cpp fundamental_cpp.cpp )
target_link_libraries( fundamental_cpp ${OpenCV_LIBS} )

add_executable( camera_cpp camera_cpp.cpp )
target_link_libraries( camera_cpp ${OpenCV_LIBS} )

