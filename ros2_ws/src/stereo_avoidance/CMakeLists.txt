cmake_minimum_required(VERSION 3.16)
project(stereo_avoidance LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime installation prefix")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_ROOT}/include" "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
  REQUIRED)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
  HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
  REQUIRED)

add_executable(stereo_avoidance_node
  src/stereo_avoidance_node.cpp
  src/depth_policy.cpp
)
target_include_directories(stereo_avoidance_node PRIVATE
  include
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(stereo_avoidance_node ${ONNXRUNTIME_LIBRARY} ${OpenCV_LIBS})
ament_target_dependencies(stereo_avoidance_node
  rclcpp sensor_msgs nav_msgs geometry_msgs cv_bridge)

install(TARGETS stereo_avoidance_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
