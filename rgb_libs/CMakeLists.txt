cmake_minimum_required(VERSION 2.8.3)
project(rgb_libs)


add_compile_options(-std=c++11)

# find system dependencies
find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})
link_directories(${Eigen3_LIBRARY_DIRS})
add_definitions(${Eigen3_DEFINITIONS})

find_package(Boost REQUIRED)

find_package(OpenCV REQUIRED)

# find catkin packages
find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  roscpp
  rospy
  cmake_modules
  geometry_msgs
  sensor_msgs
  eigen_conversions
  tf
  urdf
  cv_bridge
  image_transport
  OpenCV
)

add_message_files(
  FILES
  controllerCheck.msg
)

# add_definitions("-DUSE_ROS")

## Generate services in the 'srv' folder -> this one must be above generate_messages and catkin_package
add_service_files(
  FILES image_cmd.srv
  )

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES geometry_msgs sensor_msgs std_msgs)
# generate_messages(DEPENDENCIES geometry_msgs sensor_msgs std_msgs)


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES pahap
  CATKIN_DEPENDS cmake_modules eigen_conversions geometry_msgs message_runtime roscpp std_msgs 
  DEPENDS Eigen Boost
)
# do not wildcard install files since the root folder of the package will contain a debian folder for releasing

#####################################################################################
# Build
#####################################################################################
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include 
  ${catkin_INCLUDE_DIRS} 
  # ${PCL_INCLUDE_DIRS} 
  ${Boost_INCLUDE_DIRS}
)

add_executable(rgb_libs src/rgb_libs.cpp)
# This line tells catkin to first build the messages in turtlesim before trying to build this node.
# That is because we rely on the turtlesim/Pose message, which becomes a header file after it's compiled
add_dependencies(rgb_libs ${${PPROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} rgb_libs_gencpp)

# This links up the node with the compiled libraries catkin knows about.
# Basically, it allows the node to leverage ROS libraries.
target_link_libraries(rgb_libs ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(rgb_libs_client src/rgb_libs_client.cpp)
add_dependencies(rgb_libs_client ${${PPROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} rgb_libs_gencpp)
target_link_libraries(rgb_libs_client ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})



################################################################################
# Install
################################################################################
catkin_install_python(
  PROGRAMS scripts/bboxServer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# install(TARGETS ${PROJECT_NAME}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# install(PROGRAMS
#   scripts/create_udev_rules
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}  
# )

# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
# )

# install(DIRECTORY launch
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

# install(FILES 99-open-manipulator-cdc.rules
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

################################################################################
# Test
################################################################################
