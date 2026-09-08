cmake_minimum_required(VERSION 3.0.2)
project(mujoco_ros)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)






## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  rospy
  std_msgs
  message_generation
)
find_package(mujoco REQUIRED)    # 查找MuJoCo库
find_package(glfw3 REQUIRED)     # 查找GLFW库
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  RobotState.msg
  ControlCmd.msg
  BoardCmd.msg
  ResidualCmd.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES mujoco_ros
 CATKIN_DEPENDS roscpp roslib rospy std_msgs message_runtime
 DEPENDS mujoco glfw3
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(
# # include
#   ${catkin_INCLUDE_DIRS}
# )

## Declare a C++ library
add_library(pid_controller
  src/ECBC/pid_controller.cpp
)
target_include_directories(pid_controller
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/mujoco_ros
    ${catkin_INCLUDE_DIRS}
)
target_link_libraries(pid_controller ${catkin_LIBRARIES})

add_library(ecbc_controller
  src/ECBC/eso_based_controller.cpp
)
target_include_directories(ecbc_controller
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/mujoco_ros
    ${catkin_INCLUDE_DIRS}
)
target_link_libraries(ecbc_controller pid_controller ${catkin_LIBRARIES})

add_library(data_logger
  src/data_logger.cpp
)
target_include_directories(data_logger
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/mujoco_ros
    ${catkin_INCLUDE_DIRS}
)
target_link_libraries(data_logger ${catkin_LIBRARIES})



## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure


## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(mujoco_ros_node src/mujoco_ros_node.cpp)
add_dependencies(mujoco_ros_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(balance_control_node src/balance_control_node.cpp)
add_dependencies(balance_control_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(balance_boardcmd_node src/balance_boardcmd_node.cpp)
add_dependencies(balance_boardcmd_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_include_directories(mujoco_ros_node 
  PRIVATE 
   include/mujoco_ros
   ${catkin_INCLUDE_DIRS}
  )
target_link_libraries(mujoco_ros_node
  ${catkin_LIBRARIES}
  mujoco
  glfw3
  dl
  Threads::Threads
  yaml-cpp
  data_logger
)

target_include_directories(balance_control_node
  PRIVATE 
   include/mujoco_ros
   ${catkin_INCLUDE_DIRS}
  )
target_link_libraries(balance_control_node
  ${catkin_LIBRARIES}
  ecbc_controller
  yaml-cpp
  data_logger
)

target_include_directories(balance_boardcmd_node
  PRIVATE 
   ${catkin_INCLUDE_DIRS}
  )
target_link_libraries(balance_boardcmd_node
  ${catkin_LIBRARIES}
)

