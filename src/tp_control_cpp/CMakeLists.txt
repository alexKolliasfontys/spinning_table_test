cmake_minimum_required(VERSION 3.8)
project(tp_control_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_visual_tools REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_msgs REQUIRED)
find_package(lite6_pick_predictor_interfaces REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

include_directories(include)

add_executable(tp_control_node
              src/tp_control_cpp.cpp)
target_include_directories(tp_control_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(tp_control_node PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  tp_control_node
  "rclcpp"
  "moveit_core"
  "moveit_ros_planning"
  "moveit_ros_planning_interface"
  "moveit_visual_tools"
  "tf2_eigen"
  "geometry_msgs"
  "sensor_msgs"
  "tf2_ros"
  "std_msgs"
  "lite6_pick_predictor_interfaces"
  "tf2_geometry_msgs"
)

install(TARGETS tp_control_node
  DESTINATION lib/${PROJECT_NAME})

install (DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME}/
)

install (DIRECTORY config 
  DESTINATION share/${PROJECT_NAME}/
)

install (DIRECTORY include
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
# cmake_minimum_required(VERSION 3.8)
# project(tp_control_cpp)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# # find dependencies
# find_package(ament_cmake REQUIRED)
# find_package(rclcpp REQUIRED)
# find_package(moveit_core REQUIRED)
# find_package(moveit_ros_planning_interface REQUIRED)
# find_package(moveit_visual_tools REQUIRED)
# find_package(tf2_eigen REQUIRED)
# find_package(geometry_msgs REQUIRED)
# find_package(sensor_msgs REQUIRED)
# find_package(tf2_ros REQUIRED)
# find_package(std_msgs REQUIRED)

# include_directories(include)

# add_executable(tp_control_node src/tp_control_cpp.cpp)
# target_include_directories(tp_control_cpp PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)
# target_compile_features(tp_control_cpp PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
# ament_target_dependencies(
#   tp_control_cpp
#   "rclcpp"
#   "moveit_core"
#   "moveit_ros_planning_interface"
#   "moveit_visual_tools"
#   "tf2_eigen"
#   "geometry_msgs"
#   "sensor_msgs"
#   "tf2_ros"
#   "std_msgs"
# )

# install(TARGETS tp_control_cpp
#   DESTINATION lib/${PROJECT_NAME})

# install (DIRECTORY launch 
#   DESTINATION share/${PROJECT_NAME}/
# )

# install (DIRECTORY config 
#   DESTINATION share/${PROJECT_NAME}/
# )

# install (DIRECTORY include
#   DESTINATION share/${PROJECT_NAME}/
# )

# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   # the following line skips the linter which checks for copyrights
#   # comment the line when a copyright and license is added to all source files
#   set(ament_cmake_copyright_FOUND TRUE)
#   # the following line skips cpplint (only works in a git repo)
#   # comment the line when this package is in a git repo and when
#   # a copyright and license is added to all source files
#   set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()

# ament_package()
