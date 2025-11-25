cmake_minimum_required(VERSION 3.5)
project(unseen_obj_clst_ros2)

# Core dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)   # <-- MISSING in your previous file
find_package(std_msgs REQUIRED)

# Generate service interface
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SegImage.srv"
  "srv/SegCloud.srv"
  DEPENDENCIES sensor_msgs std_msgs
)
ament_export_dependencies(rosidl_default_runtime)

# Install your Python executables (ROS 2 nodes)
install(PROGRAMS
  unseen_obj_clst_ros2/segmentation_server.py
  unseen_obj_clst_ros2/segmentation_client.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  unseen_obj_clst_ros2/segmentation_server.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME segmentation_server
)

install(PROGRAMS
  unseen_obj_clst_ros2/segmentation_cloud_server.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME segmentation_cloud_server
)

install(PROGRAMS
  unseen_obj_clst_ros2/segmentation_client.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME segmentation_client
)

install(PROGRAMS
  unseen_obj_clst_ros2/segmentation_cloud_client.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME segmentation_cloud_client
)

# (Optional) include resource files like launch or config
# install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

ament_package()
