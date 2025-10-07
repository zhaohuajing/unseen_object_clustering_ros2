# unseen_obj_clst_ros2

ROS 2 Jazzy package for Unseen Object Clustering segmentation.

## Features
- ROS 2 service `seg_image` running a Docker-based segmentation network.
- Client-server setup using `SegImage.srv`.
- Tested with Docker container: `unseen_obj_container` and conda env `unseen_obj`.

## Run
```bash
# Start service
ros2 run unseen_obj_clst_ros2 segmentation_server

# Call from client
ros2 run unseen_obj_clst_ros2 segmentation_client
