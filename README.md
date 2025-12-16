# unseen_obj_clst_ros2

ROS 2 (Jazzy) wrapper for **Unseen Object Clustering (UOC / UCN)** using a **Docker + subprocess bridge**, supporting RGB-D images and point clouds from Gazebo or recorded data.

This repository provides multiple ROS 2 services that run the original Unseen Object Clustering inference code **inside a Docker container**, while exposing clean ROS 2 interfaces for perception pipelines (e.g., FlexBE, grasp planning).

---

## Overview

This package supports **three segmentation pipelines**:

1. **RGB-D image based segmentation (RECOMMENDED)**  
   - Uses color + depth images from a simulated or real RGB-D camera  
   - Best performance and stability  
   - Fully validated in Gazebo  
   - Server: `segmentation_rgbd_server.py`

2. **RGB-D image test (file-based, standalone)**  
   - Uses saved RGB-D images on disk  
   - Useful for debugging and sanity checks  
   - Server: `segmentation_server.py`

3. **Point-cloud based workaround (NOT recommended)**  
   - Rasterizes point clouds into depth internally  
   - Lower robustness, kept mainly for experimentation  
   - Server: `segmentation_cloud_server.py`

All three servers ultimately call the same **Unseen Object Clustering inference code** via a Docker subprocess.

---


## Architecture  

```text

+--------------------+       +-------------------+      +------------------------------+
|   Gazebo / GZ Sim   | ---> |   ROS-GZ Bridge   | ---> |      ROS 2 Server Node       |
| (RGBD camera sensor)|      | (GZ -> ROS topics)|      |  segmentation_rgbd_server.py |
+--------------------+       +-------------------+      +------------------------------+
   |   /rgbd_camera/image              |                        |
   |   /rgbd_camera/depth_image        |                        |  (save RGB+Depth, intrinsics)
   |   /rgbd_camera/camera_info        |                        v
   |                                   |         +------------------------------------+
   |                                   +-------> |      Docker (unseen_obj env)       |
   |                                             | test_images_segmentation_no_ros.py |
   |                                             |          (UOC inference)           |
   |                                             +------------------------------------+
   |                                                              |
   |                                                              v
   |                                                  +------------------------------+
   |                                                  |  Outputs on shared volume:   |
   |                                                  |  - im_label.npy (instance id)|
   |                                                  |  - segmentation.json         |
   |                                                  +------------------------------+
   |                                                              |
   |                                                              v
   +----------------------------------------------------<---------+
                                    (parse + return via ROS2 service)

+-------------------+                 ^
|   ROS2 Client     | ---------------+
| (service call)    |   /segmentation_rgbd  (SegImage.srv)
+-------------------+

```

Flow:  
1. Gazebo publishes RGB, depth, and camera intrinsics from the RGB-D camera sensor.
2. `ros_gz_bridge` bridges Gazebo topics to ROS 2 topics (`/rgbd_camera/image`, `/rgbd_camera/depth_image`, `/rgbd_camera/camera_info`).
3. A ROS 2 client calls `/segmentation_rgbd` (e.g., `{im_name: 'from_rgbd'}`).
4. `segmentation_rgbd_server.py` subscribes to RGB/depth/camera_info, saves them to a shared input folder, then launches UCN inference inside Docker via `subprocess` and `docker` exec.
5. The Docker-side script (`test_images_segmentation_no_ros.py`) runs Unseen Object Clustering and writes outputs (`im_label.npy`, `segmentation.json`) to a shared output folder.
6. The server parses the outputs and returns results (success flag + JSON + output directory path) to the ROS 2 client.              

---

## Prerequisites

- ROS 2 Jazzy
- Docker with NVIDIA runtime
- A running container named:
  ```
  unseen_obj_container
  ```
- Inside the container:
  - Conda environment: `unseen_obj` (triggered by `unseen_obj_docker/docker_run.sh`)
  - UCN code at:
    ```
    /root/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering
    ```

---

## Services

### 1. RGB-D File-Based Test

**Server**
```
ros2 run unseen_obj_clst_ros2 segmentation_server
```

**Client**
```
ros2 run unseen_obj_clst_ros2 segmentation_client
```

---

### 2. RGB-D Segmentation (Recommended)

**Server**
```
ros2 run unseen_obj_clst_ros2 segmentation_rgbd_server
```

**Service**
```
/segmentation_rgbd   (unseen_obj_clst_ros2/srv/SegImage)
```

**Test call**
```
ros2 service call /segmentation_rgbd unseen_obj_clst_ros2/srv/SegImage "{im_name: 'from_rgbd'}"
```

**Subscribed topics**
- `/rgbd_camera/image`
- `/rgbd_camera/depth_image`
- `/rgbd_camera/camera_info`

**Outputs**
- `segmentation.json`
- `im_label.npy`
- Optional visualization images

---


### 3. Point Cloud Segmentation (Experimental)

Not recommended: Tentative workaround for testing with Point Cloud inputs.

**Server**
```
ros2 run unseen_obj_clst_ros2 segmentation_cloud_server
```

**Client**
```
ros2 run unseen_obj_clst_ros2 segmentation_cloud_client
```

---

## Gazebo + ROS-GZ Bridge

Required topics:
- `/rgbd_camera/image`
- `/rgbd_camera/depth_image`
- `/rgbd_camera/points`
- `/rgbd_camera/camera_info`


---

## Repository Structure

```
unseen_obj_clst_ros2/
├── segmentation_server.py
├── segmentation_client.py
├── segmentation_cloud_server.py
├── segmentation_cloud_client.py
├── segmentation_rgbd_server.py
├── srv/
│   ├── SegImage.srv
│   └── SegCloud.srv
├── docker_run.sh
└── compare_UnseenObjectClustering/
    └── test_images_segmentation_no_ros.py
```

---

## Recommended Usage

Use the RGB-D server when ROS2 topics of `/rgbd_camera` have been published:

```
ros2 run unseen_obj_clst_ros2 segmentation_rgbd_server
```

Then test with:

```
ros2 service call /segmentation_rgbd unseen_obj_clst_ros2/srv/SegImage "{im_name: 'from_rgbd'}"
```

---

## Status

- RGB-D Gazebo → ROS 2 → Docker → UOC segmentation **WORKING**
- Multiple object instances segmented successfully
- Ready for FlexBE and grasp planning integration

---
