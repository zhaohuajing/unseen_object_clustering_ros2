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

ROS 2 Node  
└── Service callback  
    └── Save RGB / Depth / Cloud to disk  
        └── docker exec unseen_obj_container  
            └── test_images_segmentation_no_ros.py  
                └── UCN / UOC inference  

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

⚠ Not recommended: Tentative workaround for Point Cloud inputs.

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

- RGB-D Gazebo → ROS 2 → Docker → UCN → segmentation **WORKING**
- Multiple object instances segmented successfully
- Ready for FlexBE and grasp planning integration

---
