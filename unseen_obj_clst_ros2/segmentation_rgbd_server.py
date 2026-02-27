#!/usr/bin/env python3
import os
import json
import subprocess
from typing import Optional

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import numpy as np

from unseen_obj_clst_ros2.srv import SegImage  # request has: string im_name


# Note: run "ros2 service call /segmentation_rgbd unseen_obj_clst_ros2/srv/SegImage "{im_name: 'from_rgbd'}" in another terminal

class SegImageService(Node):
    """
    segmentation_rgbd_server:
      - Subscribes to /rgbd_camera/image, /rgbd_camera/depth_image, /rgbd_camera/camera_info
      - On service request, saves latest RGB + depth to PNGs
      - Calls UCN test_images_segmentation_no_ros.py inside Docker
      - Returns segmentation.json contents and logs
    """

    def __init__(self):
        super().__init__("segmentation_rgbd_server")

        # Parameters to control Docker + UCN paths; adjust defaults as needed.
        self.declare_parameter("docker_container", "unseen_obj_container")
        self.declare_parameter("conda_env", "unseen_obj")
        self.declare_parameter(
            "base_dir",
            "/root/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering",
        )
        self.declare_parameter("gpu_id", 0)
        self.declare_parameter("network_name", "seg_resnet34_8s_embedding")
        self.declare_parameter(
            "pretrained",
            "/root/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth",
        )
        # If you have no crop model, leave this empty or unset
        self.declare_parameter(
            "pretrained_crop", 
            "/root/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth"
            )
        self.declare_parameter(
            "cfg",
            "/root/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml",
        )
        # Directories inside the *container* where test_images_segmentation_no_ros.py reads/writes
        self.declare_parameter(
            "input_dir_in_container",
            "/root/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/segmentation_rgbd/input",
        )
        self.declare_parameter(
            "output_dir_in_container",
            "/root/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/segmentation_rgbd/output",
        )

        # Read parameters
        self.container = self.get_parameter("docker_container").get_parameter_value().string_value
        self.conda_env = self.get_parameter("conda_env").get_parameter_value().string_value
        self.base_dir = self.get_parameter("base_dir").get_parameter_value().string_value
        self.gpu_id = self.get_parameter("gpu_id").get_parameter_value().integer_value
        self.network_name = self.get_parameter("network_name").get_parameter_value().string_value
        self.pretrained = self.get_parameter("pretrained").get_parameter_value().string_value
        self.pretrained_crop = self.get_parameter("pretrained_crop").get_parameter_value().string_value
        self.cfg = self.get_parameter("cfg").get_parameter_value().string_value
        self.input_dir_in_container = (
            self.get_parameter("input_dir_in_container").get_parameter_value().string_value
        )
        self.output_dir_in_container = (
            self.get_parameter("output_dir_in_container").get_parameter_value().string_value
        )

        # Script path in container
        self.script_in_container = os.path.join(
            self.base_dir, "test_images_segmentation_no_ros.py"
        )

        # Local mirror dirs (bind-mounted into container)
        # These paths are on the host; they must be bind-mounted to the corresponding
        # *input_dir_in_container* and *output_dir_in_container* in your Docker setup.
        self.local_input_dir = (
            os.path.expanduser("~/graspnet_ws/src/unseen_obj_clst_ros2/")
            "compare_UnseenObjectClustering/segmentation_rgbd/input"
        )
        self.local_output_dir = (
            os.path.expanduser("~/graspnet_ws/src/unseen_obj_clst_ros2/")
            "compare_UnseenObjectClustering/segmentation_rgbd/output"
        )

        os.makedirs(self.local_input_dir, exist_ok=True)
        os.makedirs(self.local_output_dir, exist_ok=True)

        self.bridge = CvBridge()

        # Latest RGB, depth, camera_info
        self.latest_rgb: Optional[Image] = None
        self.latest_depth: Optional[Image] = None
        self.latest_cam_info: Optional[CameraInfo] = None

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, "/rgbd_camera/image", self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, "/rgbd_camera/depth_image", self.depth_callback, 10
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo, "/rgbd_camera/camera_info", self.cam_info_callback, 10
        )

        # Service
        self.srv = self.create_service(SegImage, "segmentation_rgbd", self.handle_segmentation)

        self.get_logger().info("SegImageService (RGBD→UCN via Docker) ready.")

    # -------------------------------------------------------------------------
    # Subscriptions
    # -------------------------------------------------------------------------
    def rgb_callback(self, msg: Image):
        self.latest_rgb = msg

    def depth_callback(self, msg: Image):
        self.latest_depth = msg

    def cam_info_callback(self, msg: CameraInfo):
        self.latest_cam_info = msg

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _ensure_latest_data(self) -> bool:
        if self.latest_rgb is None:
            self.get_logger().warn("No RGB image received yet.")
            return False
        if self.latest_depth is None:
            self.get_logger().warn("No depth image received yet.")
            return False
        if self.latest_cam_info is None:
            self.get_logger().warn("No camera_info received yet.")
            return False
        return True

    def _docker_exec(self, cmd_inner: str) -> subprocess.CompletedProcess:
        """
        Execute a command inside the Docker container & conda env.
        """
        cmd = (
            f"docker exec {self.container} bash -lc "
            f"\"source /opt/conda/etc/profile.d/conda.sh && conda activate {self.conda_env} && {cmd_inner}\""
        )
        self.get_logger().info(f"[SegImage] Running in container: {cmd_inner}")
        return subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=1200,
        )

    def _save_rgbd_to_disk(self, im_name: str):
        """
        Convert latest RGB + depth to PNG files on the host filesystem.
        - color: uint8 BGR PNG
        - depth: uint16 mm PNG
        """
        rgb_msg = self.latest_rgb
        depth_msg = self.latest_depth

        import cv2

        # Convert RGB (assume encoding is rgb8 or bgr8; CvBridge handles it)
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        color_path = os.path.join(self.local_input_dir, f"{im_name}-color.png")
        cv2.imwrite(color_path, cv_rgb)

        # Convert depth: assume 32FC1 in meters (Gazebo)
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_m = cv_depth.astype(np.float32)
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        depth_path = os.path.join(self.local_input_dir, f"{im_name}-depth.png")
        cv2.imwrite(depth_path, depth_mm)

        self.get_logger().info(f"[SegImage] Saved color to {color_path}")
        self.get_logger().info(f"[SegImage] Saved depth to {depth_path}")

    def _run_ucn_inference(self, im_name: str, fx: float, fy: float, px: float, py: float):
        """
        Build the inner python command and run UCN inference inside Docker.
        NOTE: only pass --pretrained_crop if a non-empty value is configured.
        """
        # Build the inner command as a list and then join, to make it easier to
        # conditionally append arguments.
        parts = [
            "python", self.script_in_container,
            "--gpu", str(self.gpu_id),
            "--network", self.network_name,
            "--pretrained", self.pretrained,
            "--cfg", self.cfg,
            "--input_dir", self.input_dir_in_container,
            "--output_dir", self.output_dir_in_container,
            "--im_name", im_name,
            "--fx", str(fx),
            "--fy", str(fy),
            "--px", str(px),
            "--py", str(py),
        ]

        # Only add --pretrained_crop if non-empty
        if self.pretrained_crop:
            parts.extend(["--pretrained_crop", self.pretrained_crop])

        cmd_inner = " ".join(parts)
        return self._docker_exec(cmd_inner)

    # -------------------------------------------------------------------------
    # Service callback
    # -------------------------------------------------------------------------
    def handle_segmentation(self, req: SegImage.Request, resp: SegImage.Response):
        """
        Handle SegImage:
          req.im_name: base name for saved RGBD & segmentation result.
        """
        im_name = req.im_name if req.im_name else "from_rgbd"

        if not self._ensure_latest_data():
            resp.success = False
            resp.json_result = ""
            resp.log_output = "No RGBD data available."
            resp.result_dir = self.local_output_dir
            return resp

        # Extract intrinsics from camera_info
        K = np.array(self.latest_cam_info.k, dtype=np.float32).reshape(3, 3)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        px = float(K[0, 2])
        py = float(K[1, 2])
        self.get_logger().info(
            f"[SegImage] Intrinsics: fx={fx:.3f}, fy={fy:.3f}, px={px:.3f}, py={py:.3f}"
        )

        # Save RGBD PNGs
        self._save_rgbd_to_disk(im_name)

        # Run inference inside Docker
        try:
            proc = self._run_ucn_inference(im_name, fx, fy, px, py)
        except Exception as e:
            resp.success = False
            resp.json_result = ""
            resp.log_output = f"error running docker: {e}"
            resp.result_dir = self.local_output_dir
            return resp

        stdout_txt = proc.stdout or ""
        stderr_txt = proc.stderr or ""
        log_output = stdout_txt + ("\n" if stdout_txt and stderr_txt else "") + stderr_txt

        if proc.returncode != 0:
            self.get_logger().error(
                f"[SegImage] UCN inference failed (returncode={proc.returncode})"
            )
            resp.success = False
            resp.json_result = ""
            resp.log_output = f"returncode={proc.returncode}\n{log_output}"
            resp.result_dir = self.local_output_dir
            return resp

        # Load segmentation.json from output_dir/segmentation_<im_name>/
        seg_dir = os.path.join(self.local_output_dir, f"segmentation_{im_name}")
        seg_json_path = os.path.join(seg_dir, "segmentation.json")

        if not os.path.exists(seg_json_path):
            # Fallback: look for a flat segmentation.json in output_dir
            alt_path = os.path.join(self.local_output_dir, "segmentation.json")
            if os.path.exists(alt_path):
                seg_json_path = alt_path

        if not os.path.exists(seg_json_path):
            self.get_logger().error(
                f"[SegImage] segmentation.json not found under {seg_dir} or {self.local_output_dir}"
            )
            resp.success = False
            resp.json_result = ""
            resp.log_output = "segmentation.json missing.\n" + log_output
            resp.result_dir = self.local_output_dir
            return resp

        with open(seg_json_path, "r") as f:
            seg_data = json.load(f)

        # Optionally attach result_dir and base_output_dir (container paths)
        seg_data["result_dir"] = self.output_dir_in_container
        seg_data["base_output_dir"] = self.output_dir_in_container

        resp.success = True
        resp.json_result = json.dumps(seg_data)
        resp.log_output = log_output
        resp.result_dir = self.output_dir_in_container

        self.get_logger().info(
            f"[SegImage] Segmentation success. result_dir={self.output_dir_in_container}"
        )
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = SegImageService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
