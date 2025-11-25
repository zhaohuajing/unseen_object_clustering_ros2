#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import PointCloud2, CameraInfo
from unseen_obj_clst_ros2.srv import SegCloud
import json


# sample command:
'''
ros2 run unseen_obj_clst_ros2 segmentation_cloud_client \
  --ros-args -p cloud_topic:=/camera/depth/points -p camera_info_topic:=/camera/color/camera_info \
             -p service_name:=run_segmentation_cloud -p im_name:=from_cloud
'''


class SegCloudClient(Node):
    """
    Minimal one-shot client:
      1) waits for one PointCloud2 on `cloud_topic`
      2) (optionally) waits for one CameraInfo on `camera_info_topic`
      3) calls /run_segmentation_cloud and prints the result_dir
    """
    def __init__(self):
        super().__init__('segmentation_cloud_client')
        self.declare_parameter('cloud_topic', '/rgbd_camera/points')
        self.declare_parameter('camera_info_topic', '/rgbd_camera/camera_info')
        self.declare_parameter('service_name', '/segmentation_cloud_server')
        self.declare_parameter('im_name', 'from_cloud')

        self.cloud_topic = self.get_parameter('cloud_topic').get_parameter_value().string_value
        self.cinfo_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.service_name = self.get_parameter('service_name').get_parameter_value().string_value
        self.im_name = self.get_parameter('im_name').get_parameter_value().string_value

        self._cloud_msg = None
        self._cinfo_msg = None

        # Subscribers
        self._sub_cloud = self.create_subscription(PointCloud2, self.cloud_topic, self._on_cloud, 1)
        self._sub_cinfo = self.create_subscription(CameraInfo, self.cinfo_topic, self._on_cinfo, 1)

        # Service client
        self._cli = self.create_client(SegCloud, self.service_name)
        self.get_logger().info(
            f"Waiting for service '{self.service_name}' and one message on:\n"
            f"  cloud: {self.cloud_topic}\n  camera_info: {self.cinfo_topic} (optional)"
        )

        self._timer = self.create_timer(0.2, self._maybe_call_service)

    def _on_cloud(self, msg: PointCloud2):
        self._cloud_msg = msg

    def _on_cinfo(self, msg: CameraInfo):
        self._cinfo_msg = msg

    def _maybe_call_service(self):
        if self._cloud_msg is None:
            return
        if not self._cli.service_is_ready():
            self.get_logger().warn("Service not ready yet…")
            return

        req = SegCloud.Request()
        req.cloud = self._cloud_msg
        # CameraInfo is optional—send it if we have one
        if self._cinfo_msg is not None:
            req.cam_info = self._cinfo_msg
        # Include an image name if your server supports it
        try:
            req.im_name = self.im_name  # if your .srv has this field
        except Exception:
            pass

        self.get_logger().info("Calling /segmentation_cloud_server …")
        future: Future = self._cli.call_async(req)
        future.add_done_callback(self._on_done)
        # prevent multiple calls
        self._timer.cancel()

    def _on_done(self, fut: Future):
        if fut.cancelled() or fut.exception() is not None:
            self.get_logger().error(f"Service call failed: {fut.exception()}")
        else:
            res = fut.result()
            if not res.success:
                self.get_logger().error(f"Segmentation failed:\n{res.log_output}")
            else:
                try:
                    seg_json = json.loads(res.json_result)
                    self.get_logger().info(f"Success. result_dir: {seg_json.get('result_dir','<none>')}")
                except Exception:
                    self.get_logger().info(f"Success. Raw JSON: {res.json_result}")
        rclpy.shutdown()

def main():
    rclpy.init()
    node = SegCloudClient()
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
