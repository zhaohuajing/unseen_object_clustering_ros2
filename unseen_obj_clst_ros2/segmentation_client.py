#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from unseen_obj_clst_ros2.srv import SegImage
import sys
import json

class SegClient(Node):
    def __init__(self):
        super().__init__('seg_client')
        self.cli = self.create_client(SegImage, 'run_segmentation')
        while not self.cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().info('Waiting for service...')
        self.req = SegImage.Request()

    def send_request(self, im_name):
        self.req.image_name = im_name
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

def main(args=None):
    rclpy.init(args=args)
    im_name = sys.argv[1] if len(sys.argv) > 1 else '000000'
    node = SegClient()
    res = node.send_request(im_name)
    node.get_logger().info(f"Requesting segmentation for images: {im_name}-color.png, {im_name}-depth.png")
    if res.success:
        print("Segmentation success.")
        print(json.dumps(json.loads(res.json_result), indent=2))
    else:
        print("Failed:", res.log_output)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
