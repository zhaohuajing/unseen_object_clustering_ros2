#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from unseen_obj_clst_ros2.srv import SegImage
import subprocess
import json
import os

class SegService(Node):
    def __init__(self):
        super().__init__('seg_service_docker')
        self.srv = self.create_service(SegImage, 'run_segmentation', self.callback)
        self.docker_name = "unseen_obj_container"
        self.base_dir = "/root/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering"
        self.python_script = "/root/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/test_images_segmentation_no_ros_backup.py"
        # self.python_script = "/root/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/test_images_segmentation_no_ros.py"
        self.input_dir = "/root/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/data/demo"
        # self.input_dir = "/root/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/data/cgn_sample"
        self.output_dir = "/root/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/output/inference_results"
        self.get_logger().info("Segmentation service ready.")

    def callback(self, request, response):
        im_name = request.im_name
        cmd = (
            f"docker exec {self.docker_name} bash -lc "
            # f"docker exec -e MPLBACKEND=Agg -e QT_QPA_PLATFORM=offscreen {self.docker_name} bash -lc "
            f"\"conda run -n unseen_obj python {self.python_script} "
            f"--gpu 0 --network seg_resnet34_8s_embedding "
            # f"--pretrained {self.base_dir}/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_cat_crop_sampling_epoch_16.checkpoint.pth " # BAD EXAMPLE
            f"--pretrained {self.base_dir}/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth " # good
            f"--pretrained_crop {self.base_dir}/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth " # good
            f"--cfg {self.base_dir}/experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml "
            f"--input_dir {self.input_dir} "
            f"--output_dir {self.output_dir} "
            f"--im_name {im_name}\""
        )

        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
            log_output = proc.stdout + "\n" + proc.stderr
            # result_dir = os.path.join(self.output_dir, f"segmentation_cgn_{im_name}")
            result_dir = os.path.join(self.output_dir, f"segmentation_{im_name}")
            json_path = os.path.join(result_dir, "segmentation.json")

            if not os.path.exists(json_path):
                response.success = False
                response.json_result = ""
                response.log_output = log_output
                return response

            with open(json_path, 'r') as f:
                result_json = json.load(f)

            response.success = True
            response.json_result = json.dumps(result_json)
            response.log_output = log_output
            return response

        except Exception as e:
            response.success = False
            response.json_result = ""
            response.log_output = str(e)
            return response


def main(args=None):
    rclpy.init(args=args)
    node = SegService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
