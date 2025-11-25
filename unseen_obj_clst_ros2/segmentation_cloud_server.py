#!/usr/bin/env python3
import os, json, subprocess, numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
from unseen_obj_clst_ros2.srv import SegCloud
import imageio.v2 as imageio  # writes png

def pc2_to_xyzrgb(cloud: PointCloud2):
    """Return (N,6)[x,y,z,r,g,b] if rgb exists, else (N,3)."""
    fields = [f.name for f in cloud.fields]
    have_rgb = ('rgb' in fields) or ('rgba' in fields)
    out = []
    if have_rgb:
        for x,y,z,rgb in pc2.read_points(cloud, field_names=("x","y","z","rgb"), skip_nans=False):
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)) or z <= 0:
                continue
            u8 = np.frombuffer(np.float32(rgb).tobytes(), dtype=np.uint8)
            r,g,b = int(u8[2]), int(u8[1]), int(u8[0])  # packed as B,G,R,?
            out.append([x,y,z,r,g,b])
        return np.asarray(out, dtype=np.float32), True
    else:
        for x,y,z in pc2.read_points(cloud, field_names=("x","y","z"), skip_nans=False):
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)) or z <= 0:
                continue
            out.append([x,y,z])
        return np.asarray(out, dtype=np.float32), False

def intrinsics_from_request(cam: CameraInfo, fallback_json: str):
    """Prefer CameraInfo; else read camera_params.json {fx,fy,x_offset,y_offset,height?,width?}."""
    if isinstance(cam, CameraInfo) and len(cam.k) == 9 and any(cam.k):
        K = np.array(cam.k, dtype=np.float32).reshape(3,3)
        H = cam.height if cam.height > 0 else 480
        W = cam.width  if cam.width  > 0 else 640
        return K, H, W
    with open(fallback_json, "r") as f:
        d = json.load(f)
    K = np.array([[d["fx"], 0., d["x_offset"]],
                  [0., d["fy"], d["y_offset"]],
                  [0., 0., 1.]], dtype=np.float32)
    H = int(d.get("height", 480)); W = int(d.get("width", 640))
    return K, H, W

# def rasterize_xyzrgb_to_images(xyzrgb: np.ndarray, K: np.ndarray, H: int, W: int, have_rgb: bool):
#     """
#     Project (N,3|6) to:
#       color  -> (H,W,3) uint8 BGR
#       depth  -> (H,W)   uint16 millimeters
#     Uses a simple z-buffer (nearest wins).
#     """
#     color = np.zeros((H, W, 3), dtype=np.uint8)
#     depth_mm = np.zeros((H, W), dtype=np.uint16)

#     X, Y, Z = xyzrgb[:,0], xyzrgb[:,1], xyzrgb[:,2]
#     u = np.round((K[0,0] * (X / Z) + K[0,2])).astype(np.int32)
#     v = np.round((K[1,1] * (Y / Z) + K[1,2])).astype(np.int32)
#     valid = (Z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

#     # Initialize z-buffer in meters; store depth_mm as we commit pixels
#     zbuf = np.full((H, W), np.inf, dtype=np.float32)
#     idxs = np.where(valid)[0]
#     for i in idxs:
#         ui, vi, zi = u[i], v[i], Z[i]
#         if zi < zbuf[vi, ui]:
#             zbuf[vi, ui] = zi
#             depth_mm[vi, ui] = int(np.clip(zi * 1000.0, 0, 65535))
#             if have_rgb and xyzrgb.shape[1] >= 6:
#                 # we saved RGB in order r,g,b; but the image expected by your script is *BGR* (OpenCV default)
#                 r,g,b = xyzrgb[i,3:6].astype(np.uint8)
#                 color[vi, ui] = np.array([b,g,r], dtype=np.uint8)

#     return color, depth_mm


def rasterize_xyzrgb_to_images(xyzrgb: np.ndarray, K: np.ndarray, H: int, W: int, have_rgb: bool):
    """
    Project (N,3|6) to:
      color  -> (H,W,3) uint8 BGR
      depth  -> (H,W)   uint16 millimeters

    If RGB is missing or effectively all zeros, synthesize a grayscale
    "pseudo-RGB" from depth so the segmentation network still gets some
    structure in the color channel.
    """
    color = np.zeros((H, W, 3), dtype=np.uint8)
    depth_mm = np.zeros((H, W), dtype=np.uint16)

    # --- reinterpret point cloud axes ---
    # incoming xyz roughly: X = forward, Y = left, Z = up
    # convert to camera optical frame: X_cam = right, Y_cam = down, Z_cam = forward
    X_raw = xyzrgb[:, 0]
    Y_raw = xyzrgb[:, 1]
    Z_raw = xyzrgb[:, 2]

    Z = X_raw             # depth
    X = -Y_raw            # right
    Y = -Z_raw            # down

    # optional near/far clipping in meters
    Z_near, Z_far = 0.1, 5.0
    u = np.round((K[0, 0] * (X / Z) + K[0, 2])).astype(np.int32)
    v = np.round((K[1, 1] * (Y / Z) + K[1, 2])).astype(np.int32)
    valid = (Z > Z_near) & (Z < Z_far) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    valid_count = int(np.count_nonzero(valid))
    print(f"[DEBUG raster] valid projections: {valid_count}")

    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    idxs = np.where(valid)[0]
    for i in idxs:
        ui, vi, zi = u[i], v[i], Z[i]
        if zi < zbuf[vi, ui]:
            zbuf[vi, ui] = zi
            depth_mm[vi, ui] = int(np.clip(zi * 1000.0, 0, 65535))
            if have_rgb and xyzrgb.shape[1] >= 6:
                r, g, b = xyzrgb[i, 3:6].astype(np.uint8)
                color[vi, ui] = np.array([b, g, r], dtype=np.uint8)

    # ---- synthesize grayscale RGB if we effectively have no color ----
    valid_depth_mask = depth_mm > 0
    num_valid_depth = int(np.count_nonzero(valid_depth_mask))
    num_colored_pixels = int(np.count_nonzero(color)) // 3

    if num_valid_depth > 0 and (not have_rgb or num_colored_pixels < 100):
        d = depth_mm.astype(np.float32)
        nz = d[valid_depth_mask]
        lo = np.percentile(nz, 2.0)
        hi = np.percentile(nz, 98.0)
        if hi <= lo:
            hi = lo + 1.0

        norm = np.zeros_like(d, dtype=np.float32)
        norm[valid_depth_mask] = np.clip((d[valid_depth_mask] - lo) / (hi - lo), 0.0, 1.0)
        gray = (norm * 255.0).astype(np.uint8)
        color = np.repeat(gray[:, :, None], 3, axis=2)

    valid_depth = int((depth_mm > 0).sum())
    print(f"[DEBUG raster] depth_mm>0 count: {valid_depth}")
    if valid_depth > 0:
        nz = depth_mm[depth_mm > 0]
        print(f"[DEBUG raster] depth_mm min/max (nonzero): {int(nz.min())}, {int(nz.max())}")

    return color, depth_mm


class SegCloudService(Node):
    def __init__(self):
        super().__init__("segmentation_cloud_server")

        # --------- CONFIG: adjust these to match your environment ----------
        # Container name that already runs your UCN environment & has the model/checkpoints inside
        self.container = "unseen_obj_container"  # e.g., set to whatever you use with segmentation_server.py
        self.base_dir = "/root/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering"
        self.base_dir_host = "/home/csrobot/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering"

        # Paths inside the container (what the image script expects)
        self.script_in = f"{self.base_dir}/test_images_segmentation_no_ros.py"
        self.cfg_in    = f"{self.base_dir}/experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml"
        self.pre_in    = f"{self.base_dir}/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth"
        self.pre_crop_in = f"{self.base_dir}/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth"
        self.network_name = "seg_resnet34_8s_embedding"
        self.conda_env = "unseen_obj"
        self.gpu_id = 0

        # Host <-> Container I/O mapping (make sure you `-v` mount these when starting the container)
        # self.input_dir_host  = f"{self.base_dir_host}/output/flexbe_inference_results/input_img"     # host writes RGBD here
        # self.output_dir_host = f"{self.base_dir_host}/output/flexbe_inference_results/output_seg"    # container writes results here (via bind mount)
        # self.input_dir_in    = f"{self.base_dir}/output/flexbe_inference_results/input_img"        # path inside container (bind of input_dir_host)
        # self.output_dir_in   = f"{self.base_dir}/output/flexbe_inference_results/output_seg"        # path inside container (bind of output_dir_host)

        self.input_dir_host  = f"{self.base_dir_host}/segmentation_from_cloud/"     # host writes RGBD here
        self.output_dir_host = f"{self.base_dir_host}/segmentation_from_cloud/"    # container writes results here (via bind mount)

        self.input_dir_in    = f"{self.base_dir}/segmentation_from_cloud/"        # path inside container (bind of input_dir_host)
        self.output_dir_in   = f"{self.base_dir}/segmentation_from_cloud/"        # path inside container (bind of output_dir_host)

        # Fallback intrinsics file on host (used if request.cam_info missing); mirror of the one inside container
        self.camera_json_host = f"{self.base_dir}/camera_params.json"

        # Ensure host dirs exist
        os.makedirs(self.input_dir_host, exist_ok=True)
        os.makedirs(self.output_dir_host, exist_ok=True)

        # Service server
        self.srv = self.create_service(SegCloud, "/segmentation_cloud_server", self.callback)
        self.get_logger().info("SegCloudService (docker-exec) ready.")

    def _docker_exec(self, cmd_inner: str) -> subprocess.CompletedProcess:
        # pass headless env to avoid GUI issues; activate conda env then run python script
        cmd = (
            f"docker exec {self.container} bash -lc "
            f"\"source /opt/conda/etc/profile.d/conda.sh && conda activate {self.conda_env} && {cmd_inner}\""
        )
        return subprocess.run(cmd, shell=True, capture_output=False, text=True, timeout=1200)

    def callback(self, req, res):
        try:
            # 1) intrinsics & target size
            K, H, W = intrinsics_from_request(req.cam_info, self.camera_json_host)

            # 2) cloud -> xyzrgb -> color/depth images
            xyzrgb, have_rgb = pc2_to_xyzrgb(req.cloud)
            self.get_logger().info(f"[DEBUG] xyzrgb shape:{xyzrgb.shape}")
            if xyzrgb.size > 0:
                Z = xyzrgb[:, 2]
                self.get_logger().info(f"[DEBUG] Z min/max/mean:{float(np.nanmin(Z)), float(np.nanmax(Z)), float(np.nanmean(Z))}")
                self.get_logger().info(f"[DEBUG] X min/max:{float(np.nanmin(xyzrgb[:,0])), float(np.nanmax(xyzrgb[:,0]))}")
                self.get_logger().info(f"[DEBUG] Y min/max:{float(np.nanmin(xyzrgb[:,1])), float(np.nanmax(xyzrgb[:,1]))}")

            if xyzrgb.size == 0:
                res.json_result = ""
                res.log_output = "Empty/invalid cloud"
                return res

            im_name = (req.im_name if hasattr(req, "im_name") and req.im_name else "from_cloud")

            color, depth_mm = rasterize_xyzrgb_to_images(xyzrgb, K, H, W, have_rgb)

            # 3) write inputs to host IO dir (container sees them via bind mount)
            color_path = os.path.join(self.input_dir_host,  f"{im_name}-color.png")
            depth_path = os.path.join(self.input_dir_host,  f"{im_name}-depth.png")
            imageio.imwrite(color_path, color)      # BGR8
            imageio.imwrite(depth_path, depth_mm)   # uint16 mm

            # 4) run the image script *inside* the container
            cmd_inner = (
                f"python {self.script_in} "
                f"--gpu {self.gpu_id} "
                f"--network {self.network_name} "
                f"--pretrained {self.pre_in} "
                f"--pretrained_crop {self.pre_crop_in} "
                f"--cfg {self.cfg_in} "
                f"--input_dir {self.input_dir_in} "
                f"--output_dir {self.output_dir_in} "
                f"--im_name {im_name}"
            )
            # proc = self._docker_exec(cmd_inner)
            # log_output = proc.stdout + "\n" + proc.stderr

            proc = self._docker_exec(cmd_inner)
            stdout_txt = proc.stdout or ""
            stderr_txt = proc.stderr or ""
            log_output = stdout_txt + ("\n" if stdout_txt and stderr_txt else "") + stderr_txt

            if proc.returncode != 0:
                res.success = False
                res.json_result = ""
                res.log_output = f"[docker] returncode={proc.returncode}\n{log_output}"
                return res

            # 5) read results from host output dir
            result_dir_host = os.path.join(self.output_dir_host, f"segmentation_{im_name}")
            seg_json_path = os.path.join(result_dir_host, "segmentation.json")
            if not os.path.exists(seg_json_path):
                res.success = False
                res.json_result = ""
                res.log_output = f"segmentation.json not found in {result_dir_host}\n{log_output}"
                return res

            with open(seg_json_path, "r") as f:
                seg_json = json.load(f)

            # embed absolute host paths so FlexBE can use them directly
            seg_json["result_dir"] = result_dir_host
            seg_json["base_output_dir"] = self.output_dir_host

            res.success = True
            res.json_result = json.dumps(seg_json)
            res.log_output = log_output
            return res

        except Exception as e:
            res.success = False
            res.json_result = ""
            res.log_output = f"error: {e}"
            return res

def main():
    rclpy.init()
    node = SegCloudService()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
