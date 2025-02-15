import threading
import rclpy
import numpy as np
#import spatialmath as sm
from .utils_node import *

from .inference_node_base import *

from vmf_contact_main.camera_utils import *
from vmf_contact_main.active_grasp.policy import make, registry
from vmf_contact_main.active_grasp.bbox import AABBox
from vmf_contact_main.active_grasp.spatial import *
from vmf_contact_main.active_grasp.timer import Timer

import argparse
from functools import partial

# from vgn.utils import ros_utils
from pathlib import Path

O_RESOLUTION = 40
O_SIZE = .3
O_VOXEL_SIZE = O_SIZE / O_RESOLUTION
min_z_dist = 0.45

azi_ele_groups = [
            [(-90, -50), (40, 145)], 
            [(-50, 0), (50, 135)],
            [(0, 50), (50, 135)],
            [(50, 90), (40, 145)]
            ]

azi_ele_groups = [[(0, 1), (50, 135)]]


class State:
    def __init__(self, tsdf):
        self.tsdf = tsdf

class AIRNodeVLM(AIRNode):

    def __init__(self):
        super().__init__(use_langsam=True)

        self.camera_ready_pose = list_to_pose_stamped([-0.370, -0.592, 1.224, 0.942, 0.007, -0.005, 0.336], "world")
        self.pcd_shift=np.array([-0.86, 0.1, 0.0])
        self.pcd_center = list_to_pose_stamped(self.pcd_shift.tolist() + [0., 0., 0., 1.], "base_link")
        self.publish_new_frame("center", self.pcd_center)
            
        lower = [self.pcd_center.pose.position.x - O_SIZE / 2, 
                    self.pcd_center.pose.position.y - O_SIZE / 2, 
                    self.pcd_center.pose.position.z]
        upper = [self.pcd_center.pose.position.x + O_SIZE / 2,
                self.pcd_center.pose.position.y + O_SIZE / 2,
                self.pcd_center.pose.position.z + 0.09]
        
        middle = (np.array(lower) + np.array(upper)) / 2
        self.box_center = list_to_pose_stamped(middle.tolist() + [0., 0., 0., 1.], "base_link")
        self.publish_new_frame("box_center", self.box_center)

        self.bbox: AABBox = AABBox(lower, upper)
        
        # Active search setting
        parser = create_parser()
        args = parser.parse_args()                
        self.user_input_thread = threading.Thread(target=self.handle_user_input)
        self.user_input_thread.start()
        self.set_vel_acc(.1, .1)

        if not os.path.exists("/home/yitian/data_active_grasp"):
            os.makedirs("/home/yitian/data_active_grasp")
        elif len(os.listdir("/home/yitian/data_active_grasp")) > 0:
            for file in os.listdir("/home/yitian/data_active_grasp"):
                if file.endswith(".npz") or file.endswith(".jpg"):
                    os.remove(os.path.join("/home/yitian/data_active_grasp", file))

    def process_point_cloud_and_rgbd(self, save_data=False, pcd_only=False):
        # TODO: add rgb image processing
        if self.last_point_cloud_msg is None:
            self.get_logger().info("No point cloud message received yet.")
            if pcd_only:
                return None, False
            return [None] * 4, False

        if self.last_image_msg is None:
            self.get_logger().info("No image message received yet.")
            return [None] * 4, False
    
        if self.last_depth_msg is None:
            self.get_logger().info("No depth message received yet.")
            return [None] * 4, False

        if self.last_point_cloud_msg is None:
            self.get_logger().info("No camera info message received yet.")
            return [None] * 4, False
        
        # Get the latest point cloud and image messages
        pcd_msg_camera = self.last_point_cloud_msg
        img = self.last_image_msg

        # transform the point cloud to base_link frame
        camera= CameraInfo(
            width=self.image_width, height=self.image_height, fx=self.camera_matrix[0, 0], fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2], cy=self.camera_matrix[1, 2], scale=1000.0
        )
        pcd_from_depth = create_point_cloud_from_depth_image(self.last_depth_msg, camera, organized=True).reshape(-1, 3)
        
        # Look up for the transformation between base_link and the frame_id of the point cloud
        t_robot_2_camera = self.tf_buffer.lookup_transform(
                "base_link", pcd_msg_camera.header.frame_id, rclpy.time.Time()
            )
        pcd_numpy_base_link = transform_points(pcd_from_depth, t_robot_2_camera.transform) 

        if pcd_only:
            return pcd_numpy_base_link, True 

        # remove previous masks
        for file in os.listdir(current_file_folder):
            if file.endswith(".jpg"):
                os.remove(os.path.join(current_file_folder, file))
        
        # transformation to pose
        cam_pose_robot = transform_to_pose(t_robot_2_camera.transform)
        
        if self.langsam_model is not None:
            scene_objects = self.generate_langsam(pcd_numpy_base_link, img)
            bbox_center_dict = {obj.instance_id: obj.center for obj in scene_objects}
            bbox_corners_dict = {obj.instance_id: obj.bbox_3d for obj in scene_objects}
            bbox_lwh_dict = {obj.instance_id: obj.bbox_dims for obj in scene_objects}
        
        # save the image, depth, point cloud and camera pose as a dictionary of numpy arrays
        # viszualize the rgbd data
        if save_data:
            pos = [cam_pose_robot.position.x - gaze_point_robot[0], 
                   cam_pose_robot.position.y - gaze_point_robot[1], 
                   cam_pose_robot.position.z - gaze_point_robot[2]]
            azimuth, elevation = pos_to_azi_elev(pos)
            name = f"/home/yitian/data_active_grasp/{azimuth}_{elevation}"
            cv2.imwrite(f"/home/yitian/data_active_grasp/{azimuth}_{elevation}.jpg", img[..., ::-1])
            dict_rgbd = {
                "image": img,
                "depth": self.last_depth_msg,
                "pcd": pcd_numpy_base_link,
                "cam_pose": [cam_pose_robot.position.x, 
                            cam_pose_robot.position.y, 
                            cam_pose_robot.position.z,
                            cam_pose_robot.orientation.x, 
                            cam_pose_robot.orientation.y, 
                            cam_pose_robot.orientation.z, 
                            cam_pose_robot.orientation.w],
                "bbox_center_dict": bbox_center_dict if self.langsam_model is not None else None,
                "bbox_corners_dict": bbox_corners_dict if self.langsam_model is not None else None,
                "bbox_lwh_dict": bbox_lwh_dict if self.langsam_model is not None else None
            }
            np.savez(f"{name}.npz", **dict_rgbd)

        return (pcd_numpy_base_link, 
                self.last_image_msg, 
                self.last_depth_msg.astype(np.float32) / 1000.0, 
                cam_pose_robot), True
    
    def generate_langsam(self, pcd, img):

        def mark_duplicates(labels):
            label_count = {}
            result = []
            
            for label in labels:
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1
                result.append(f"{label} {label_count[label]}")
            return result
        
        # Process the prompt point cloud
        time_curr = time.time()
    
        # predict masks with lang_sam
        results = self.langsam_model.predict([Image.fromarray(img)], [". ".join(obj_list)])

        print(f"[LangSAM] Time taken: {time.time() - time_curr}")
                
        # check if there are labels detected
        labels = results[0]["labels"]
        if len(labels) == 0:
            print("[VLM]: No labels detected.")
            return pcd
        # check duplicates in labels, if there are duplicates, mark them with a number
        labels = mark_duplicates(labels)

        # sort labels by score
        scores = results[0]["scores"]
        labels = [label for _, label in sorted(zip(scores, labels), reverse=True, key=lambda pair: pair[0])]
        masks = [mask for _, mask in sorted(zip(scores, results[0]["masks"]), reverse=True, key=lambda pair: pair[0])]
        scores = sorted(scores, reverse=True)

        print("[VLM]: Results: ", labels)
        print("[VLM]: Scores: ", results[0]["scores"])

        vis_list = []
        scene_objects = []
        # mask point cloud and image
        for i, text in enumerate(labels):

            if "1" not in text:
                continue

            mask = masks[i].astype(np.uint8)[:, :, None]

            # mask image and point cloud
            pcd_masked = pcd[mask.reshape(-1) == 1]

            # filter out noises out of range:
            pcd_masked = pcd_masked[(pcd_masked[:, 2] > 0.01) & (pcd_masked[:, 2] < min_z_dist)]
            #if hasattr(self, "gaze_point_robot"):
            pcd_masked_dist_to_gaze = np.linalg.norm(pcd_masked[:, :3] - gaze_point_robot, axis=1)
            pcd_masked = pcd_masked[pcd_masked_dist_to_gaze < min_z_dist - 0.3]

            # compile scene object
            if pcd_masked.shape[0] == 0:
                print(f"Object: {text} is out of range.")
                continue
            try:
                scene_object = SceneObject(pcd_masked, text)
            except:
                print(f"Object: {text} failed to compute OBB.")
                print(pcd_masked)
                continue
            scene_objects.append(scene_object)
            print(f"Object: {scene_object.instance_id}. Center: {scene_object.center}") 

            # visualize the scene object
            vis_list += visualize_pcd_with_obb(pcd_masked, scene_object.bbox_3d)
            # save masked image
            # cv2.imwrite(f"{current_file_folder}/{text}.jpg", img[..., ::-1] * mask)
        
        # o3d.visualization.draw_geometries(vis_list)
                
        return scene_objects  
    
    def handle_user_input(self):
        self.get_camera_info()
        self.to_camera_ready_pose()
        # create folder to save data
        self.generate_trajectory(func=partial(self.process_point_cloud_and_rgbd, save_data=True))
    
    def generate_trajectory(self, func=None):
        self.change_state_to_cartesian_ctl()
        self.set_eelink("tcp")
        gaze_point_robot = [-0.73, 0.1, 0.1] # TODO: remove this line, this is a test for gazing at middle of the desk
        self.publish_new_frame("gaze_point", list_to_pose_stamped(gaze_point_robot + [0., 0., 0., 1.], "base_link"))
        
        round = 0
        traj = []
        self.movement_finished_flag.clear()
        for azi_ele in azi_ele_groups:
            for azimuth in range(*azi_ele[0], 10):
                t = []            
                ele_range = range(*azi_ele[1]) if round % 2 == 0 else range(*azi_ele[1])[::-1]
                for elevation in ele_range:
                    pos = azi_to_pos(azimuth, elevation, min_z_dist)
                    pos =[pos[i] + gaze_point_robot[i] for i in range(3)]
                    quaternion = look_at_transformation(gaze_point_robot, pos)
                    t.append(pos + quaternion)
                round += 1
                traj.append(self.create_trajectory(t))

        print(f"Trajectory: {traj}")
        for t in traj:
            self.movement_finished_flag.clear()
            self.send_goal_traj(t) 
            while not self.movement_finished_flag.is_set():
                if func is not None:
                    func()
                # if is_data:
                #     pcd = camera_data[0]
                #     grasp, grasp_criterien = self.agent_inference(pcd)
        print("Trajectory over")
                       

def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)
    pcd_listener = AIRNodeVLM()
    try:
        rclpy.spin(pcd_listener)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info("Exiting")

    pcd_listener.destroy_node()
    rclpy.shutdown()

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, choices=registry.keys(), default="vlm")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--wait-for-input", action="store_true")
    parser.add_argument("--logdir", type=Path, default="logs")
    parser.add_argument("--seed", type=int, default=1)
    return parser

if __name__ == "__main__":
    main()

