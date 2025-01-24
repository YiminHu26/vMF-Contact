import numpy as np
import open3d as o3d
import torch
from typing import Union, Optional

#### Codes borrowed from pytorch3d ####


from typing import Optional, Union, Tuple, List
import math
import torch
import torch.nn.functional as F
import random
from vmf_contact.model.utils import *
from vmf_contact.nn.util import match
from .matching import *

Device = Union[str, torch.device]
INTEG = True
normal_o3d_vis = True

class GraspBuffer:
    def __init__(self, device="cuda:0"):
        self.buffer_dict = {
            "pcds": [],
            "baselines": [], 
            "approaches": [], 
            "cp": [], 
            "cp2": [], 
            "kappa": [], 
            "graspness": [],
            "bin_score": []}
        self.buffer_size = 0
        self.baseline_fused = None
        self.approach_fused = None
        self.cp_fused= None
        self.cp2_fused= None
        self.kappa_fused = None
        self.graspness_fused = None
        self.bin_score_fused = None
        
    def create_vis(self):
        if normal_o3d_vis:
            self.vis = None
        else:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()

    def update(self, 
               pcds, 
               predictions,
               shift=0.0,
               resize=1.0, 
               grasp_height_th=5e-3, 
               grasp_width_th=0.1, 
               graspness_th=0.3, 
               pcd_from_prompt=None):
        
        if not isinstance(shift, torch.Tensor):
            shift = torch.tensor(shift, device=pcds.device, dtype=torch.float32)
        if not isinstance(resize, torch.Tensor):
            resize = torch.tensor(resize, device=pcds.device, dtype=torch.float32)
        
        cp = predictions["contact_point"]
        cp2 = predictions["contact_point"] + predictions["grasp_width"].unsqueeze(-1) * predictions["baseline"]
        grasp_width = predictions["grasp_width"].unsqueeze(-1)
        graspness = predictions["graspness"].unsqueeze(-1)
        approach = predictions["approach"]
        baseline = predictions["baseline"]
        kappa = predictions["kappa"].unsqueeze(-1)
        bin_score = predictions["bin_score"]

        filter = (graspness.squeeze(-1) > graspness_th) & \
                    (grasp_width.squeeze(-1) < grasp_width_th) & \
                    (cp[..., -1] > grasp_height_th) & \
                    (cp2[..., -1] > grasp_height_th)
        filter = filter.squeeze(-1)

        if pcd_from_prompt is not None:
            pcd_from_prompt = torch.tensor(pcd_from_prompt, device=pcds.device, dtype=torch.float32)
            # calculate the distance between the contact points and the prompt points
            dist = torch.cdist(cp, pcd_from_prompt)
            # dist2 = torch.cdist(cp2, pcd_from_prompt)
            filter = filter & (dist.min(1).values < 0.01)
        
        pcds = pcds * resize + shift

        if filter.sum() == 0:
            return False

        # print(f"Number of grasps: {filter.sum()}")
        else:
            cp = cp[filter] * resize + shift
            cp2 = cp2[filter] * resize + shift
            mid_pt = (cp2 + cp) / 2
            
            baseline = baseline[filter]
            kappa = kappa[filter]
            approach = approach[filter]
            graspness = graspness[filter]
            bin_score = bin_score[filter]

            self.buffer_dict["pcds"].append(pcds)  
            self.buffer_dict["baselines"].append(baseline)
            self.buffer_dict["approaches"].append(approach)
            self.buffer_dict["cp"].append(cp)
            self.buffer_dict["cp2"].append(cp2)
            self.buffer_dict["kappa"].append(kappa)
            self.buffer_dict["graspness"].append(graspness)
            self.buffer_dict["bin_score"].append(bin_score)

            self.buffer_size += 1
            
            if INTEG:
                self.integrate(pcds, baseline, bin_score, cp, cp2, kappa, graspness)
            return True
        
    def integrate(self, pcds, baseline, bin_score, cp, cp2, kappa, graspness):
        if self.baseline_fused is None:
            self.baseline_fused = baseline
            self.bin_score_fused = bin_score
            self.cp_fused= cp
            self.cp2_fused= cp2
            self.kappa_fused = kappa
            self.graspness_fused = graspness
        else:
            self.cp_dist = torch.cdist(self.cp_fused, cp)
            # minmum distance between the current grasp and new discovered grasp over threshold will be integrated
            new_grasp_idx = torch.where(self.cp_dist.min(0).values > 0.01)[0]
            # match the grasps lower than the threshold with the old grasps, perform bayesian update
            matched_idx = torch.where(self.cp_dist.min(0).values <= 0.01)[0]
            if len(matched_idx) > 0:
                # Match fused and new contact points
                pair_ind = match(self.cp_fused, cp)[0]
                cp_fused_matched = pair_ind[0]
                cp_matched = pair_ind[1]
                
                # Fuse graspness
                graspness_sum, unique_indices, counts = group_and_sum(graspness, cp_matched, cp_fused_matched)
                graspness_mean = graspness_sum / counts
                w = graspness_sum / (self.graspness_fused[unique_indices] + graspness_sum)
                self.graspness_fused[unique_indices] = self.graspness_fused[unique_indices] + graspness_sum
                
                # Fuse contact points
                cp_all = torch.cat((cp, cp2), dim=1)
                cp_sum, unique_indices, counts = group_and_sum(cp_all, cp_matched, cp_fused_matched)
                cp_sum, cp2_sum = cp_sum.split(cp.size(-1), dim=-1)

                cp_mean = cp_sum / counts
                self.cp_fused[unique_indices] = self.cp_fused[unique_indices] * (1-w) + cp_mean * w
                cp2_sum = cp2_sum / counts
                self.cp2_fused[unique_indices] = self.cp2_fused[unique_indices] * (1-w) + cp2_sum * w

                # Perform Bayesian update on the fused baseline
                kappa_sum, unique_indices, counts = group_and_sum(kappa, cp_matched, cp_fused_matched)
                baseline_kappa_sum, unique_indices, counts = group_and_sum(baseline * kappa, cp_matched, cp_fused_matched)
                self.baseline_fused[unique_indices] = (
                    self.baseline_fused[unique_indices] * self.kappa_fused[unique_indices] + baseline_kappa_sum
                ) / (self.kappa_fused[unique_indices] + kappa_sum)
                self.kappa_fused[unique_indices] += kappa_sum

                # Bayesian update on the fused approach using Dirichlet distribution
                bin_score_sum, unique_indices, counts = group_and_sum(bin_score, cp_matched, cp_fused_matched)
                self.bin_score_fused[unique_indices] = self.bin_score_fused[unique_indices] + bin_score_sum
                self.approach_fused = self.approach_from_bin_score(self.bin_score_fused, self.baseline_fused)
                    
            # Integrate new grasp points
            self.cp_fused = torch.cat((self.cp_fused, cp[new_grasp_idx]), dim=0)
            self.cp2_fused = torch.cat((self.cp2_fused, cp2[new_grasp_idx]), dim=0)
            self.baseline_fused = torch.cat((self.baseline_fused, baseline[new_grasp_idx]), dim=0)
            self.bin_score_fused = torch.cat((self.bin_score_fused, bin_score[new_grasp_idx]), dim=0)
            self.kappa_fused = torch.cat((self.kappa_fused, kappa[new_grasp_idx]), dim=0)
            self.graspness_fused = torch.cat((self.graspness_fused, graspness[new_grasp_idx]), dim=0)                
  
    def get_pcds_all(self):
        return torch.cat(self.buffer_dict["pcds"], dim=0)
    
    def approach_from_bin_score(self, bin_score, baseline):
        bin_num = bin_score.shape[-1]
        bin_vectors = rotate_circle_to_batch_of_vectors(bin_num, baseline)
        torch.gather(bin_vectors, 1, bin_score.argmax(dim=-1, keepdim=True)[...,None].expand(-1, -1, 3)).squeeze(1)
    
    def get_grasp_all(self):
        baselines = torch.cat(self.buffer_dict["baselines"], dim=0)
        approaches = torch.cat(self.buffer_dict["approaches"], dim=0)
        cp = torch.cat(self.buffer_dict["cp"], dim=0)
        cp2 = torch.cat(self.buffer_dict["cp2"], dim=0)
        kappa = torch.cat(self.buffer_dict["kappa"], dim=0)
        graspness = torch.cat(self.buffer_dict["graspness"], dim=0)
        return baselines, approaches, cp, cp2, kappa, graspness
    
    def get_pose_all(self, convention="xzy"):
        baselines, approaches, cp, cp2, kappa, graspness = self.get_grasp_all()
        poses = rotation_from_contact(baseline=baselines, 
                                      approach=approaches, 
                                      translation=(cp+cp2)/2, 
                                      convention=convention)
        return poses, kappa, graspness
    
    def get_grasp_fused(self):
        return self.baseline_fused, self.approach_fused, self.cp_fused, self.cp2_fused, self.kappa_fused, self.graspness_fused
    
    def get_pcds_curr(self):
        return self.buffer_dict["pcds"][-1]
    
    def get_grasp_curr(self):
        baseline = self.buffer_dict["baselines"][-1]
        approach = self.buffer_dict["approaches"][-1]
        cp = self.buffer_dict["cp"][-1]
        cp2 = self.buffer_dict["cp2"][-1]
        kappa = self.buffer_dict["kappa"][-1]
        graspness = self.buffer_dict["graspness"][-1]
        return baseline, approach, cp, cp2, kappa, graspness
    
    def get_pose_curr(self, convention="xzy"):
        baseline, approach, cp, cp2, kappa, graspness = self.get_grasp_curr()
        poses = rotation_from_contact(baseline=baseline, 
                                      approach=approach, 
                                      translation=(cp+cp2)/2,
                                      convention=convention)
        return poses, kappa, graspness
    
    def set_view(self, center):
        """Set a specific viewpoint."""
        ctr = self.vis.get_view_control()

        # Set camera parameters
        ctr.set_zoom(1)  # Zoom factor
        ctr.set_lookat(center)  # Look at center
        ctr.set_front([-1, 0, 1])  # View direction
        ctr.set_up([0, 0, 1])  # Up vector
    
    def vis_grasps(self, all = False):

        if len(self.buffer_dict["pcds"]) == 0:
            print("Buffer is empty, no grasp to visualize")
            return

        pcd = self.get_pcds_curr()
        baseline, approach, cp, cp2, kappa, graspness = self.get_grasp_curr()
        vis_list = vis_grasps(
                    samples=pcd,
                    cp=cp,
                    cp2=cp2,
                    kappa=kappa,
                    approach=approach,
                    score = graspness,
                )
        
        if not hasattr(self, "vis"):
            self.create_vis()
        
        if self.vis is None:
            o3d.visualization.draw_geometries(vis_list)
        else:
            if not all:
                self.vis.clear_geometries()

            for geom in vis_list:
                self.vis.add_geometry(geom)
            # Update the visualizer
            center = pcd.mean(0).cpu().numpy()
            self.set_view(center = center)
            self.vis.poll_events()
            self.vis.update_renderer()            

    def get_pose_curr_best(self, convention="xzy", sort_by="kappa", sample_num=1):

        if len(self.buffer_dict["pcds"]) == 0:
            print("Buffer is empty, no grasp to choose")
            return None
        
        poses, kappa, graspness = self.get_pose_curr(convention)
        
        score = kappa if sort_by == "kappa" else graspness
        
        #sort poses by criterion
        sample_num = min(sample_num, poses.size(0))
        poses_candidates = poses[torch.argsort(score, descending=True)][:sample_num]

        #randomly sample 1 poses
        pose_chosen = poses_candidates[random.randint(0, sample_num-1)].squeeze(0)
        return pose_chosen