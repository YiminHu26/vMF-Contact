import math
import numpy as np
from typing import List
from .bbox_langsam import SceneObject, obb_collision_expanded
import random

class SceneConstraints:
    """
    Class to determine spatial relationships between objects in a 3D space.
    Includes methods for checking relative positioning and determining if an object is between others.
    """
    def __init__(self, threshold: float = 0.01):
        """Initialize the SceneConstraints object with a specified threshold."""
        self.threshold = threshold
    
    def _get_coordinates(self, inst):
        """Extracts the center coordinates of the given instance."""
        return np.array(inst.center)
    
    def _get_highest_point(self, inst):
        """Extracts the highest point of the given instance."""
        corners = inst.bbox_3d
        return max(corners, key=lambda corner: corner[2])
    
    def is_below(self, inst_0, inst_1, ratio = 0.8) -> bool:
        """Checks if inst_0 is below inst_1 within the defined threshold."""
        c0, c1 = self._get_coordinates(inst_0), self._get_coordinates(inst_1)
        dx, dy, dz = c0 - c1
        dxy = np.sqrt(dx**2 + dy**2)
        l0, w0, h0 = inst_0.bbox_dims
        l1, w1, h1 = inst_1.bbox_dims
        height_threshold = (h0 + h1) / 2 * ratio
        wl_threshold = (max(w0, w1) + max(l0, l1)) / 2 * ratio
        return dz > height_threshold and dxy < wl_threshold
    
    def classify_relation(center_A, size_A, center_B, size_B, tolerance=0.1):
        """
        Classifies the relation between two OBBs as "below" or "aside".
        
        center_A, center_B : tuple (x, y, z) - Center coordinates
        size_A, size_B : tuple (w, h, d) - Width, Height, Depth of OBB
        tolerance : float - Fractional tolerance to account for PCD noise
        """
        
        # Compute center differences
        dx = abs(center_A[0] - center_B[0])
        dy = abs(center_A[1] - center_B[1])
        dz = abs(center_A[2] - center_B[2])  # Vertical difference
        
        # Euclidean lateral distance (XY plane)
        d_xy = np.sqrt(dx**2 + dy**2)
        
        # Height threshold for "below" relation
        height_threshold = (size_A[1] + size_B[1]) / 2 - tolerance * max(size_A[1], size_B[1])
        
        # Width threshold for "aside" relation
        width_threshold = (size_A[0] + size_B[0]) / 2 - tolerance * max(size_A[0], size_B[0])
        
        # Classification logic
        if dz > height_threshold:
            return "below"
        elif d_xy > width_threshold:
            return "aside"
        else:
            return "uncertain (close proximity)"
    
    def _check_relative_position_xy(self, inst_0, inst_1) -> bool:
        """Generalized method to check relative positioning based on x and z directionality."""
        c0, c1 = self._get_coordinates(inst_0), self._get_coordinates(inst_1)
        return c0[1] < c1[1]
    
    def _check_relative_position_z(self, inst_0, inst_1, sign=1) -> bool:
        """Generalized method to check relative positioning based on x and z directionality."""
        c0, c1 = self._get_highest_point(inst_0), self._get_highest_point(inst_1)
        return c0[2]*sign < c1[2]*sign - self.threshold
    
    def is_low(self, inst_0, inst_1) -> bool:
        """Checks if inst_0 is to the left and below inst_1."""
        return self._check_relative_position_z(inst_0, inst_1)
    
    def is_high(self, inst_0, inst_1) -> bool:
        """Checks if inst_0 is to the left and above inst_1."""
        return self._check_relative_position_z(inst_0, inst_1, sign=-1)
    
    def is_left(self, inst_0, inst_1) -> bool:
        """Checks if inst_0 is to the left and above inst_1."""
        return self._check_relative_position_xy(inst_0, inst_1)
    
    def is_between_line(self, inst_0, inst_a, inst_b, threshold: float = 0.8) -> tuple[bool, float]:
        """Determines if inst_0 is between inst_a and inst_b along a line."""
        c0, cA, cB = self._get_coordinates(inst_0), self._get_coordinates(inst_a), self._get_coordinates(inst_b)
        diff_0A, diff_B0 = c0 - cA, cB - c0
        diff_0A, diff_B0 = diff_0A[:2], diff_B0[:2] # ignore z-axis
        #calculate cosine of the angle between the two vectors
        cos_theta = np.dot(diff_0A, diff_B0) / (np.linalg.norm(diff_0A) * np.linalg.norm(diff_B0))
        # print(f"for {inst_0.label}, cos_theta between {inst_a.label} and {inst_b.label}: {cos_theta}")

        return cos_theta > threshold # cos(30) = 0.866
    
    def print_distance_xy(self, inst_0, inst_1):
        """Calculates the Euclidean distance between two instances."""
        return np.linalg.norm(self._get_coordinates(inst_0)[:2] - self._get_coordinates(inst_1)[:2])
    
    def identify_relation(self, inst_0, inst_1, related_objs) -> List[str]:
        """Identifies the spatial relation between inst_0 and inst_1, considering nearby objects."""
        relations = []
        if self.is_below(inst_0, inst_1):
            relations.append(f"{inst_0.label} is below {inst_1.label}")
        
        relation = "left" if self.is_left(inst_0, inst_1) else "right"
        for lh in ("low", "high"):
            if getattr(self, f"is_{lh.lower()}")(inst_0, inst_1):
                relation += lh

        relations.append(f"is {relation} to {inst_1.label}")
        return relations
        
    def identify_between_relations(self, inst_0, related_objs) -> List[str]:
        relations = []
        if len(related_objs) >= 2:
            # Check if inst_0 is between two other objects
            sorted_objs = sorted(related_objs, key=lambda obj: self.print_distance_xy(inst_0, obj))
            for i in range(0, len(sorted_objs) - 1):
                inst_a = sorted_objs[i]
                for j in range(i + 1, len(sorted_objs)):
                    inst_b = sorted_objs[j]
                    if self.is_between_line(inst_0, inst_a, inst_b, threshold=0.7):
                        relations.append(f"is between {inst_a.label} and {inst_b.label}")
                    else:
                        continue
        return relations

def compile_relation(scene_objects):
    relation_dict = {}

    # If the target object is not found, return an error message
    # if target_obj is None:
        # return
    
    for target_obj in scene_objects.values():
        relation_dict[target_obj.label] = []
        # Find related objects within a certain distance threshold
        DIST_THRESHOLD = 0.4
        related_objs = [obj for obj in scene_objects.values() if obj.label != target_obj.label and obb_collision_expanded(target_obj.bbox_3d, obj.bbox_3d)]

        # If no related objects are found, return an error message
        if not len(related_objs):
            # print(f"Within distance {DIST_THRESHOLD}, can't find {target_obj.label} related objects")
            continue
        
        # print(f"\nRelated objects for {target_obj.label} (center: {target_obj.center}):")
        # check spatial relations
        checker = SceneConstraints(threshold=0.05)
        for robj in related_objs:
            relation_dict[target_obj.label] += checker.identify_relation(target_obj, robj, related_objs)
        relation_dict[target_obj.label] += checker.identify_between_relations(target_obj, related_objs)

        # print(f"It: {relation_dict[target_obj.label]}")

    return relation_dict

        