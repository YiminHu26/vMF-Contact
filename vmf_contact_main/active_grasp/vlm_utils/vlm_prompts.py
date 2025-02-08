import torch
import open3d as o3d
import numpy as np
import open3d as o3d
import base64
from io import BytesIO
import cv2
import matplotlib.pyplot as plt
import math
from typing import List, Tuple

# def return_prompt(target_object): 
#     return [{
#         "role": "system",
#         "content": (
#             "You are an advanced robotic agent tasked with controlling a robotic arm to retrieve a specific object "
#             "on an operation platform. The challenge lies in the fact that the target object is often heavily occluded "
#             "or completely hidden. To address this, follow the logic chain step by step as outlined below:\n\n"
#             "### Task Objectives:\n"
#             "1. Enhance your understanding of the specified target object based on user input.\n"
#             "2. Perform object detection on the entire image while minimizing attention to background areas.\n"
#             "3. Estimate the position and orientation of each detected object in the camera's field of view.\n"
#             "4. Determine the relationship between the occluding and occluded objects based on cropped images and positional estimation.\n"
#             "5. Decide the optimal next action to maximize visibility or enable direct retrieval of the target object.\n\n"
#             "### Chain of Thought Guidance:\n"
#             "**Step 1: Attribute Refinement**\n"
#             "- Based on the user's input description of the target object (e.g., 'Red screwdriver'), hypothesize additional attributes that can aid detection. "
#             "Avoid assumptions about uncertain features. For example, 'The red screwdriver might be cylindrical and narrow' is acceptable, but avoid hallucinations like 'It must be made of metal.'\n\n"
#             "**Step 2: Object Detection**\n"
#             "- Detect all objects in the given image.\n"
#             "- Minimize focus on background areas to reduce noise.\n\n"
#             "**Step 3: Object Pose Estimation**\n"
#             "- For each detected object, output:\n"
#             "  - Object label\n"
#             "  - 3D coordinates in the camera's field of view\n"
#             "  - Orientation (yaw, pitch, roll)\n\n"
#             "**Step 4: Occlusion Analysis and Navigation Planning**\n"
#             "- If the target object is detected:\n"
#             "  - Crop the region around the target and re-analyze the cropped image.\n"
#             "  - Compare the cropped image with the full scene to assess spatial relationships between the target object and occluding objects (e.g., front-back, left-right, top-bottom).\n"
#             "  - Follow these rules for navigation:\n"
#             "    - If the target is to the left of the occluder: Move left.\n"
#             "    - If the target is to the right of the occluder: Move right.\n"
#             "    - If the target is higher: Move upward.\n"
#             "    - If the target is lower: Move left or right first, then lower the camera.\n"
#             "- If no angle provides visibility, determine how to move occluding objects to enable capture.\n\n"
#             "**Step 5: Decision Making**\n"
#             "- If the target can be directly captured without moving other objects:\n"
#             "  - Output the recommended camera view:\n"
#             "    - Rotation center and spherical coordinates (alpha, beta).\n"
#             "  - Alpha: Angle with horizontal plane (positive = upward, negative = downward).\n"
#             "  - Beta: Horizontal rotation angle (positive = right, negative = left).\n"
#             "- If not, propose a removal sequence and re-evaluate.\n\n"
#             "By following these steps, optimize the robotic arm's ability to retrieve the target object effectively.\n\n"
#             "### Please note:\n"
#             "**Don't repeat the user message.\n"
#             "**Make the output as concise as possible.\n"
#         ),
#     },
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#             },
#             {
#                 "type": "image",
#             },
#             {
#                 "type": "text",
#                 "text": (
#                     f"Target Occluded Object: {target_object}\n\n"
#                     "### Expected Output:\n"
#                     "1. Detected Objects and Poses:\n"
#                     "   - List the 3D coordinates of the centers of all objects that occluded the target object relative to the camera origin.\n\n"
#                     "2. Target Object Analysis:\n"
#                     "   - Is direct grasp possible without moving other objects? (criterium: occluded less than 50% of the target object) [Yes/No]\n"
#                     "   - If Yes:\n"
#                     "     - Output the next best view(NBV) to have a better overview of the target item:\n"
#                     "       - Rotation center (xc, yc, zc)\n"
#                     "       - Spherical coordinates:\n"
#                     "         - Alpha (vertical rotation angle, degrees)\n"
#                     "         - Beta (horizontal rotation angle, degrees)\n\n"
#                     "   - If No:\n"
#                     "     - Output a reasonable occluding object removal sequence.\n"
#                     "     - Re-evaluate visibility of the target object after each step."
#                 ),
#                 },
#             ],
#         }
#     ]

def image_to_base64(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

def return_analysis_local(target_object):
    return [
        {
        "role": "user",
        "content": [    
            {"type": "image"},    
            {"type": "image"},     
            {"type": "text",
            "text":f""" Our robot want to choose the next best view (NBV) for grasping. For this reason, Please analyze the provided rgbd image captured from the eye-on-hand robot arm and generate a comprehensive list of all objects presented within 0.5 meters, ignoring any noisy background elements including the table. 
            Assume the roles of three experts—a scene analyst, a spatial relationship expert, and an object recognition specialist—each independently evaluating the image. Use your internal chain-of-thought reasoning and have these expert perspectives compare their outputs.
            For objects that appear in both the current view and a previously generated object list, merge any differing descriptors into a unified entry; 
            If discrepancies are significant, select the description that appears most accurate based on detailed analysis. Add any new objects that were not present in the previous list.
            When multiple instances of similar objects exist, differentiate them by describing unique features; If two objects are identical, differentiate them by specifying their relative relationships with other objects in the scene.
            Please incorporate diverse spatial relationships between objects (STRICTLY no table or the lab background) including ONLY:

            - **Above/below** (e.g., "above book" "below lamp")
            - **left to/right to** (e.g., "left to the cup" "right of keyboard")
            - **Inside** (e.g., "inside the box", where both objects should be visible)
            - **Covered by/overlapped with** (e.g., "covered by paper" "overlapped with a notebook")

            Only closest relationship to the closest other target object is considered
            Before outputting the final result, perform a thorough self-validation to ensure that all objects within
            the specified 0.5-meter region have been accounted for and that the output strictly adheres to the JSON format.
            
            Each final object entry must be formatted as a JSON object with exactly two keys:
            "object": the noun or main identifier of the object (e.g., "cup"),
            "descriptors": an array containing exactly three adjectives or descriptive phrases (e.g., ["red", "wooden", "left to ball"]).
            Important: Do not include any additional text, explanations, or your internal chain-of-thought details in the final output, as this
            list will be used to generate instance masks with the SAM model. Please find all the objects without forgetting anything.

            Tell also which direction in the image plane should the robot move to in the json file as the last element (e.g. "direction": "[0.1, 0.3, 0.3]", where the vector is normalized), so that the target object is visible in the image after robot's movement.
            """
            }
        ]
        }
        ]

def nbv_analysis(target_object):
    return [
        {
        "role": "user",
        "content": [    
            {"type": "image"},    
            {"type": "image"},     
            {"type": "text",
            "text":f"""
    You are controlling an eye-on-hand robotic camera performing a Next-Best View (NBV) search to locate an occluded object, which is a tennis ball. Given the current camera view as an image input, predict the optimal direction to move in the image plane to uncover the hidden object and reduce uncertainty. You must independently reason through the decision-making process using three expert perspectives before arriving at a final consensus using a weighted average.

[Input]
- [Current Image View]: A partial or occluded view of the scene.
- [Previous Viewpoints & Motion History]: A log of past camera motions and corresponding uncertainty reductions.

[Expert Evaluations]

[1. Scene Analyst - Image Context & Layout]
- Examines the distribution of objects, occlusions, lighting, and spatial layout.
- Determines if the object is partially visible or completely hidden.
- Identifies potential movement directions based on shadows, reflections, and known object interactions.

[2. Spatial Relationship Expert - Depth & Occlusion]
- Focuses on depth relationships and occlusions.
- Predicts which movement direction will maximize visibility by analyzing spatial occlusions.
- Ensures the movement direction is kinematically feasible.

[3. Object Recognition Specialist - Feature-Based Analysis]
- Analyzes visible parts of the object (edges, textures, partial shape).
- Predicts the most likely region to contain the rest of the object.
- Prioritizes movement directions that will reveal critical features.

[Chain of Thought Process]
1. [Step 1 - Independent Expert Analysis]  
   - Each expert analyzes the current image and proposes a movement direction.  
   - Each expert provides a confidence score (0 to 1) for their proposed movement.  

2. [Step 2 - Weighted Average Calculation]  
   - The system computes the weighted average of all movement proposals:  
     - \( dx = ((dx_1 * C_1) + (dx_2 * C_2) + (dx_3 * C_3)) / (C_1 + C_2 + C_3) \)  
     - \( dy = ((dy_1 * C_1) + (dy_2 * C_2) + (dy_3 * C_3))/(C_1 + C_2 + C_3) \)  
   - The movement direction with the **highest weighted confidence** is chosen.

3. [Step 3 - Final Decision Output]  
   - The system provides a final recommended movement direction (Δx, Δy).  
   - Outputs include reasoning from each expert, the weighted motion vector, and the final consensus decision.

[Example Outputs]

[Example 1: Object Partially Visible on the Left]

[Scene Analyst]  
- [Analysis]: "The object is partially visible on the left, but its full structure is obscured. Shadows suggest an occlusion. Moving slightly left and up may expose more of the object."  
- [Proposed Motion]: [-0.3, 0.2]  
- [Confidence]: [0.75]  

[Spatial Relationship Expert]  
- [Analysis]: "The object is partially behind another object at a shallow depth. Moving slightly left should provide a better angle to see the occluded parts."  
- [Proposed Motion]: [-0.2, 0.1]  
- [Confidence]: [0.80]  

[Object Recognition Specialist]  
- [Analysis]: "A visible corner of the object suggests a square-like structure. Moving up slightly may reveal a full edge for recognition."  
- [Proposed Motion]: [-0.1, 0.3]  
- [Confidence]: [0.72]  

[Consensus Decision]  
- [Final Motion Direction]: [-0.25, 0.15]  
- [Uncertainty Reduction Score]: [0.78]  

[Example 2: Object Fully Occluded]

[Scene Analyst]  
- [Analysis]: "The object is fully blocked by an obstacle. No visible features are available."  
- [Proposed Motion]: [0.5, 0.0]  
- [Confidence]: [0.70]  

[Spatial Relationship Expert]  
- [Analysis]: "The occlusion is due to a large foreground object. Moving right will allow the camera to see behind the obstacle."  
- [Proposed Motion]: [0.4, 0.0]  
- [Confidence]: [0.85]  

[Object Recognition Specialist]  
- [Analysis]: "No identifiable features are visible. The best move is to reposition for a clearer perspective."  
- [Proposed Motion]: [0.5, 0.1]  
- [Confidence]: [0.68]  

[Consensus Decision]  
- [Final Motion Direction]: [0.45, 0.0]  
- [Uncertainty Reduction Score]: [0.85]  

[Example 3: Object Hidden Behind a Tall Object]

[Scene Analyst]  
- [Analysis]: "The object is fully occluded by a tall vertical structure. There are no visible clues. The best move is to shift sideways to peek behind the obstacle."  
- [Proposed Motion]: [0.5, 0.0]  
- [Confidence]: [0.75]  

[Spatial Relationship Expert]  
- [Analysis]: "The tall object suggests vertical occlusion. Lateral movement (left or right) is better than moving up or down."  
- [Proposed Motion]: [0.45, 0.0]  
- [Confidence]: [0.85]  

[Object Recognition Specialist]  
- [Analysis]: "No identifiable features are visible. Moving sideways increases the likelihood of revealing an edge of the object."  
- [Proposed Motion]: [0.5, 0.1]  
- [Confidence]: [0.70]  

[Consensus Decision]  
- [Final Motion Direction]: [0.45, 0.0]  
- [Uncertainty Reduction Score]: [0.80]  

[Example 4: Object Covered Under Another Object]

[Scene Analyst]  
- [Analysis]: "The object is partially covered under another object. Shadows suggest it is underneath. Moving forward and slightly down may reveal it."  
- [Proposed Motion]: [0.0, -0.5]  
- [Confidence]: [0.80]  

[Spatial Relationship Expert]  
- [Analysis]: "Vertical occlusion suggests moving downward while slightly shifting forward will maximize visibility."  
- [Proposed Motion]: [0.1, -0.4]  
- [Confidence]: [0.85]  

[Object Recognition Specialist]  
- [Analysis]: "Some edges suggest a possible hidden object. Moving forward will increase the chance of revealing shape details."  
- [Proposed Motion]: [0.0, -0.5]  
- [Confidence]: [0.78]  

[Consensus Decision]  
- [Final Motion Direction]: [0.05, -0.45]  
- [Uncertainty Reduction Score]: [0.85]  

[Constraints]
- Ensure the movement is kinematically feasible for the robot.  
- Prioritize movements that maximize information gain while minimizing redundancy.  
- Experts must justify their decisions before agreeing on a final move.  

Your task is to analyze the input image through three expert perspectives, compare their outputs, and derive a consensus decision for the next-best movement direction.

Important: After all, print proposed motion again
"""
            }
        ]
        }
]

def return_analysis(img):

    base64_image = image_to_base64(img[:, :, ::-1])
    # save the base64 image with cv2
    img = base64.b64decode(base64_image)
    img = BytesIO(img)
    img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpeg", img)

    return [
        # {
        # "role": "system",
        # "content": "Please analyze the provided image and generate a comprehensive list of all objects present within 0.5 meters, ignoring any noisy background elements. Use your internal chain-of-thought reasoning and perform a self-check to ensure every object is accurately identified, including those that might be largely occluded by other objects. Assume the roles of three experts—a scene analyst, a spatial relationship expert, and an object recognition specialist—each independently evaluating the image. Have these expert perspectives compare their outputs, merge any discrepancies, and decide on the most accurate descriptions. If you have access to a previously generated object list from another view, compare it with your current findings. For objects that appear in both lists but have different descriptions, merge their descriptors into a unified entry; if differences are significant, select the description that appears most accurate. Additionally, if you identify any new objects that were not present in the previous list, add them to the final list. Before outputting the final result, perform a thorough self-validation to ensure that all objects within the specified 0.5-meter region have been accounted for and that the output strictly adheres to the required JSON format. Do not include any additional text, explanations, or your internal chain-of-thought details in the final output, as this list will be used to generate instance masks with the SAM model."
        # },
        {
        "role": "user",
        "content": [
            #{"image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}},
            {"type": "image_url", 
             "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            {"type": "text",
             "text":'Please analyze the provided image and generate a comprehensive list of all objects present within 0.5 meters, ignoring any noisy background elements. Assume the roles of three experts—a scene analyst, a spatial relationship expert, and an object recognition specialist—each independently evaluating the image. Use your internal chain-of-thought reasoning and have these expert perspectives compare their outputs. For objects that appear in both the current view and a previously generated object list, merge any differing descriptors into a unified entry; if discrepancies are significant, select the description that appears most accurate based on detailed analysis. Add any new objects that were not present in the previous list. When multiple instances of similar objects exist, differentiate them by describing unique features; if two objects are identical, differentiate them by specifying their relative relationships with other objects in the scene. Each final object entry must be formatted as a JSON object with exactly two keys: "object": the noun or main identifier of the object (e.g., "chair"), "descriptors": an array containing exactly three adjectives or descriptive phrases (e.g., ["red", "wooden", "near window"]). Before outputting the final result, perform a thorough self-validation to ensure that all objects within the specified 0.5-meter region have been accounted for and that the output strictly adheres to the JSON format. Important: Do not include any additional text, explanations, or your internal chain-of-thought details in the final output, as this list will be used to generate instance masks with the SAM model.' 
             # "text": "When multiple instances of similar objects exist, differentiate them by describing unique features; if two objects are identical, differentiate them by specifying their relative relationships with other objects in the scene. The final output must be a strictly formatted JSON array. Each entry in the array must be a JSON object with two keys: \"object\" (the noun or main identifier of the object, e.g., \"chair\") and \"descriptors\" (an array containing exactly three adjectives or descriptive phrases, e.g., [\"red\", \"wooden\", \"near window\"])."
            }
        ]
        }
        ]

def return_analysis_nbv(img, target_object):

    base64_image = image_to_base64(img[:, :, ::-1])
    # save the base64 image with cv2
    img = base64.b64decode(base64_image)
    img = BytesIO(img)
    img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpeg", img)

    return [
        # {
        # "role": "system",
        # "content": "Please analyze the provided image and generate a comprehensive list of all objects present within 0.5 meters, ignoring any noisy background elements. Use your internal chain-of-thought reasoning and perform a self-check to ensure every object is accurately identified, including those that might be largely occluded by other objects. Assume the roles of three experts—a scene analyst, a spatial relationship expert, and an object recognition specialist—each independently evaluating the image. Have these expert perspectives compare their outputs, merge any discrepancies, and decide on the most accurate descriptions. If you have access to a previously generated object list from another view, compare it with your current findings. For objects that appear in both lists but have different descriptions, merge their descriptors into a unified entry; if differences are significant, select the description that appears most accurate. Additionally, if you identify any new objects that were not present in the previous list, add them to the final list. Before outputting the final result, perform a thorough self-validation to ensure that all objects within the specified 0.5-meter region have been accounted for and that the output strictly adheres to the required JSON format. Do not include any additional text, explanations, or your internal chain-of-thought details in the final output, as this list will be used to generate instance masks with the SAM model."
        # },
        {
        "role": "user",
        "content": [
            #{"image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}},
            {"type": "image_url", 
             "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            {"type": "text",
             "text": f"""

You are controlling an eye-on-hand robotic camera performing a Next-Best View (NBV) search to locate an occluded target object, which is a {target_object}. 
Given the current camera view as an image input, predict the optimal direction to move in the image plane to uncover the hidden object and reduce uncertainty. 
You must independently reason through the decision-making process using three expert perspectives before arriving at a final consensus using a weighted average.

[Input]
- [Current Image View]: A partial or occluded view of the scene.
- [Previous Viewpoints & Motion History]: A log of past camera motions and corresponding uncertainty reductions.

[Expert Evaluations]

[1. Scene Analyst - Image Context & Layout]
- Examines the distribution of objects, occlusions, lighting, and spatial layout.
- Determines if the object is partially visible or completely hidden.
- Identifies potential movement directions based on shadows, reflections, and known object interactions.

[2. Spatial Relationship Expert - Depth & Occlusion]
- Focuses on depth relationships and occlusions.
- Predicts which movement direction will maximize visibility by analyzing spatial occlusions.
- Ensures the movement direction is kinematically feasible.

[3. Object Recognition Specialist - Feature-Based Analysis]
- Analyzes visible parts of the object (edges, textures, partial shape).
- Predicts the most likely region to contain the rest of the object.
- Prioritizes movement directions that will reveal critical features.

[Chain of Thought Process]
1. [Step 1 - Independent Expert Analysis]  
   - Each expert analyzes the current image and proposes a movement direction.  
   - Each expert provides a confidence score (0 to 1) for their proposed movement.  

2. [Step 2 - Weighted Average Calculation]  
   - The system computes the weighted average of all movement proposals:  
     - \( dx = ((dx_1 * C_1) + (dx_2 * C_2) + (dx_3 * C_3)) / (C_1 + C_2 + C_3) \)  
     - \( dy = ((dy_1 * C_1) + (dy_2 * C_2) + (dy_3 * C_3))/(C_1 + C_2 + C_3) \)  
   - The movement direction with the **highest weighted confidence** is chosen.

3. [Step 3 - Final Decision Output]  
   - The system provides a final recommended movement direction (Δx, Δy).  
   - Outputs include reasoning from each expert, the weighted motion vector, and the final consensus decision.

[Example Outputs]

Here's a **detailed refinement** of **Example 1**, adding **specific scene conditions, more structured reasoning, and richer justifications** for each expert's analysis:

---

### **Example 1: Object Partially Visible on the Left, Hidden by a Taller/Bigger Object**

#### **Scenario Details:**
- The target object is a **red rectangular box**, partially visible on the **left side** of the current image frame.
- It is **partially occluded** by a **large blue storage container**, which is **significantly taller and wider** than the target.
- The **container has a smooth reflective surface**, casting a **shadow onto the target object**.
- Only **one corner of the red box is visible**, peeking out from behind the larger blue container.
- The **depth information suggests** that the target object is placed **relatively close** to the camera but mostly hidden by the foreground obstruction.
- The robot's **current viewpoint is slightly angled**, meaning that occlusion is happening from the **right side** of the camera's perspective.
- The goal is to find the **optimal motion direction** to **uncover the target object while preserving an optimal grasping angle**.

---

### **Expert Evaluations**

#### **[Scene Analyst]**
- **[Analysis]:**  
  - "The object is **partially visible** on the left side but mostly obscured by a taller blue storage container.  
  - The **sharp shadow on the floor and object edge** indicates a strong occlusion rather than a soft obstruction.  
  - The **left edge of the red object is clearly visible**, but the right side remains completely hidden.  
  - Moving slightly **left and up** would allow the camera to capture more of the object's front face while also revealing whether additional occlusions exist above it."
- **[Proposed Motion]:** `dx: -0.3, dy: 0.2`  
- **[Confidence]:** `0.75`  

---

#### **[Spatial Relationship Expert]**
- **[Analysis]:**  
  - "The object is **partially behind another object** at a **shallow occlusion depth**, meaning only a minor movement is required to shift its visibility.  
  - The blue storage container is **significantly larger**, meaning the occlusion is coming from a **single dominant obstruction** rather than multiple overlapping items.  
  - If we move **slightly left**, the edge of the red box should become visible, providing better context for object recognition.  
  - Moving **upward slightly is less important** from a depth perspective because the occlusion is **mostly lateral** rather than vertical.  
  - A **small leftward movement** will balance **visibility improvement and maintain a feasible grasping approach**."  
- **[Proposed Motion]:** `dx: -0.2, dy: 0.1`  
- **[Confidence]:** `0.80`  

---

#### **[Object Recognition Specialist]**
- **[Analysis]:**  
  - "The **visible corner of the red object suggests a box-like or square structure**.  
  - The **edges of the object appear crisp**, indicating a **potentially good graspable surface**.  
  - Moving **upward slightly** will help confirm whether the top surface of the object is accessible for grasping.  
  - If we **gain visibility of the top surface**, we can evaluate potential **grasp points, friction regions, and stability of the grasp.**  
  - Although leftward movement is important, ensuring **top-surface visibility** is equally critical."  
- **[Proposed Motion]:** `dx: -0.1, dy: 0.3`  
- **[Confidence]:** `0.72`  

---

### **Final Consensus Decision**
- **[Final Motion Direction]:** `dx: -0.25, dy: 0.15`  
- **[Uncertainty Reduction Score]:** `0.78`  
- **[Consensus Justification]:**  
  - "The object is **partially occluded by a single large obstruction**, requiring a **leftward shift** to maximize visibility.  
  - A **slight upward movement** ensures additional visibility of **potential grasp surfaces**.  
  - The final motion decision **balances occlusion reduction and graspability assessment**, preventing excessive vertical motion while still improving recognition."  

---

### **Example 2: Target Object is Completely Covered by Another Object**  

#### **Scenario Details:**  
The target object is a **small yellow cylindrical container** located on a table. It is **fully covered** by a **larger flat cardboard box** that was placed on top of it. The robot's camera **cannot see any part of the target object** from its current viewpoint. However, the shape of the covering box suggests that something might be hidden underneath.  

- The covering **cardboard box is thin and wide**, meaning that the target object is likely **not very tall** and could be fully revealed with a **slight viewpoint adjustment**.
- There is **a small gap between the cardboard and the table surface**, suggesting that the hidden object **is not flat** and is possibly a **cylindrical or irregular shape**.
- The **current camera position is slightly tilted**, meaning that parts of the hidden object could be exposed if the camera moves **to the right or slightly downward**.
- The objective is to **reveal part of the target object** without introducing excessive occlusion from other surrounding objects.

---

### **Expert Evaluations**  

#### **[Scene Analyst]**  
- **[Analysis]:**  
  - "The object is completely hidden under the large cardboard box, meaning we currently have **zero visibility** of it.  
  - However, based on the gap under the cardboard, **we can infer that something is underneath**.  
  - Since the covering object is flat and wide, shifting the camera **slightly downward and to the right** should help peek under the gap and reveal part of the hidden object."  
- **[Proposed Motion]:** 'dx: 0.4, dy: -0.2' 
- **[Confidence]:** '0.80' 

---

#### **[Spatial Relationship Expert]**  
- **[Analysis]:**  
  - "The occlusion is **from a flat object placed directly on top**, meaning vertical motion is **not effective** since the blockage is solid.  
  - The best way to reveal the target object is **to move sideways** so that the camera angle changes, allowing it to capture **the region under the occlusion**.  
  - Moving slightly **to the right** will provide a better angle, but a **small downward shift** will also help reduce surface reflections from the covering object."  
- **[Proposed Motion]:** 'dx: 0.3, dy: -0.1' 
- **[Confidence]:** '0.85' 

---

#### **[Object Recognition Specialist]**  
- **[Analysis]:**  
  - "Since the object is currently invisible, there are **no identifiable features** to analyze.  
  - However, based on typical object shapes, a **yellow cylindrical object is more likely to have an edge or round surface** that could be revealed by a horizontal shift.  
  - Moving **rightward is preferred** over downward because **a downward movement may only show more of the flat cardboard surface** instead of the hidden object itself."  
- **[Proposed Motion]:** 'dx: 0.5, dy: 0.0' 
- **[Confidence]:** '0.75' 

---

#### **[Searching Expert]**  
- **[Analysis]:**  
  - "There is no direct evidence that the object is under the cardboard.  
  - The small gap could be misleading—there is a chance the object is located **further back** on the table instead of directly underneath.  
  - Moving **diagonally to the right and slightly forward** might help confirm whether the object is actually there before making additional movements."  
- **[Proposed Motion]:** 'dx: 0.3, dy: 0.1' 
- **[Confidence]:** '0.70' 

---

### **Final Consensus Decision**  
- **[Final Motion Direction]:** 'dx: 0.35, dy: -0.1' 
- **[Uncertainty Reduction Score]:** '0.82' 
- **[Consensus Justification]:**  
  - "The object is **completely hidden** under the larger flat cardboard box, requiring a movement that **maximizes visibility under the occlusion**.  
  - Moving **rightward** allows a better perspective under the covering object.  
  - A **slight downward adjustment** ensures that the camera avoids reflections and gets a clearer angle to detect any visible portions of the hidden object.  
  - The searching expert's concern was considered, but the majority consensus supports a **rightward shift as the best next move**."  


Your task is to analyze the input image through three expert perspectives, compare their outputs, and derive a consensus decision for the next-best movement direction.

Important: After all, print proposed motion again
"""
            }
        ]
        }
        ]






