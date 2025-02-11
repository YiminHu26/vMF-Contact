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

def image_to_base64(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

def return_analysis_local(additional_text=None):
    return [
        {
        "role": "user",
        "content": [    
            {"type": "image"},    
            {"type": "image"},     
            {"type": "text",
            "text":"""
              # 
              Analyze the provided image and generate a **comprehensive list of all objects within 0.5 meters**, ignoring any noisy background elements. This analysis will be performed through the independent evaluations of **three expert perspectives**:

              1. **Scene Analyst** - Identifies objects based on spatial context, occlusions, and visibility.
              2. **Spatial Relationship Expert** - Evaluates object depth, relative positioning, and interactions.
              3. **Object Recognition Specialist** - Classifies objects based on shape, texture, and known visual characteristics.

              ### **Processing Steps**
              1. **Independent Expert Evaluations**  
                - Each expert **independently** analyzes the scene and produces a preliminary list of detected objects, along with descriptive attributes.
                - Attributes should focus on **size, material, color, and spatial positioning**.

              2. **Comparison and Merging of Object Lists**  
                - Objects that **appear in both the current view and a previously generated object list** are **merged**, consolidating differing descriptors into a unified entry.
                - If discrepancies arise, the **most accurate** descriptor is selected based on detailed analysis.
                - **Newly detected objects** (not in prior lists) are added to the final set.
                - **Duplicate objects** (e.g., multiple cups) are **differentiated** based on **unique spatial relationships**.

              3. **Validation and JSON Formatting**
                - Perform a **thorough validation** to confirm that all objects **within 0.5 meters** are accounted for.
                - Ensure strict **JSON format adherence**:
                  - Each object is represented as a **JSON object** with exactly **two keys**:
                    - `"object"`: **Main identifier** (e.g., `"cup"`, `"block"`), **no more than 2 words**.
                    - `"descriptors"`: **An array containing exactly three adjectives or descriptive phrases** (e.g., `["red", "plastic", "near box"]`).
                - **No extra text, explanations, or chain-of-thought reasoning** should appear in the final output.
              """ 
              + additional_text if additional_text is not None else "" + 
              """
              ### **Final Output Example**
              ```json
              [
                {
                  "object": "cup",
                  "descriptors": ["red", "plastic", "near wooden block"]
                },
                {
                  "object": "tennis ball",
                  "descriptors": ["yellow", "partially visible", "inside red cup"]
                },
                {
                  "object": "wooden block",
                  "descriptors": ["rectangular", "light brown", "supporting wooden board"]
                },
                {
                  "object": "wooden board",
                  "descriptors": ["flat", "medium brown", "resting on two objects"]
                },
                {
                  "object": "cheese box",
                  "descriptors": ["red", "cardboard", "next to wooden block"]
                }
              ]``` \n
              """ 
            }
        ]
        }
        ]

def return_analysis(img, key_not_detected = None):

    base64_image = image_to_base64(img[:, :, ::-1])
    # save the base64 image with cv2
    img = base64.b64decode(base64_image)
    img = BytesIO(img)
    img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpeg", img)

    if key_not_detected is not None:
        key_not_detected = ", ".join(key_not_detected)
        additional_text = f"**Cautious: Objects not found in your last analysis by SAM:** {key_not_detected}. Please ensure to correct them in this round."
        print(additional_text)   
    else:
        additional_text = None

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
             "text": """
              # 
              Analyze the provided image and generate a **comprehensive list of all objects within 0.5 meters**, ignoring any noisy background elements. This analysis will be performed through the independent evaluations of **three expert perspectives**:

              1. **Scene Analyst** - Identifies objects based on spatial context, occlusions, and visibility.
              2. **Spatial Relationship Expert** - Evaluates object depth, relative positioning, and interactions.
              3. **Object Recognition Specialist** - Classifies objects based on shape, texture, and known visual characteristics.

              ### **Processing Steps**
              1. **Independent Expert Evaluations**  
                - Each expert **independently** analyzes the scene and produces a preliminary list of detected objects, along with descriptive attributes.
                - Attributes should focus on **size, material, color, and spatial positioning**.

              2. **Comparison and Merging of Object Lists**  
                - Objects that **appear in both the current view and a previously generated object list** are **merged**, consolidating differing descriptors into a unified entry.
                - If discrepancies arise, the **most accurate** descriptor is selected based on detailed analysis.
                - **Newly detected objects** (not in prior lists) are added to the final set.
                - **Duplicate objects** (e.g., multiple cups) are **differentiated** based on **unique spatial relationships**.

              3. **Validation and JSON Formatting**
                - Perform a **thorough validation** to confirm that all objects **within 0.5 meters** are accounted for.
                - Ensure strict **JSON format adherence**:
                  - Each object is represented as a **JSON object** with exactly **two keys**:
                    - `"object"`: **Main identifier** (e.g., `"cup"`, `"block"`), **no more than 2 words**.
                    - `"descriptors"`: **An array containing exactly three adjectives or descriptive phrases** (e.g., `["red", "plastic", "near box"]`).
                - **No extra text, explanations, or chain-of-thought reasoning** should appear in the final output.
              """ 
              + additional_text if additional_text is not None else "" + 
              """
              ### **Final Output Example**
              ```json
              [
                {
                  "object": "cup",
                  "descriptors": ["red", "plastic", "near wooden block"]
                },
                {
                  "object": "tennis ball",
                  "descriptors": ["yellow", "partially visible", "inside red cup"]
                },
                {
                  "object": "wooden block",
                  "descriptors": ["rectangular", "light brown", "supporting wooden board"]
                },
                {
                  "object": "wooden board",
                  "descriptors": ["flat", "medium brown", "resting on two objects"]
                },
                {
                  "object": "cheese box",
                  "descriptors": ["red", "cardboard", "next to wooden block"]
                }
              ]``` \n
              """ 
        }
        ]
        }
        ]

