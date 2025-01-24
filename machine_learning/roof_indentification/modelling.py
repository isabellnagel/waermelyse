import pandas as pd
import numpy as np
import sys
import os
import skimage.io
from skimage.transform import resize
import cv2
import json
import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image


print(os.getcwd())
print(sys.path)
current_dir = os.getcwd()
working_dir = os.path.join("machine_learning", "roof_indentification")
mask_rcnn_dir = os.path.join(working_dir, "Mask_RCNN")

modelling_py = os.path.join(working_dir, "modelling.py")
sys.path.append(modelling_py)

if mask_rcnn_dir in working_dir:
    pass
else:
    sys.path.append(mask_rcnn_dir)
#sys.path.append(r"C:\Users\MSI\Uni\master_projekt\waermelyse\machine_learning\roof_inpythodentification\Mask_RCNN")
print(sys.path)
print("Does Mask_RCNN exist?", os.path.exists(mask_rcnn_dir))
print("Does mrcnn exist?", os.path.exists(os.path.join(mask_rcnn_dir, "mrcnn")))
import mrcnn
from mrcnn import utils
from mrcnn.utils import Dataset
from mrcnn.model import log
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.config import Config


#COCO_MODEL_PATH = r"C:\Users\MSI\Uni\master_projekt\waermelyse\machine_learning\roof_indentification\data\transfer_pretrained_weights.h5"

#sys.path.append(r"C:\Users\MSI\Uni\master_projekt\waermelyse\machine_learning\roof_indentification\Mask_RCNN\samples\coco")  # To find local version
coco_path = os.path.join(mask_rcnn_dir, "samples", "coco")

if coco_path in sys.path:
    pass
else:
    sys.path.append(coco_path)

import coco
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

class SetupConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 1 Background + 1 Building
    IMAGE_MAX_DIM=320
    IMAGE_MIN_DIM=320

def binary_mask_to_polygon(binary_mask, epsilon=2.0):
    """
    Convert a binary mask to COCO polygon format, simplifying the contour.
    
    Args:
        binary_mask (numpy.ndarray): Binary mask with shape (H, W).
        epsilon (float): Approximation accuracy for simplifying contours.
    
    Returns:
        list: A list of polygons where each polygon is a list of coordinates.
    """
    # Ensure binary mask is uint8
    binary_mask = (binary_mask * 255).astype(np.uint8)

    # Find contours based on OpenCV version
    if cv2.__version__.startswith('4'):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        # Simplify the contour to reduce the number of points
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        # Flatten and convert to a list of coordinates
        simplified_contour = simplified_contour.flatten().tolist()
        if len(simplified_contour) >= 6:  # Polygon must have at least 3 points (6 coordinates)
            polygons.append(simplified_contour)

    return polygons

def process_folder_and_save(folder_path, model, output_json_path):
    """
    Process all images in a folder, predict annotations, and save to a COCO-style JSON.
    
    Args:
        folder_path (str): Path to the folder containing images.
        model: Your trained Mask R-CNN model for making predictions.
        output_json_path (str): Path to save the output JSON file.
    """
    # Initialize COCO JSON structure
    output_dict = {
        "images": [],
        "categories": [
            {"id": 1, "name": "roofs"}  # Adjust categories as needed
        ],
        "annotations": []
    }
    
    annotation_id = 0
    image_id = 0
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load image
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path).convert("RGB")  # Ensure 3 channels (RGB)
            width, height = image.size
            
            # Add image metadata to JSON
            output_dict["images"].append({
                "width": width,
                "height": height,
                "id": image_id,
                "file_name": f"images\\{file_name}"
            })
            
            # Prepare image for prediction (resize or preprocess if needed)
            image_array = np.array(image)  # Convert PIL Image to numpy array
            
            # Perform prediction with your model
            predictions = model.detect([image_array], verbose=0)[0]
            
            # Process predictions
            rois = predictions['rois']
            masks = predictions['masks']
            class_ids = predictions['class_ids']
            scores = predictions['scores']
            
            for i in range(len(rois)):
                # Extract bounding box
                y1, x1, y2, x2 = rois[i]
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                
                # Extract mask
                binary_mask = masks[:, :, i]
                polygons = binary_mask_to_polygon(binary_mask)

                # Add annotation
                output_dict["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_ids[i]),
                    "segmentation": polygons,  # Pass the polygons here
                    "bbox": bbox,
                    "iscrowd": 0,
                    "area": float(bbox[2] * bbox[3]),  # Width * Height
                    "score": float(scores[i])  # Confidence score
                })
                annotation_id += 1
            
            image_id += 1
        
    # Save JSON
    with open(output_json_path, 'w') as f:
        json.dump(output_dict, f, indent=2)  