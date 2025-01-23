import pandas as pd
import numpy as np
import sys
import os
import skimage.io
from skimage.transform import resize
import cv2
import json
import numpy as np


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


class CustomDataset(Dataset):
    def __init__(self, coco_json, images_dir, img_height=320, img_width=320):
        super().__init__()
        self.coco_json = coco_json
        self.images_dir = images_dir
        self.img_height = img_height
        self.img_width = img_width
        self.coco = COCO(coco_json)  # Load COCO annotations

    def load_data(self):
        # Add only 'Dächer' class to the dataset
        for category in self.coco.dataset['categories']:
            if category['name'] == 'trees':  # Only add 'Dächer' class
                print(category["name"])
                print("done")
                self.add_class("dataset", category['id'], category['name'])
        print("Classes added to dataseeeet:", self.class_info)  # Debugging line

        # Add images to the dataset
        image_ids = self.coco.getImgIds()  # Get all image IDs in the dataset
        print(f"Found {len(image_ids)} images in COCO annotations.")  # Debugging line

        for img_id in image_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.images_dir, img_info['file_name'])

            # Check if the image file exists
            if not os.path.exists(img_path):
                print(f"Warning: Image file {img_path} not found!")
                continue

            self.add_image(
                "dataset", 
                image_id=img_id,
                path=img_path,
                width=img_info['width'],
                height=img_info['height']
            )

        # Debugging: Check if dataset has been populated
        print(f"Dataset loaded with {len(self.image_info)} images.")
        self.debug_dataset()

    def debug_dataset(self):
        """Debugging function to check the contents of the dataset."""
        # Check if the images are correctly loaded
        if not self.image_info:
            print("Dataset is empty! No images were loaded.")
            return

        print("\nFirst 5 Images in Dataset:")
        for i, info in enumerate(self.image_info[:5]):
            print(f"Image {i + 1}:")
            print(f"  ID: {info['id']}")
            print(f"  Path: {info['path']}")
            print(f"  Dimensions: {info['width']} x {info['height']}")

        # Check class info
        print("\nClasses in Dataset:")
        for class_info in self.class_info:
            print(f"  Class ID: {class_info['id']}, Name: {class_info['name']}")

        # Check total counts
        print(f"\nTotal images: {len(self.image_info)}")
        print(f"Total classes: {len(self.class_info)}")




    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        image_info = self.image_info[image_id]
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_info["id"]], catIds=[0], iscrowd=False))  # Adjusted category IDs

        # Initialize mask with zeros
        mask = np.zeros((image_info["height"], image_info["width"], len(annotations)), dtype=np.uint8)
        class_ids = []

        print(f"\nProcessing Image ID: {image_id}")
        print(f"  Image Path: {image_info['path']}")
        print(f"  Annotations Count: {len(annotations)}")

        for i, ann in enumerate(annotations):
            print(f"  Annotation {i + 1}/{len(annotations)} ID: {ann['id']}")
            if not ann.get("segmentation", []):
                print(f"    Warning: Annotation {ann['id']} has no segmentation. Skipping.")
                continue

            if isinstance(ann['segmentation'], list):
                for polygon in ann['segmentation']:
                    try:
                        rr, cc = skimage.draw.polygon(polygon[1::2], polygon[::2])
                        rr = np.clip(rr, 0, image_info["height"] - 1)
                        cc = np.clip(cc, 0, image_info["width"] - 1)
                        mask[rr, cc, i] = 1
                    except Exception as e:
                        print(f"    Error processing polygon for annotation {ann['id']}: {e}")

            else:
                print(f"    Skipping annotation {ann['id']} - unsupported segmentation format.")

            # Append class ID
            try:
                class_id = self.map_source_class_id("dataset.0")  # Update mapping if needed
                class_ids.append(class_id)
            except Exception as e:
                print(f"    Error mapping class ID for annotation {ann['id']}: {e}")

        print(f"  Generated Mask Shape: {mask.shape}")
        print(f"  Generated Class IDs: {class_ids}")

        # Ensure mask is in correct format
        return mask.astype(np.bool_), np.array(class_ids, dtype=np.int32)



    def image_reference(self, image_id):
        """Return the path to the image in the dataset."""
        return self.image_info[image_id]["path"]

    def preprocess_image(self, image):
        """Preprocess image: resize to fixed shape and normalize."""
        # Resize image to the fixed size (320x320)
        image_resized = cv2.resize(image, (self.img_width, self.img_height))
        # Normalize the image
        image_resized = image_resized.astype(np.float32)
        image_resized -= np.array([123.7, 116.8, 103.9], dtype=np.float32)  # Mean pixel values
        return image_resized