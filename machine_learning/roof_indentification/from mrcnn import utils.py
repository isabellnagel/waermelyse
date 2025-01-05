import sys
import os
import skimage.io
from skimage.transform import resize
import cv2


os.chdir('Mask_RCNN')
print(os.getcwd())
print(sys.path)
sys.path.append(r"C:\Users\Isabell\master_projet_2\Mask_RCNN")

from mrcnn import utils
from mrcnn.model import log
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.config import Config


COCO_MODEL_PATH = r'C:\Users\Isabell\master_projet_2\Data\pretrained_weights.h5'

sys.path.append(r"C:\Users\Isabell\master_projet_2\Mask_RCNN\samples\coco")  # To find local version

print(sys.path)

import coco


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 1 Background + 1 Building
    IMAGE_MAX_DIM=320
    IMAGE_MIN_DIM=320   
config = InferenceConfig()
config.display()


ROOT_DIR = r"C:\Users\Isabell\master_projet_2"
print("loading mask R-CNN model")
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=ROOT_DIR)
#-------------- Load weights trained on MS-COCO -------------------------------
model.load_weights(COCO_MODEL_PATH, by_name=True)
# Access the internal Keras model and print its summary
model.keras_model.summary()

class_names = ['BG', 'building'] # In our case, we have 1 class for the background, and 1 class for building
#file_names = next(os.walk(IMAGE_DIR))[2]

image_path = r"C:\Users\Isabell\master_projet_2\Data\tile_489476_5881989_highres.jpg"

random_image = skimage.io.imread(image_path)
resized_image = cv2.resize(random_image, (300, 300))

predictions = model.detect([resized_image]*config.BATCH_SIZE, verbose=1) # We are replicating the same image to fill up the batch_size

p = predictions[0]
visualize.display_instances(resized_image, p['rois'], p['masks'], p['class_ids'], 
                            class_names, p['scores'])