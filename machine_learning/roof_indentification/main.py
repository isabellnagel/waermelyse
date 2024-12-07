import tensorflow as tf
from tensorflow.keras.optimizers import Adam
#from tensorflow import keras
#from keras.models import load_model
#from tensorflow.keras.optimizers import Adam
import numpy as np
import h5py
import matplotlib.pyplot as plt
#import seaborn as sns

import cv2
import os
print("done")
root_dir = os.getcwd()
model_weights = r'machine_learning\roof_indentification\data\unet_final.h5'

#with h5py.File(model_weights, 'r') as f:
    #for key in f.attrs:
        #print(key, f.attrs[key])

model_path = os.path.join(root_dir, model_weights)
#print(model_path)


model = tf.keras.models.load_model(model_path)
print(model.summary())

pred_list = []
#image_path = r"C:\Users\Isabell\master_projekt\waermelyse\machine_learning\roof_indentification\data\train\images\000000000012.jpg"
image_path = r"C:\Users\Isabell\master_projekt\waermelyse\machine_learning\wms_output\tile_488640_5881989_highres.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (256, 256))
pred_list.append(image)
pred_list = np.array(pred_list) / 255.0

print(f"Input shape: {pred_list.shape}")
print(f"Expected input shape: {model.input_shape}")
#pred_list = np.expand_dims(image, axis=0)  # Add batch dimension


predictions = model.predict(pred_list)
print(predictions)

# Apply thresholding to convert predicted masks to binary images
def post_process(predictions, threshold=0.5):
    binary_images = (predictions > threshold).astype(np.uint8)
    return binary_images

processed_predictions = post_process(predictions, threshold=0.5)
#accuracies = []
#for i in range(len(val_processed_predictions)):
    #accuracy = (val_processed_predictions[i] == val_labels[i]).mean()
    #accuracies.append(accuracy)

#np.argsort(accuracies)[:10]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original image (ensure it has RGB format if needed)
axes[0].imshow(pred_list[0])  # Assumes pred_list[0] is in the correct format
axes[0].set_title('Original Image')
axes[0].axis('off')

# Prediction (ensure the mask is properly scaled or processed)
axes[1].imshow(processed_predictions[0], cmap='gray')  # Assumes processed_predictions[0] is 2D
axes[1].set_title('Predicted Mask')
axes[1].axis('off')

plt.tight_layout()
plt.show()