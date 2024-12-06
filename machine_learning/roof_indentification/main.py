import tensorflow as tf
from tensorflow.keras.optimizers import Adam
#from tensorflow import keras
#from keras.models import load_model
#from tensorflow.keras.optimizers import Adam
import numpy as np
import h5py
#import matplotlib.pyplot as plt
#import seaborn as sns

#import cv2
import os
print("done")
root_dir = os.getcwd()
model_weights = r'machine_learning\roof_indentification\unet_final.h5'

#with h5py.File(model_weights, 'r') as f:
    #for key in f.attrs:
        #print(key, f.attrs[key])

model_path = os.path.join(root_dir, model_weights)
#print(model_path)

custom_objects = {'Custom>Adam': Adam}
model = tf.keras.models.load_model(model_path)
print(model.summary())