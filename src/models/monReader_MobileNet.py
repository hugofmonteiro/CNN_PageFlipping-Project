import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1.0/255.)
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1.0/255.)

#Pass the images through the generator
trainImageData = train_generator.flow_from_directory("../data/training",
                                                     batch_size=32, #how much images to give per iteration in an epoch
                                                     class_mode="binary", #Incase of multi-class classification, "categorical"
                                                     target_size=(64,64) #Ensures all images are of same size (resizing)
                                                     ) 

testImageData = train_generator.flow_from_directory("../data/testing",
                                                     batch_size=32, #how much images to give per iteration in an epoch
                                                     class_mode="binary", #Incase of multi-class classification, "categorical"
                                                     target_size=(64,64) #Ensures all images are of same size (resizing)
                                                     ) 


# Learning Transferlearning with MobileNet from keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNet, imagenet_utils
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

# Importing the MobileNet model but discarding the last 1000 neuron layer for adjustment to our dataset
base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Freeze the base_model
base_model.trainable = False

# Create the model
model = tf.keras.models.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(1, activation='sigmoid')  # Single neuron for binary classification
])
model.summary()

#Compile
model.compile(optimizer="adam",
              loss="binary_crossentropy", #For multi-class classification: categorical_crossentropy | sparse_categorical_crossentropy
              metrics=[tfa.metrics.F1Score(num_classes=1, threshold=0.5)])

len(trainImageData.filenames) // trainImageData.batch_size

# Fit the model
model.fit(trainImageData,
          validation_data=testImageData,
          epochs=10,
          steps_per_epoch = len(trainImageData.filenames) // trainImageData.batch_size,
          validation_steps= len(testImageData.filenames) // testImageData.batch_size)

# Save the model
model.save('../models/MobileNet.keras')

