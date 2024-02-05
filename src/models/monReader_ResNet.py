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

# Loading ResNet from keras to perform transfer learning
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Importing the ResNet model but discarding the last 1000 neuron layer for adjustment to our dataset
base_model = tf.keras.applications.ResNet152(weights = 'imagenet', include_top = False, input_shape=(64, 64, 3))

# Freeze the base_model so we keep the pre-trained structure
base_model.trainable = False

# Adjusting the model to the dataset

model = tf.keras.models.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(1, activation='sigmoid')
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
          epochs=200,
          steps_per_epoch = len(trainImageData.filenames) // trainImageData.batch_size,
          validation_steps= len(testImageData.filenames) // testImageData.batch_size)

# Save the model
model.save('../models/ResNet.keras')

