# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 18:24:17 2020

@author: sethi
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
print(tf.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


print(tf.test.is_built_with_cuda())

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('cell_images/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

training_set.class_indices

# Creating the Test set
test_set = test_datagen.flow_from_directory('cell_images/test',
                                            target_size = (64, 64),
                                            batch_size = 32, 
                                            class_mode = 'binary')

print(len(training_set))
print(len(test_set))

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Step 3  - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit_generator(training_set,
                  steps_per_epoch = len(training_set),
                  epochs = 25,
                  validation_data = test_set,
                  validation_steps = len(test_set))



import pickle
# open a file, where you ant to store the data
cnn_model = open('cnn_model.pkl', 'wb')

# dump information to that file
pickle.dump(cnn, cnn_model)




#Saving the Model
cnn.save("model.h5")
print("Saved model to disk")


from tensorflow.keras.models import load_model

# load model
model = load_model('model.h5')






import numpy as np
from keras.preprocessing import image
test_image = image.load_img('cell_images\\Prediction\\P1.png', target_size=(64,64) )
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis= 0)

result = model.predict(test_image)

if result[0][0] == 0:
    print("Parasatized")
else:
    print("Uninfected")    






