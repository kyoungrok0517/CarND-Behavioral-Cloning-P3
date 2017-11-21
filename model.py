import csv
import os
import cv2
import numpy as np
import sklearn
from random import shuffle

BASE_DIR = './data'

samples = []
with open(os.path.join(BASE_DIR, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                correction = 0.2
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                angle = [center_angle, left_angle, right_angle]

                for i in range(3):
                    name = os.path.join(BASE_DIR, 'IMG', batch_sample[i].split('\\')[-1])
                    image = cv2.imread(name)
                    images.append(image)
                    angles.append(angle[i])
                    images.append(cv2.flip(image, 1))
                    angles.append(angle[i]*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
BATCH_SIZE = 32
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

# the model
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D
from keras import optimizers

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))     
model.add(Cropping2D(cropping=((70,25),(0,0))))                             
model.add(Convolution2D(24, (5, 5), activation='relu'))
model.add(Convolution2D(36, (5, 5), activation='relu'))
model.add(Convolution2D(48, (5, 5), activation='relu'))
# model.add(Dropout(0.5))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
# model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

from keras.optimizers import Adam

EPOCHS = 5
model.compile(loss='mse', optimizer=Adam(lr=0.0001))
history = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/BATCH_SIZE, validation_data=validation_generator, 
                        validation_steps=len(validation_samples)/BATCH_SIZE, epochs=EPOCHS)
model.save('model.h5')

# ### print the keys contained in the history object
print(history.history.keys())

# ### plot the training and validation loss for each epoch
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

exit()