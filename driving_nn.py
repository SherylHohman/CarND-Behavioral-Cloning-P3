# neural network for behavioural cloning driving behaviour that was captured in driving_log.csv
# reads training data from driving_log.csv
# trains a neural network from that data
# saves trained model as model.h5
# model.h5 can then be used to autonomously drive the car in the simulator that was used to produce the training data.

# Read data from CSV file
import csv
import cv2
import numpy as np
import time

# TODO: add flags so can input custom driving_log_path, in case save several versions
driving_log_filename = 'driving_log.csv'
driving_log_path = './data/latest_training_data/'

lines = []
with open(driving_log_path + driving_log_filename) as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

print(lines[0], '\n')
for field in lines[0]:
  print(field)

# separate lines of data into types of data ?? dunno how to say this
# gather features and "labels"
# initialize feature set
#images = []
camera_1_images = []
#camera_1_images, camera_2_images, camera_3_images = [[], [], []]

#initialize measurements: outputs/label sets
#measurements = []
steering_angles = []    # values: (-1, 1)
#throttle, brake, speed = [[], [], []]  # values: (0,1), (0), (0, 30)

# line: [camera1_image_path, camera2_image_path, camera3_image_path, steering_angle, throttle, brake, speed]
for line in lines:
  # features (images)
  image_path = str(line[0])
  # load image using openCV
  image = cv2.imread(image_path)
  camera_1_images.append(image)

  # "labels" (measurements)
  steering_angle = float(line[3])
  steering_angles.append(steering_angle)

# convert to numpy arrays, and save as train and "label" datasets
X_train = np.array(camera_1_images)
y_train = np.array(steering_angles[:])

image_input_shape = X_train.shape[1:]
# for regression, we want a single value, ie steering angle predicted.
# unlike classification, we do not map to a set of predefined values, or calculate probabilites for predefined class ids
output_shape = 1

print(image_input_shape)
print(output_shape)

#implement simple regression network with keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten


model = Sequential()
model.add(Flatten(input_shape=image_input_shape))
model.add(Dense(output_shape))
# no activation on a single layer network
# no softmax or maxarg on regression network
# just the raw output value

# for regression, we use mse, no cross_entropy flavors, no softmax
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, shuffle=True, validation_split=0.2)

# save model in h5 format for running in automode on simulator
print("Saving model..")
model_timestamp = time.strftime("%y%m%d_%H%M")
path_to_saved_model = './trained_models/'
model_filename = 'model_' + model_timestamp + '.h5'
model.save(model_filename)
print("Model Saved as ", model_filename)

