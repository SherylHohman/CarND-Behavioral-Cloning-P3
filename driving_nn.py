# neural network for behavioural cloning driving behaviour that was captured in driving_log.csv
# reads training data from driving_log.csv
# trains a neural network from that data
# saves trained model as model.h5
# model.h5 can then be used to autonomously drive the car in the simulator that was used to produce the training data.

# Read data from CSV file
import csv
import cv2
import numpy as np

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

print(X_train.shape)
print(y_train.shape)

