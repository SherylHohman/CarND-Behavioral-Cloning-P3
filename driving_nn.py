# neural network for behavioural cloning driving behaviour that was captured in driving_log.csv
# reads training data from driving_log.csv
# trains a neural network from that data
# saves trained model as model.h5
# model.h5 can then be used to autonomously drive the car in the simulator that was used to produce the training data.

# Read data from CSV file
import csv
# read image
import cv2
import numpy as np
# for time_stamping the saved model's filename
import time
# for command line flags
import tensorflow as tf

# command line flags
flags = tf.app.flags
FLAGS = flags.FLAGS

# (remote is default cuz AWS costs $, thus less typing and fewer mistakes
env_default    = 'remote'
subdir_default = 'latest'

#'remote': need to parse the absolute image paths saved in driving_log.csv
#      for use on current machine (ie AWS)
#'local'  (not default): use image paths as saved in driving_log.csv
#      for use on current machine (ie AWS)
flags.DEFINE_string('env', env_default, "env: local training machine, or remote machine")
# name_of_directory under './data/' that has the training data to use
flags.DEFINE_string('subdir', subdir_default, "subdir that training data is stored in, relative to ./data/")

def load_data(ENV, SUBDIR):

  def get_current_path_to_images(local_path_to_images, ENV):

    if ENV == 'local':
      return local_path_to_images

    if ENV == 'remote':
      # filename is last segment of full_path
      # my data is taken from windows machine
      #   splitting on '\\' instead of '\' as python is interpreting
      #   the latter as an escape character, so I must escape the escape
      # if data is taken from non-windows, split would be '/'

      # determine if remote_image_path is from Windows, or Non-Windows machine
      if local_image_path.find('/')    != -1:
        # looks like linux style path
        split_str = '/'
      elif local_image_path.find('\\') != -1:
        # Note: must type '\\' in order to search on '\',
        #   python string thinks I'm trying to escape something
        # looks like windows style path
        split_str = '\\'
      else:
        # hmm.. dunno, perhaps a filename with no path is here, else error
        # split_string setting may thus be arbitrary
        print("\n---remote_image_path: ", remote_image_path)
        split_str = '/'

      filename = local_path_to_images.split(split_str)[-1]
      remote_image_path = driving_log_path + 'IMG/' + filename
      return remote_image_path

    else:
      print("-----ENV='", ENV, "': incorrect flag value\n")
      return("")
      # need to END program execution here.
      assert ("ENV:" == "incorrect flag value supplied")
      # TODO: there's a proper way to end execution on error



  driving_log_filename  = 'driving_log.csv'
  driving_log_path      = './data/' + SUBDIR + '/'

  lines = []
  with open(driving_log_path + driving_log_filename) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      lines.append(line)

    # remove header, if exists. Note: all columns are currently strings
    # if steering_column cannot be cast to a number --> it is a heading
    possible_steering_angle = lines[0][3]
    try:
      float(possible_steering_angle)
    except:
      print("removing header row \n")
      del lines[0]

  # for field in lines[0]:
  #   print(field)

  # gather data for features and "labels"
  # initialize feature set containers
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
    local_image_path = str(line[0])
    current_image_path = get_current_path_to_images(local_image_path, ENV)
    # load image using openCV
    image = cv2.imread(current_image_path)
    camera_1_images.append(image)

    # "labels" (measurements)
    steering_angle = float(line[3])
    steering_angles.append(steering_angle)

  # convert to numpy arrays, and save as train and "label" datasets
  X_train = np.asarray(camera_1_images)
  y_train = np.asarray(steering_angles[:])

  return X_train, y_train

def main(_):

  # print("\nFLAGS\n", FLAGS.env, FLAGS.subdir, "\n")
  X_train, y_train = load_data(FLAGS.env, FLAGS.subdir)

  image_input_shape = X_train.shape[1:]
  # for regression, we want a single value, ie steering angle predicted.
  # unlike classification, we do not map to a set of predefined values, or calculate probabilites for predefined class ids
  output_shape = 1

  # print(image_input_shape)
  # print(output_shape)

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
  path_to_saved_models = './trained_models/'
  model_filename = 'model_' + model_timestamp + '.h5'
  model.save(path_to_saved_models + model_filename)
  print("Model Saved as ", path_to_saved_models + model_filename)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()




# to test the model locally (in anaconda)
#     at the command command line, type:
# python drive.py model_{path_to_saved_models}model_{model_timestamp}.h5
# examples:
# python drive.py model.h5
# python drive.py ./trained_models/model_170417_1741.h5

# (if using docker, see insturctions in "Running Your Network" lesson)
#  https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/1ff2cbb5-2d9e-43ad-9424-4546f502fe20