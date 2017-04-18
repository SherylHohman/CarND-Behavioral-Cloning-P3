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
# for progress bar
import sys

# command line flags
flags = tf.app.flags
FLAGS = flags.FLAGS

# (remote is default cuz AWS costs $, thus less typing and fewer mistakes
subdir_default = 'latest'
#batch_size_default = '32'#'128'  #keras default batch size is 32 ??
epochs_default = '10'

# name_of_directory under './data/' that has the training data to use
flags.DEFINE_string('subdir', subdir_default, "subdir that training data is stored in, relative to ./data/")
#flags.DEFINE_string('batch_size', batch_size_default, "batch size")
flags.DEFINE_string('epochs', epochs_default, "EPOCHS")


def load_data(SUBDIR):

  def get_current_path_to_images(local_path_to_images):

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


  ## Read Data File

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
      print("removing header row")
      del lines[0]
    print('finished reading driving_log.csv\n')

  # for field in lines[0]:
  #   print(field)


  ## Parse Data

  # features (input data: images from cameras)
  camera_1_images = []
  #camera_1_images, camera_2_images, camera_3_images = [[], [], []]
  #images = []

  # outputs (nn should output: steering_angles) (remaining items are optional)
  steering_angles = []    # values: (-1, 1)
  #throttle, brake, speed = [[], [], []]  # values: (0,1), (0), (0, 30)
  #measurements = []


  print('parsing data')
  for i, line in enumerate(lines):

    # progress bar to show feedback on data parsing
    # def update_progress(i):
    barLength = 30 # Modify this to change the length of the progress bar
    progress = float(i/len(lines))
    if i == len(lines)-1:
        progress = 1
    block = int(round(barLength*progress))
    #padding = " "*10
    text = "\r          [{}] {}%".format("="*block + "."*(barLength-block), int(progress*100))
    sys.stdout.write(text)
    sys.stdout.flush()

    # sys.stdout.write('\r')
    # sys.stdout.write("%-20s] %d%%" % ('='*i, 5*i))
    # sys.stdout.flush()

    # data stored in each line:
    # line: [center_camera_image_path, L_camera_image_path, R_camera_image_path, steering_angle, throttle, brake, speed]

    # features (images)
    local_image_path = str(line[0])
    current_image_path = get_current_path_to_images(local_image_path)

    # load image using openCV
    image = cv2.imread(current_image_path)
    camera_1_images.append(image)

    # "labels" (measurements)
    steering_angle = float(line[3])
    steering_angles.append(steering_angle)

  # end of parsing data
  print()

  # convert to numpy arrays, and save as train and "label" datasets
  X_train = np.asarray(camera_1_images)
  y_train = np.asarray(steering_angles[:])

  return X_train, y_train

def main(_):
  #implement simple regression network using keras

  from keras.models import Sequential
  from keras.layers.core import Dense, Flatten

  print('\nflag values:')
  EPOCHS =     int(FLAGS.epochs)
  print(EPOCHS, "EPOCHS")
  subdir = FLAGS.subdir
  print(subdir, ": subdir of './data/' that training data resides in")
  #batch_size = int(FLAGS.batch_size)
  #print('batch_size', batch_size)
  print()

  X_train, y_train = load_data(FLAGS.subdir)
  print('dataset shapes', X_train.shape, y_train.shape, "\n")

  image_input_shape = X_train.shape[1:]
  output_shape = 1
  # for regression, we want a single value, ie steering angle predicted.
  # unlike classification, we do not map to a set of predefined values,
  #  or calculate probabilites for predefined class ids

  # define model
  model = Sequential()
  model.add(Flatten(input_shape=image_input_shape))
  model.add(Dense(output_shape))
  # no activation on a single layer network / output layer
  # no softmax or maxarg on regression network; just the raw output value

  # for regression, we use mse, no cross_entropy flavors, no softmax
  model.compile(loss='mse', optimizer='adam')
  model.fit(X_train, y_train, shuffle=True, validation_split=0.2, nb_epoch=EPOCHS)

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

# (if using docker, see instructions in "Running Your Network" lesson)
#  https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/1ff2cbb5-2d9e-43ad-9424-4546f502fe20