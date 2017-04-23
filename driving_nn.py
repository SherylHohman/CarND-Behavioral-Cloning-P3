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

subdir_default = 'latest'
#subdir_default = 'sample_training_data'
epochs_default = '10'
#batch_size_default = '32'#'128'  # is keras default batch size==32 ??

# name_of_directory under './data/' that has the training data to use
flags.DEFINE_string('subdir', subdir_default, "subdir that training data is stored in, relative to ./data/")
flags.DEFINE_string('epochs', epochs_default, "EPOCHS")
#flags.DEFINE_string('batch_size', batch_size_default, "batch size")


def load_data(SUBDIR):

  def get_current_path_to_images(local_path_to_images):

      if local_image_path.find('/')    != -1:
        # likely a linux style path
        split_str = '/'
      elif local_image_path.find('\\') != -1:
        # Note: must type '\\' in order to search on '\',
        #   python string thinks I'm trying to escape the second quotation mark
        # likely a windows style path
        split_str = '\\'
      else:
        # hmm.. dunno; perhaps a filename with no path; else an unknown error
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
  y_train = np.asarray(steering_angles)

  return X_train, y_train

def preprocess(X_train, y_train):
  # PreProcess:
  # change RGB to YUV
  print("converting to YUV..")
  for image in X_train:
    # our cv2 images are actually BGR, not RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

  print("Normalizing")
  # Normalize, zero-center (-1,1) could instead do a MinMax thing
  X_train_shape = X_train.shape
  print(X_train.shape)
  print(X_train[0][0][0], ':X_train[0][0][0] :before')
  X_pixels = X_train.flatten()
  print(X_pixels.shape, ":X_pixels shape")

  X_pixels = (X_pixels - 128.0)/277.0

  X_train = X_pixels.reshape(X_train_shape)
  print(X_train.shape)
  print(X_train[0][0][0], ':X_train[0][0][0] :after')

  # Crop: hood of car; some amount above the horizon
  #print("Cropping..")
  # Data Augmentation (apply during training)

  return(X_train, y_train)


def main(_):

  from keras.models import Sequential
  from keras.layers.core import Dense, Activation, Flatten
  from keras.layers.convolutional import Convolution2D

  # get flag values from command line (or defaults)
  print('\nflag values:')
  EPOCHS = int(FLAGS.epochs)
  print(EPOCHS, "EPOCHS")
  SUBDIR = FLAGS.subdir
  print(SUBDIR, ": subdir of './data/' that training data resides in")
  #batch_size = int(FLAGS.batch_size)
  #print('batch_size', batch_size)
  print()

  # load raw data
  X_train_ORIG, y_train_ORIG = load_data(SUBDIR)
  print('dataset shapes', X_train_ORIG.shape, y_train_ORIG.shape, "\n")

  # Pre-Process the Data
  print('preprocessing..')
  X_train, y_train = preprocess(X_train_ORIG, y_train_ORIG)
  print("Done Preprocessing.")


  # for regression, we want a single value, ie steering angle predicted.
  image_input_shape = X_train.shape[1:]
  output_shape = 1


  ## Define Model

  model = Sequential()

  # 3 conv layers with kernal 5, stride 2, relu activation
  filter_size = 5
  strides = (2,2)
  num_output_filters = image_input_shape[-1]  # 3
  conv_params = {'nb_filter': num_output_filters,
                 'nb_row': filter_size,
                 'nb_col': filter_size,
                 'subsample': strides,
                 'activation': 'relu',
                 'input_shape': image_input_shape
                }

  # (66,200,3)->(31,98,24)
  conv_params['nb_filter'] = num_output_filters = 24
  model.add(Convolution2D(**conv_params))
  #model.add(Convolution2D(input_shape=image_input_shape, stride=2, activation='relu'))
  # input_shape is only ever passed in to the first layer
  del conv_params['input_shape']

  # #  (31,98,24)->(14,47,36)
  conv_params['nb_filter'] = num_output_filters = 36
  model.add(Convolution2D(**conv_params))

  # #  (14,47,36)->(5,22,48)
  conv_params['nb_filter'] = num_output_filters = 48
  model.add(Convolution2D(**conv_params))

  # 2 conv layers with kernal 3, stride none (1), relu activation
  filter_size = 3
  strides = (1,1)
  num_output_filters = 64
  conv_params['nb_row'] = filter_size
  conv_params['nb_col'] = filter_size
  conv_params['subsample'] = strides
  conv_params['nb_filter'] = num_output_filters
  # ( 5,22,48)->( 3,20,64)
  model.add(Convolution2D(**conv_params))
  #  ( 3,20,64)->( 1,18,64)
  model.add(Convolution2D(**conv_params))

  #  ( 1,18,64)->(1164)
  model.add(Flatten())

  # 3 fully connected layers
  #  (1164) -> (100)
  model.add(Dense(100))
  model.add(Dense(50))
  #model.add(Dense(10))  #?? 1 too many dense layers ?
  model.add(Dense(1))
  # no softmax or maxarg on regression network; just the raw output value

  ## TRAIN Model
  print("Training Model..")
  # for regression: use mse. no cross_entropy, no softmax
  model.compile(loss='mse', optimizer='adam')
  model.fit(X_train, y_train, shuffle=True, validation_split=0.2, nb_epoch=EPOCHS)

  ## SAVE model: h5 format for running in autonomous mode on simulator
  print("Saving model..")
  model_timestamp = time.strftime("%y%m%d_%H%M")
  path_to_saved_models = './trained_models/'
  model_filename = 'model_' + model_timestamp + '_' + SUBDIR +'.h5'
  model.save(path_to_saved_models + model_filename)
  print("Model Saved as ", path_to_saved_models + model_filename)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()



# ------
# to test the model locally (in anaconda)
#     at the command command line, type:
# python drive.py model_{path_to_saved_models}model_{model_timestamp}_{SUBDIR}.h5

# examples:
# python drive.py model.h5
# python drive.py ./data/trained_models/model_170422_2224_sample_training_data.h5
# python ../drive.py ./trained_models/model_170417_1741_sample_training_data.h5

# (if using docker, see instructions in Udacity Lesson: "8 Running Your Network")
