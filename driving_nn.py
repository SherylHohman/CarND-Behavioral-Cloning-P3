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

#subdir_default = 'latest'
subdir_default = 'sample_training_data'
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

  def crop_and_scale_images(X_train):

    # # Architecture assumes image sizes (66x200x3)
    # # resize image to match that spec

    # TODO
    # default interpolation = cv2.INTER_LINEAR
    # can set interpolation, depending if upsizing, or downsizing.
    # downsample: CV_INTER_AREA interpolation,
    # upsample: CV_INTER_CUBIC (slow) or CV_INTER_LINEAR

    # TODO
    # save some images for visual inspection

    # Crop Images
    #image size required for model is 66x200 == the dims used in NVIDIA's paper
    final_height, final_width = (66,200)
    # The input image size I used for computations was (160, 320)
    # At that size, to crop the hood of the car out of the image
    #  - 25px could be cropped off the bottom of the image.
    #  - From the top, I figure I can actually crop 25-30%, or 40-45px of the top
    #    without interfering with the horizon even when driving uphill.
    #  - I could then..
    # crop 160-105.6 or 54.6px from top of image,
    #   and scale the entire result by 1/1.6 or .625 to reach final
    # OR
    # crop 10px from left and 10px from right of image
    #   crop 35 px from top of image (21.875% of orig size)
    #   and scale the (99,300) result by 1/1.5 to achieve (66,200)
    # OR something in between the two, OR scale height, width independantly (skew)
    input_height, input_width = X_train.shape[1], X_train.shape[2]
    basis_height, basis_width = (160, 320)
    # magic numbers to achieve NVidia's final shape,
    # as determined by manually inspecting a (160, 320) image
    basis_hood_crop = 25
    # can try between 25-30% max, ie 40-45px on example-sized image
    max_top_crop_percentage = 0.25
    #magic number to not crop anything from sides, and minimize stretching from scaling x and y differently
    basis_top_crop2_minimize_stretch = 29 #36 for even 1.5x and 1.6y scaling
    # if use different scales, image may be skewed, which is possibly better than
    #   discarding information from left and right of image;
    # otherwise crop from sides to use the same x and y scale and avoid stretching
    # basis_scale would resize image to (160, 320)
    basis_scale_y, basis_scale_x = (input_height/basis_height, input_width/basis_width)
    npx_crop_from_bottom = basis_hood_crop * basis_scale_y
    avail_y = basis_scale_y*(input_height - npx_crop_from_bottom) - final_height
    min_horizon_crop    = input_height*max_top_crop_percentage            #1
    min_distortion_crop = basis_top_crop2_minimize_stretch*basis_scale_y  #2
    npx_crop_from_top   = min(avail_y, min_distortion_crop)  # add #1 and/or #2
    # if decide to Not scale width and height independendly:
    #  split the difference by cropping pixels from left and right
    #  get diff, divide by two, set to npx_crop_l and npx_crop_r
    # npx_crop_from_left = npx_crop_from_right = 0
    # print(npx_crop_from_bottom, npx_crop_from_top, npx_crop_from_left, npx_crop_from_right)
    print(npx_crop_from_bottom, npx_crop_from_top)
    print((input_height-npx_crop_from_bottom-npx_crop_from_top)/final_height, input_width/final_width, "scale_x, scale_y")

    # crop and scale
    y_start = int(npx_crop_from_top)
    y_end   = int(input_height - npx_crop_from_bottom + 1)
    # x_start = int(npx_crop_from_left)
    # x_end   = int(input_width  - npx_crop_from_right + 1)
    print(X_train.shape, "before crop and resize")

    # can't store resized image back into np.array of input image shape
    X_resized=[]
    for image in X_train:
      # crop
      # image = image[y_start:y_end, x_start:x_end, :]
      image = image[y_start:y_end, :, :]
      #print(image.shape, "after crop")
      # scale to (66,200),
      # results in stretched image if cropped image isn't proportional
      image = cv2.resize(image, dsize=(final_width, final_height))
      X_resized.append(image)
    X_resized = np.asarray(X_resized)
    print(X_resized.shape, "X_resized")
    return X_resized


  print("cropping and scaling images..")
  X_train = crop_and_scale_images(X_train)
  print(X_train.shape, "after crop and resize\n")

  # change RGB to YUV
  print("converting to YUV..")
  for i in range(X_train.shape[0]):
    image = X_train[i]
    # our cv2 images are actually BGR, not RGB
    X_train[i] = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

  # Normalize, zero-center (-1,1)
  print("Normalizing..")
  X_train_shape = X_train.shape
  print(X_train.shape)
  print(X_train[0][0][0], ':X_train[0][0][0] :before')
  X_pixels = X_train.flatten()
  print(X_pixels.shape, ":X_pixels shape")

  X_pixels = (X_pixels - 128.0)/277.0

  X_train = X_pixels.reshape(X_train_shape)
  print(X_train.shape)
  print(X_train[0][0][0], ':X_train[0][0][0] :after\n')


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

  # load raw training data
  X_train_ORIG, y_train_ORIG = load_data(SUBDIR)
  print('dataset shapes', X_train_ORIG.shape, y_train_ORIG.shape, "\n")
  # save pickled data??

  # Pre-Process the Data
  print('preprocessing..')
  X_train, y_train = preprocess(X_train_ORIG, y_train_ORIG)
  print("Done Preprocessing.\n")
  # TODO:
  # save pickled preproccessed data
  # add command line flag to skip above steps..
  #.. start HERE by reading this data in instead

  # Visualize Data
  # sample cropped/resized images (center, left, right)
  # distribution of steering angles (see how dist changes with augmentation)
  # - orig data, center camera only
  # - plus left, right cameras after steering compenation
  # - sample images at max -1, +1 steering angles
  # - plus augmentation added steering angles
  # examples of augmented data
  # - left/right cameras, if used
  # - skew, shift, flip, etc, if used


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
# python drive.py ./trained_models/model_170422_2224_sample_training_data.h5
# python ../drive.py ./model_170417_1741_sample_training_data.h5

# (if using docker, see instructions in Udacity Lesson: "8 Running Your Network")
