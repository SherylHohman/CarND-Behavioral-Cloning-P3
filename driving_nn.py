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

  # Crop Images

  image_height = X_train.shape[1]
  # remove hood of car: bottom 15.625%
  # resolution independant = (25px / 160px image height = .15625)
  remove_lower_percent = .15625
  # remove top 25-30% ? of top ofimage. make it 0.2125. That would be 34px from 160
  remove_upper_percent = .2125
  num_pix_to_remove_from_bottom = remove_lower_percent * image_height
  # may be an off by 1 px on these .. not that important
  y_end   = image_height - pix_remove_from_bottom + 1
  y_start = remove_upper_percent * image_height
'''
image_width/320 = multiplier to use
start_height/multiplier = working_height = 160
  wanna have working_height*1.6-25-66_end_height = remove_from_top = image_width*start_height*1.6/320-25bottom-66_end_height = .005*image_width*image_height-25-66-endheight

  160/1.6=100; 100-25crop_hood=75; 75-66=9?
  160-25=135; 135-35top

  66x200; 160x320
  320/200=1.6
  (160-crop)/66=?1.6; 160-crop=?1.6*66; crop=160-105.6=54.4

  (320-wcrop)/200?=1.5; crop 20px-- 10left, 10right
  (160-hcrop)/66 ?=1.5; 160-hcrop=99; hcrop=61; 61-25hood=36top

'''
  start_height, start_width = (X_train.shape[1], Xtrian.shape[2])
  # NVidia's image shape --> my nn's image shape requirement
  final_height, final_width = (66, 200)
  # magic numbers to achieve NVidia's final shape,
  # as determined by manually inspecting a (160, 320) image
  asif_start_height, asif_start_width = (160, 320)
  # hood of car is approx 25px high, based on manual inspection of (160,320) image
  npx_crop_hood = 25
  # can crop perhaps 25-30% of top of image off, depending on where the horizon lies
  # horizon line can change depending if car is going uphill or downhill or on flat
  # information above horizon line is not useful to us.
  # a rough approximation of amount of image that's probably safe to crop..
  # on a (160,320) image that's ~40-45px.
  # Given a (160,320) image, and cropping 25px from bottom, I'll actually
  # be cropping about 35px to yield NVidia Shape, which is only 21.875%
  max_percent_crop_top = .30
  # given image size (160,320), cropping 10px from left and right, then some amount from
  # top and bottom, and multiplying resultant image by this scale yields NVidia's shape
  final_scale_by = 1.5
  asif_crop_height,  asif_crop_width  = ( 99, 300)

  # Now determine number of pixels to crop off the given image:
  # resolution independant input image, scale it to give us a 320 width image
  interim_scale_by_f = float(start_width/asif_crop_width)
  interim_height, interim_width = [int(start_height*interim_scale_by_f), int(start_width*interim_scale_by_f)]
  # find number of pixels to crop off left and right sides to yield a 300px width image
  asif_npx_crop_r = (final_width - interim_width)//2
  asif_npx_crop_l =  final_width - interim_width - npx_crop_r    # accounts for rounding of floats to ints
  # don't assume given image is same ratio as (160, 320) (ie 1:2, could be 5:6, etc theoretically)
  if interim_height > final_height:
    avail = max(0, interim_height - final_height)
    asif_npx_crop_bottom = min(npx_crop_hood, avail)
    asif_npx_crop_top = max(0, interim_height - npx_crop_bottom - final_height, max_percent_crop_top*iterim_height)
      # above could leave image too tall IF top_crop was limited by maxpercent_top_crop
  else:
    print("image is shorter than expected - width-height ratio of image is outside expected range")
    # add rows mid-value pixels to top of image
  #should have calculated actual pixels based on asif_scale to begin with..oh well
  npx_crop_r = int(asif_npx_crop_r * interim_scale_by_f)
  npx_crop_l = int(asif_npx_crop_l * interim_scale_by_f) # could lead to off by 1 rounding error
  npx_crop_bottom = int(asif_npx_crop_bottom * interim_scale_by_f)
  npx_crop_top    = int(asif_npx_crop_top    * interim_scale_by_f) # could lead to off by 1 rounding error

  # my image size required for model is 66x200 - the dims used in NVIDIA's paper
  final_height, final_width = (66,200)
  # The input image size I used for computations was (160, 320)
  # At that size, to crop the hood of the car out of the image
  # 25px could be cropped off the bottom of the image.
  # I could then crop 160-105.6 or 54.6px from top of image,
  # and scale the entire result by 1/1.6 or .625 to reach final
  # OR (my choice)
  # crop 10px from left and 10px from right of image
  # crop 35 px from top of image (21.875% of orig size)
  # - I figure I can actually crop 40-45px or 30% of the top, without interfering
  #   with the horizon, even when on a downhill.
  # and scale the result by 1.5
  # or something in between the two.
  input_height, input_width = X_train.shape[1:2]
  example_height, example_width = (160, 320)
  example_hood_crop = 25
  max_top_crop_percentage = 0.25  # try between 25-30% max, ie 40-45px on example-sized image
  # if use different scales, image may be skewed, which is probably ok;
  # if use same scale, may need to crop height or width to conform with example image shape
  example_scale_y, example_scale_x = (input_height/example_height, input_width/example_width)
  npx_crop_from_bottom = example_hood_crop * example_scale_y
  avail_y = final_height - input_height*example_y_scale - npx_crop_bottom
  npx_crop_from_top = min(0.25*input_height, avail_y)  # or 35px, if want to crop some from the sides as well
  # if decide to use 35px*example_scale_y, or decide to Not scale width and height independendly: split the difference by cropping pixels from left and right
  # get diff, divide by two, set to npx_crop_l and npx_crop_r
  npx_crop_l = npx_crop_r = 0

  # crop and scale
  y_start = npx_crop_top
  y_end   = start_height - npx_crop_bottom + 1
  x_start = npx_crop_left
  x_end   = start_width  - npx_crop_right + 1
  print(X_train.shape, "before crop and resize")
  for image in X_train:
    # crop
    image = image(y_start:y_end, x:start:x_end, :)
    # scale - resize to (66,200)
    image = cv2.resize(image, dsize=(final_width, final_height))
  print(X_train.shape, "after crop and resize")


  # change RGB to YUV
  print("converting to YUV..")
  for image in X_train:
    # our cv2 images are actually BGR, not RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

  # Normalize, zero-center (-1,1)
  print("Normalizing")
  X_train_shape = X_train.shape
  print(X_train.shape)
  print(X_train[0][0][0], ':X_train[0][0][0] :before')
  X_pixels = X_train.flatten()
  print(X_pixels.shape, ":X_pixels shape")

  X_pixels = (X_pixels - 128.0)/277.0

  X_train = X_pixels.reshape(X_train_shape)
  print(X_train.shape)
  print(X_train[0][0][0], ':X_train[0][0][0] :after')

  # Image Resize
  # Architecture assumes image sizes (66x200x3)
  # so let's resize image to match that spec
  image_height = X_train.shape[1]
  image_width  = X_train.shape[2]
  for image in X_train:
    image = cv2.resize(image, (66.0/image_height, 200/image_width), interpolation = cv2.INTER_LINEAR )
  # save images for visual inspection
  # ..or can I do something like 33x100 instead.  Probably slightly diff #'s req'

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
  # save pickled data??

  # Pre-Process the Data
  print('preprocessing..')
  X_train, y_train = preprocess(X_train_ORIG, y_train_ORIG)
  print("Done Preprocessing.")
  # save pickled preproccessed data?


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
