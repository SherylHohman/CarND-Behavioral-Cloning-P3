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
pickle_default = ''

# name_of_directory under './data/' that has the training data to use
flags.DEFINE_string('subdir', subdir_default, "subdir that training data is stored in, relative to ./data/")
flags.DEFINE_string('epochs', epochs_default, "EPOCHS")
# if already preprocessed this data, skip parsing raw data, and skip preprocessing
# by supplying the base name of a pnz file (yep, pickle didn't work on np.array)
# flag name remains
flags.DEFINE_string('pickle', pickle_default, "preprocessed data file to read, relative to ./data/ (omit the '.pnz' filetype suffix")


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
    barLength = 30
    progress = float(i/len(lines))
    if i == len(lines)-1:
        progress = 1
    block = int(round(barLength*progress))
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

def preprocess(X_train):
  # takes in a list or array of images
  # ie: [image] - list with a single image,
  #     if live preprocessing a simulator image in autonomous mode
  # or: X_train - array of images, if training

  def crop_and_scale_images(images):
    #takes in an array of images: ie X_train, or [image] (as np.array, not list)

    # # Architecture assumes image sizes (66x200x3)
    if verbose:
      print("cropping and scaling images..")
      print(images.shape, "before crop and resize")

    # TODO
    # save some images for visual inspection

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
    # OR something in between the two, OR scale height, width independantly (stretch)

    input_height, input_width = images.shape[1], images.shape[2]
    #image size required for model: (66,200) == dims used in NVIDIA's paper
    final_height, final_width = (66,200)
    basis_height, basis_width = (160, 320)

    # interpolation
    if (input_height < basis_height) and (input_width < basis_width):
      interpolation = cv2.INTER_AREA    # best for downsampling
    else:
      interpolation = cv2.INTER_LINEAR  # better(slow) upsizing=CV_INTER_CUBIC

    # magic numbers to achieve NVidia's final shape
    #   based on manual inspection of a (160, 320) image
    basis_hood_crop = 25
    # magic number to minimize stretching without cropping from sides
    basis_top_crop2_minimize_stretch = 29  # (1.606 x 1.6); 36 for (1.5x1.6
      # can crop from L and R to avoid stretching image
      # however, image stretching may be preferred to cropping L and R pixels..
    # maximum crop to horizon is perhaps about 25-30% of input image
    # 40-48px on (160,320)
    max_top_crop_percentage = 0.30   # ~48px  (stretches image: 1.3x1.6)

    # basis_scale would resize input image to (160, 320)
    basis_scale_y = input_height/basis_height
    basis_scale_x = input_width/basis_width

    # crop out hood of car
    npx_crop_from_bottom = int(basis_hood_crop * basis_scale_y)

    # crop out some amount above the horizon
    avail_y = int(basis_scale_y*(input_height - npx_crop_from_bottom) - final_height)
    # to minimize the amount of horizon left in photo.
    # Induces stretching to achieve correct aspect ratio
    max_horizon_crop    = int(input_height*max_top_crop_percentage)               #1
    # minimize distortion by cropping an amount off the top to get as close to the final aspect ratio as possible
    min_distortion_crop = int(basis_top_crop2_minimize_stretch*basis_scale_y)     #2

    #npx_crop_from_top   = min(avail_y, min_distortion_crop)  # add #1 and/or #2
    npx_crop_from_top = min(avail_y, max_horizon_crop)

    # want to crop sides, to achieve final aspect ratio without stretching..
    #   change below to not be zero
    npx_crop_from_left = npx_crop_from_right = 0

    if verbose:
      print(npx_crop_from_bottom, npx_crop_from_top, npx_crop_from_left, npx_crop_from_right, "number of pixels to crop (b,t,l,r)")
      print((input_height - npx_crop_from_bottom - npx_crop_from_top, input_width - npx_crop_from_left - npx_crop_from_right), "anticipated size after crop")
      print("scale_y:", (input_height - npx_crop_from_bottom - npx_crop_from_top)/final_height, " scale_x:", input_width/final_width)

    # crop and scale
    y_start = npx_crop_from_top
    y_end   = input_height - npx_crop_from_bottom + 1
    x_start = npx_crop_from_left
    x_end   = input_width  - npx_crop_from_right + 1

    # "images" array expects input_shaped images; so,
    # must store resized images to new list/array
    resized=[]
    for image in images:
      # crop
      image = image[y_start:y_end, x_start:x_end, :]
      # scale to (66,200)
      # image stretches if cropped image isn't same aspect ratio as final
      # It's fine to stretch the image, I think - as long as the automated driving images are stretched in the same manner as the training images
      image = cv2.resize(image, dsize=(final_width, final_height), interpolation=interpolation)
      resized.append(image)
    resized = np.asarray(resized)

    if verbose:
        print(resized.shape, "after crop and resize\n")

    return resized

  def convert_to_YUV(images):
    # takes in an array of images ie: [image], or X_train
    if verbose:
        print("converting to YUV..")
    images_yuv = []
    for i in range(len(images)):
      image = images[i]
      # our images (read in via cv2) are actually BGR, not RGB
      # ?? is this the same for images fed in via autonomous mode of simulator ?
      image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
      images_yuv.append(image)
    images_yuv = np.asarray(images_yuv, np.float64)
    if verbose:
        print(images_yuv.shape, "after YUV conversion\n")
    return images_yuv

  def normalize_pixels(matrix):
    # takes in an image(array) or an array of image(arrays):
    # ie image, [images] (as an np.array), X_train
    # cannot be a list
    if verbose:
      print("Normalizing..")
      print(matrix[0][0][0][:], ':matrix[0][0][0] :before')

    # Normalize, zero-center (-1,1)
    matrix_shape = matrix.shape
    pixels = matrix.flatten()
    pixels = (pixels - 128.0)/277.0
    matrix = pixels.reshape(matrix_shape)

    if verbose:
        print(matrix[0][0][0][:], ':matrix[0][0][0] :after')
        print(matrix.shape, "after Normalization\n")
    return matrix


  # begin preprocessing

  if len(X_train)>1:
    # training: give feedback on progress
    verbose = True
  else:  # simulator is driving in autonomous mode
    verbose = False

  # TODO:
  # might be more efficient to have outter loop cycle through all images
  #  calling each function on a single image only
  #  if so, would need to convert back to np.asarray only once..
  #  then again, may not matter..

  # drive.py must pass in [image_array] as an np.array, cannot be a list

  # crop and scale images to (66, 200)
  X_train = crop_and_scale_images(X_train)

  # change RGB to YUV
  X_train = convert_to_YUV(X_train)

  # Normalize, zero-center (-1,1)
  X_train = normalize_pixels(X_train)

  return(X_train)

def stratified_dataset_split(X_all, y_all, training_proportion=0.80):
    import numpy as np
    def get_indexes_for_split(y_train):
        y_train=np.array(y_train)
        # create buckets for steering angles: positive, negative, zero
        buckets = ['-1', '0', '1']
        # since buckets are known, create them all, init them all to empty sets
        steering_angles = {}
        for bucket in buckets:
            steering_angles[bucket] = []

        # get indexes of images with zero, positive, and negative steering angles
        for i in range(len(y_train)):
          if y_train[i] == 0:
            steering_angles['0'].append(i)
          elif y_train[i] < 0:
            steering_angles['-1'].append(i)
          else:
            steering_angles['1'].append(i)

        # shuffle the indexes of images in each bucket
        for bucket in buckets:
            np.random.shuffle(steering_angles[bucket])

        # siphon off training_proportion of each list into training set;
        #   remaining become validation set
        training_set_indexes, validation_set_indexes = [ [], [] ]
        for bucket in buckets:
            num_images_in_bucket = len(steering_angles[bucket])
            num_training = int((training_proportion * num_images_in_bucket) // 1)
            training_set_indexes   += steering_angles[bucket][:num_training]
            validation_set_indexes += steering_angles[bucket][num_training:]
        num_train= len(training_set_indexes)
        num_valid= len(validation_set_indexes)
        print(num_train/len(y_train), num_valid/len(y_train), num_train, num_valid, num_train+num_valid, len(y_train))
        return training_set_indexes, validation_set_indexes

    # get indexes to use for training and validation sets
    training_indexes, validation_indexes = get_indexes_for_split(y_all)
    # create new training set
    X_train_new, y_train_new = ([],[])
    for training_index in training_indexes:
            X_train_new.append(X_all[training_index])
            y_train_new.append(y_all[training_index])
    # create validation set
    X_valid_new, y_valid_new = ([], [])
    for validation_index in validation_indexes :
            X_valid_new.append(X_all[validation_index])
            y_valid_new.append(y_all[validation_index])
    # verification
    assert( len(X_train_new) == len(y_train_new) )
    assert( len(X_valid_new) == len(y_valid_new) )
    print("train: ", len(X_train_new)/len(X_all), "valid", len(X_valid_new)/len(X_all) )

    # return training and validation sets as numpy arrays
    return np.asarray(X_train_new), np.asarray(y_train_new), np.asarray(X_valid_new), np.asarray(y_valid_new)


def main(_):

  from keras.models import Sequential
  from keras.layers.core import Dense, Activation, Flatten
  from keras.layers.convolutional import Convolution2D
  from keras.callbacks import ModelCheckpoint, EarlyStopping
  from keras.preprocessing.image import ImageDataGenerator

  # get flag values from command line (or defaults)
  print('\nflag values:')
  EPOCHS = int(FLAGS.epochs)
  print(EPOCHS, "EPOCHS")
  SUBDIR = FLAGS.subdir
  print(SUBDIR, ": subdir of './data/' that training data resides in")
  NPZ_FILE = FLAGS.pickle
  if NPZ_FILE != "":
    print(NPZ_FILE, ": reading preprocessed data file")
  print()

  if NPZ_FILE == "":
    # load raw training data
    X_train_ORIG, y_train_ORIG = load_data(SUBDIR)
    print('dataset shapes', X_train_ORIG.shape, y_train_ORIG.shape, "\n")

    # Pre-Process the Data
    print('preprocessing..')
    X_train = preprocess(X_train_ORIG)
    y_train = y_train_ORIG
    print("Done Preprocessing.")

    # save preprocessed data as npz file for later-reuse
    print("  Saving Preprocessed data for later re-use..")
    npz_filename = "./data/" + SUBDIR + ".npz"
    np.savez(npz_filename, X_train=X_train, y_train=y_train)
    print("  Preprocessed training data saved as:", npz_filename, "\n")

  else:
    # load preprocessed X_train and y_train data
    npz_filename = './data/' + NPZ_FILE + '.npz'
    print("reading preprocessed data from:", npz_filename)
    npzfile = np.load(npz_filename)
    X_train = npzfile['X_train']
    y_train = npzfile['y_train']

    print(X_train.shape, y_train.shape, "preprocessed shapes: X_train, y_train\n")

  # TODO:
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

  batch_size = 32   # keras' default

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

  # Define Loss function, and Optimization function
  # for regression: use mse. no cross_entropy; no softmax.
  model.compile(loss='mse', optimizer='adam')

  ## SAVE model: h5 format for running in autonomous mode on simulator
  model_timestamp = time.strftime("%y%m%d_%H%M")
  path_to_saved_models = './trained_models/'
  model_filename = 'model_' + model_timestamp + '_' + SUBDIR +'.h5'

  ## TRAIN Model
  print("Training Model..")

  # generator_fit does not have an argument "validation_split" to automatically separate out validation data, so must do this myself
  #   either a validation_set generator, or a (X_valid, y_valid) tuple MUST be passed in to the "fit_generator" function as validation_data=xx)
  X_valid, y_valid = [], []
  print(X_train.shape, y_train.shape, ": before split, train shapes")
  # split X_train, y_train into training and validation sets
  X_train, y_train, X_valid, y_valid = stratified_dataset_split(X_train, y_train, training_proportion=0.8)
  print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, ": after split, train and valid data")

  # at each epoch, save the best model so far, as defined by lowest validation loss
  callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1),
    ModelCheckpoint(path_to_saved_models+model_filename, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    ]

  ## Data Augmentation
  def custom_data_generator(X_train, y_train, batch_size):
    # based on keras.image.ImageDataGenerator()
    #  ..Generate minibatches of image data with real-time data augmentation

    # current augmentation:
    # flip half the images in each batch (hence the X_train) horizontally,
    #   and adjust steering angle accordingly

    # if only want a percentage of the x's with a zero value, then
    # samples_per_epoch will be:
    # number_samples_where_y_is_NOT_zero +
    #      percentage_0s_to_use * number_of_samples_where_y_IS_zero

    image_row_axis, img_col_axis, channel_axis = (0,1,2)

    def flip_axis(x, axis):
      x = np.asarray(x).swapaxes(axis, 0)
      x = x[::-1, ...]
      x = x.swapaxes(0, axis)
      return x

    def horizontal_flip_50_50(image, label):
      error_checking_input = (image[0][0][0], label)
      if np.random.random() < 0.5:
        image = flip_axis(image, img_col_axis)
        if label != 0:
          label = -1 * label
        assert((image[0][-1][0], label) == (error_checking_input[0], -error_checking_input[1]))
      else:
        assert( (image[0][0][0], label) == error_checking_input)
      return image, label

    # def generate_batches(X_train, y_train, batch_size):
    # TODO: re-shuffle X_train at each epoch?
    print(X_train.shape, "\ntrain shape, entering custom generator")
    x_batch, y_batch = [], []
    i=0
    while (True):
      print("how do I reset X_train / generator for new epoch?")
      # while i < len(X_train):

      # final batch_size of an epoch == remaining elements in X_train..
      this_batch_size = min(batch_size, len(X_train)-i)
      i_start, i_end = i, i+this_batch_size
      # print("i:", i, "start:", i_start, "end:", i_end, "this_batch_size:", batch_size, "len(X_train)", len(X_train))
      x_batch = X_train[i_start:i_end]
      y_batch = y_train[i_start:i_end]
      # print(x_batch.shape, y_batch.shape)

      # horizontally flip half the batch at random, and adjust steering angle accordingly
      for i in range(this_batch_size):
        x_batch[i], y_batch[i] = horizontal_flip_50_50(X_train[i], y_train[i])

      # yield this batch
      print("\nyielding on i_start to i_end", i_start, i_end, "with this_batch_size:", this_batch_size)
      # print(x_batch.shape, y_batch.shape, ": shapes, i=", i)
      yield (x_batch, y_batch)
      i = i_end
      if i >= len(X_train):
        # start again
        i = 0
        #shuffle X_train
        
      print("\nend of dataset reached..", i, "\n")
      # rem: once i reaches the end of the X_train array, an exception ought occur
      # at that point, 1 epoch would have finished, and it should start over, with a new instance of custom_data_generator
      # does fit_generator reset my generator at each epoch?
      # or dp I manually need to do this somehow ??
      # break

      # generate data for EPOCHS epochs
      # for e in range(EPOCHS):
      #   new_data = generate_batches(X_train, y_train, batch_size)
      #   yield new_data()  #???
      # print("\n end of EPOCHS reached. no more data to generate.\n")

      # generate_batches(X_train, y_train, batch_size)

  # -----------------------------

  # randomly flip half the dataset horizontally

  # built in "ImageDatagenerator" generates training data, but can't change labels
  # imageDataGen       = ImageDataGenerator(horizontal_flip=True)
  # hence must create my own generator function
  imageDataGenerator = custom_data_generator(X_train, y_train, batch_size)

  # fits the model on batches with real-time data augmentation:
  #model.fit_generator(imageDataGenerator(X_train, y_train, batch_size=batch_size),

  model.fit_generator(imageDataGenerator,
                      samples_per_epoch = X_train.shape[0],
                      nb_epoch=EPOCHS,
                      validation_data = (X_valid, y_valid),
                      callbacks=callbacks)

  # model.fit(imageDataGenerator,
  #           samples_per_epoch = X_train.shape[0],
  #           nb_epoch=EPOCHS,
  #           validation_data = (X_valid, y_valid),
  #           callbacks=callbacks)

  """
  ## from docs..
  # model_fit_generator is equivalent to:
  for e in range(epochs):
      print('Epoch', e)
      batches = 0
      for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
          model.fit(x_batch, y_batch)
          batches += 1
          if batches >= len(x_train) / 32:
              # we need to break the loop by hand because
              # the generator loops indefinitely
              break

  ## from docs..
  fit_generator(self, generator,
                      samples_per_epoch,
                      nb_epoch,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,   **
                      nb_val_samples=None,    **
                      class_weight=None,
                      max_q_size=10,
                      nb_worker=1,
                      pickle_safe=False,
                      initial_epoch=0)

                  ** validation_data: this can be either
                      - a generator for the validation data
                      - a tuple (inputs, targets)
                      - a tuple (inputs, targets, sample_weights).
                     nb_val_samples: only relevant if validation_data is a     generator. number of samples to use from validation generator at the end of every epoch.

"""


  # if no augmentation use this fit function:
  # fit paramaters on preprocessed data
  # model.fit(X_train, y_train, shuffle=True, validation_split=0.2, nb_epoch=EPOCHS, callbacks=callbacks)


  print("\nSaving model..")
  #model.save(path_to_saved_models + model_filename)
  print("Model Saved as ", path_to_saved_models + model_filename)

  # ------
  print(
  '''
    to test this model locally (in anaconda)
    at the command command line, type:
  '''
  )
  print(' python drive.py ' + path_to_saved_models + model_filename)
  print(
  '''
     examples:
     python drive.py model.h5
     python drive.py ./trained_models/model_170422_2224_sample_training_data.h5
     python ../drive.py ./model_170417_1741_sample_training_data.h5

  python drive.py model_{path_to_saved_models}model_{model_timestamp}_{SUBDIR}.h5
  '''
  )
  # (if using docker, see instructions in Udacity Lesson: "8 Running Your Network")
  # -----


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()



