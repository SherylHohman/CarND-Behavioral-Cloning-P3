# neural network for behavioural cloning driving behaviour that was captured in driving_log.csv
# reads training data from driving_log.csv
# trains a neural network from that data
# saves trained model as model.h5
# model.h5 can then be used to autonomously drive the car in the simulator that was used to produce the training data.

# Read data from CSV file
import csv

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

