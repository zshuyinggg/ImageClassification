### YOUR CODE HERE
import torch
import os, argparse
import numpy as np
from Model import MyModel

from DataLoader import load_data, train_valid_split,load_testing_images
# from Configure import model_configs, training_configs  # It is not a good design to use .py as configurations. I rewrote config files
from Configure import load_config
from ImageUtils import visualize


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default="simDLA_private_test.yaml", type=str, help="the configuration file that stores all the hyparameters")


# all the arguments are saved and passed using config_file
# parser.add_argument("mode", help="train, test or predict") 
# parser.add_argument("data_dir", help="path to the data")
# parser.add_argument("--save_dir", help="path to save the results")
args = parser.parse_args()
config = load_config(args.config_file)
if __name__ == '__main__':
	model = MyModel(config)
	if config['mode'] == 'train':
		x_train, y_train, x_test, y_test = load_data(config['data_dir'])
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

		model.train(x_train, y_train, config, x_valid, y_valid)
		model.evaluate(x_test, y_test,'Public test')

	elif config['mode'] == 'test':
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(config['data_dir'])
		model.load(config['ckpt'])
		model.evaluate(x_test, y_test,'Public test')

	elif config['mode'] == 'predict':
		# Loading private testing dataset
		x_test = load_testing_images(config['data_dir'])
		# visualizing the first testing image to check your image shape

		visualize(x_test[0], 'test.png')
		# Predicting and storing results on private testing dataset 
		predictions = model.predict_prob(x_test)
		np.save(config['result_dir'], predictions)
		

### END CODE HERE

