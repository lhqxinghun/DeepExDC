# The following code is a derivative of an open-source project licensed under the GNU General Public License v3.
# Original Project: [https://github.com/hfawaz/dl-4-tsc]
# License: GNU General Public License v3
#
# Copyright (c) [2019] [Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
import keras
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam, RMSprop, SGD, Nadam
import tensorflow as tf
import numpy as np
import os
try:
	from .utils import save_logs
except:
	from utils import save_logs
print('Using Tensorflow version: {}, '
		  'and Keras version: {}.'.format(tf.__version__,
									  keras.__version__))

    # 'optimizer': ['adam', 'rmsprop', 'sgd', 'nadam'],
    # 'learning_rate': [0.0001, 0.001, 0.01],
    # 'base_filters': [8, 16, 32],
    # 'num_layers': [1, 2, 3]

class Classifier_CNN:
	"""
	A CNN classifier class built with Keras and TensorFlow.

	This class represents a Convolutional Neural Network (CNN) classifier.
	It provides methods for building the model, training, and prediction.
	"""
	def __init__(self, config, input_shape, nb_classes, verbose=False, build=True):
		"""
		Initialize the Classifier_CNN instance.

		Args:
			config (dict): A dictionary containing the model configuration.
			input_shape (tuple): The shape of the input data.
			nb_classes (int): The number of classes in the classification problem.
			verbose (bool): Whether to print detailed information during training. Defaults to False.
			build (bool): Whether to build the model immediately. Defaults to True.
			optimizer (str): You can select the optimizer, which defaults to Adam.
			base_filters (int): The number of convolutional layer base filters, default is 8.
			num_layers (int): The number of convolution blocks in the model, which is 2 by default.
		"""
		self.config = config
		self.output_directory = config["model_dir"]
		self.acti = config["acti"]
		self.mini_batch_size = config["batch_size"]
		self.epoch = config["epoch"]
		self.learning_rate = config["learning_rate"]
		self.optimizer = config["optimizer"]
		self.base_filters = config["base_filters"]
		self.num_layers = config["num_layers"]

		self.sim = config["sim_parm"]

		if build == True:
			self.model = self.build_model(input_shape, nb_classes)
			if (verbose == True):
				self.model.summary()
			self.verbose = verbose
		return

	def build_model(self, input_shape, nb_classes):
		"""
		Build the CNN model.

		Args:
			input_shape (tuple): The shape of the input data.
			nb_classes (int): The number of classes in the classification problem.

		Returns:
			keras.Model: The built Keras model.
		"""
		padding = 'valid'
		acti=self.acti
		print("acti:",acti)
		if self.optimizer == 'adam':
			opt = Adam(learning_rate=self.learning_rate)
		elif self.optimizer == 'rmsprop':
			opt = RMSprop(learning_rate=self.learning_rate)
		elif self.optimizer == 'sgd':
			opt = SGD(learning_rate=self.learning_rate)
		elif self.optimizer == 'nadam':
			opt = Nadam(learning_rate=self.learning_rate)
		else:
			opt = Adam(learning_rate=self.learning_rate)

		input_layer = Input(input_shape)
		if input_shape[0] < 60: # for italypowerondemand dataset
			padding = 'same'
		x = input_layer
    
		for i in range(self.num_layers):
			x = Conv1D(filters=self.base_filters*(2**i),  # 每层filter数量指数增长
					kernel_size=3,
					padding=padding,
					activation=self.acti)(x)
			x = MaxPooling1D(pool_size=2)(x)
		
		x = Flatten()(x)
		output_layer = Dense(nb_classes, activation='softmax')(x)
		
		model = Model(inputs=input_layer, outputs=output_layer)
		model.compile(loss='categorical_crossentropy',
					optimizer=opt,
					metrics=['accuracy'])
		
		
		# conv1 = keras.layers.Conv1D(filters=8,kernel_size=3,padding=padding,activation=acti)(input_layer)
		# conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)


		# conv2 = keras.layers.Conv1D(filters=16,kernel_size=3,padding=padding,activation=acti)(conv1)
		# conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)

		# flatten_layer = keras.layers.Flatten()(conv2)

		# output_layer = keras.layers.Dense(units=nb_classes,activation='softmax')(flatten_layer)

		# model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		# model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
		# 			  metrics=['accuracy'])
		
		if self.sim is None:

			file_path = self.output_directory + '/'+'best_model.h5'
		else:
			file_path = self.output_directory + '/'+self.sim+'_'+'best_model.h5'
		early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',  
								mode='min',          
								patience=10,          
								verbose=2)

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
														   save_best_only=True)

		self.callbacks = [early_stopping,model_checkpoint]

		return model

	def fit(self, x_train, y_train, x_val, y_val):
		"""
		Train the model.

		Args:
			x_train (numpy.ndarray): The training data.
			y_train (numpy.ndarray): The training labels.
			x_val (numpy.ndarray): The validation data.
			y_val (numpy.ndarray): The validation labels.

		Returns:
			str: The path to the saved best model.
		"""
		if not tf.test.is_gpu_available:
			print('error')
			exit()
		mini_batch_size = self.mini_batch_size
		nb_epochs = self.epoch
		print(self.output_directory)

		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
							  verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)


		if self.sim is None:
			self.model.save(self.output_directory+'/'+'last_model.h5')
			model_path = self.output_directory + '/'+'best_model.h5'
			model = keras.models.load_model(self.output_directory + '/'+'best_model.h5')
			save_logs(self.output_directory, hist)
		else:
			model_path = self.output_directory + '/'+self.sim + '_' + 'best_model.h5'
			model = keras.models.load_model(self.output_directory + '/'+self.sim + '_' + 'best_model.h5')
			logs_dir=f"{self.output_directory}/{self.sim}_logs"
			if os.path.exists(logs_dir):
				print(f"Directory already exists: {logs_dir}")
			else:
				os.makedirs(logs_dir)
			save_logs(logs_dir, hist)

		_, accuracy = model.evaluate(x_val, y_val)
		print('Accuracy: %.2f%%' % (accuracy * 100))

		keras.backend.clear_session()

		return model_path

	def predict(self, data,label):
		"""
		Make predictions using the trained model.

		Args:
			data (numpy.ndarray): The input data.
			label (numpy.ndarray): The true labels.

		Returns:
			tuple: The predicted labels and the true labels.
		"""
		if self.sim is None:
			model_path = self.output_directory +'/'+ 'best_model.h5'
		else:
			model_path = self.output_directory +'/'+ self.sim + '_' + 'best_model.h5'
		model = keras.models.load_model(model_path)
		y_pred = model.predict(data)
		y_pred = np.argmax(y_pred, axis=1)
		y_true=np.argmax(label, axis=1)

		return y_pred,y_true

