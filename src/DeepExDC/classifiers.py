#分类模型构建及训练

# 1D-CNN model
import keras
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

class Classifier_CNN:

	def __init__(self, config, input_shape, nb_classes, verbose=False, build=True):
		self.config = config
		self.output_directory = config["model_dir"]
		self.acti = config["acti"]
		self.mini_batch_size = config["batch_size"]
		self.epoch = config["epoch"]
		self.learning_rate = config["learning_rate"]

		self.sim = config["sim_parm"]

		if build == True:
			self.model = self.build_model(input_shape, nb_classes)
			if (verbose == True):
				self.model.summary()
			self.verbose = verbose
			# self.model.save_weights(self.output_directory + 'model_init.hdf5')

		return
		"""
		定义CNN类
		output_directory：字符串，模型及训练日志保存目录
		input_shape：输入数据形状
		nb_class：分类类别数
		sim:使用模拟数据时的参数，默认为None,用于结果命名区分
		verbose:默认为False,不打印每个epoch的精度，为True打印精度
		build:是否调用构建模型函数 True构建模型
		"""

	def build_model(self, input_shape, nb_classes):
		"""
		构建模型函数
		input_shape：输入数据形状
		nb_class：分类类别数
		padding：填充方式，默认valid，不填充
		acti:卷积层激活函数
		"""

		padding = 'valid'
		acti=self.acti
		print("acti:",acti)
		input_layer = keras.layers.Input(input_shape)

		if input_shape[0] < 60: # for italypowerondemand dataset
			padding = 'same'

		conv1 = keras.layers.Conv1D(filters=8,kernel_size=3,padding=padding,activation=acti)(input_layer)
		conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)


		conv2 = keras.layers.Conv1D(filters=16,kernel_size=3,padding=padding,activation=acti)(conv1)
		conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)

		flatten_layer = keras.layers.Flatten()(conv2)

		output_layer = keras.layers.Dense(units=nb_classes,activation='softmax')(flatten_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
					  metrics=['accuracy'])
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
		训练模型
		输入为划分的数据集和标签
		返回值为训练后的模型
		"""
		if not tf.test.is_gpu_available:
			print('error')
			exit()

		# x_val and y_val are only used to monitor the test loss and NOT for training
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
		if self.sim is None:
			model_path = self.output_directory +'/'+ 'best_model.h5'
		else:
			model_path = self.output_directory +'/'+ self.sim + '_' + 'best_model.h5'
		model = keras.models.load_model(model_path)
		y_pred = model.predict(data)
		y_pred = np.argmax(y_pred, axis=1)
		y_true=np.argmax(label, axis=1)

		return y_pred,y_true
