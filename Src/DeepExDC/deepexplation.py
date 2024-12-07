import argparse
import shap
import os
from tqdm import tqdm
import keras


try:
	from utils import *
	from process import *
except:
	try:
		from .utils import *
		from .process import *
	except:
		raise EOFError

shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough


class RunSHAP():
	"""
	A class to run SHAP (SHapley Additive exPlanations) analysis on a Keras model.

	This class encapsulates the functionality to load a Keras model, prepare data for SHAP analysis,
	calculate SHAP values, and save or return the results.

	Attributes:
		config (dict): A dictionary containing configuration parameters.
		model (keras.Model): The loaded Keras model.
		sim_parm (dict): Parameters for simulation.
		group (str): The group name for saving SHAP values.
		explain_mode (str): The mode of explanation, either 'all' or 'test'.
		save_dir (str): The directory to save SHAP results.
		save_shap_file (bool): A flag to indicate whether to save SHAP values to a file.

	Methods:
		load_data_for_shap(): Load and prepare data for SHAP analysis.
		save_shap_values(shap_arr, y): Save SHAP values to an HDF5 file.
		model_eval(X, y): Evaluate the model on a given dataset.
		cal_shap_arr(background, ex_data): Calculate SHAP values for a given dataset.
		select_background(data, y, label): Select a background dataset for SHAP analysis.
		select_data(data, y, label): Select a subset of data for SHAP analysis.
		explain_test(): Perform SHAP analysis on the test dataset.
		explain_all(): Perform SHAP analysis on the entire dataset.
		run(): Run the SHAP analysis and return the results.
	"""
	def __init__(self, config):
		"""
		Initialize a new instance of RunSHAP.

		Args:
			config (dict): A dictionary containing configuration parameters.
		"""
		model_path = config["model_path"]
		self.config = config
		self.model = keras.models.load_model(model_path)
		self.sim_parm=config["sim_parm"]
		self.group=config["group"]
		self.explain_mode = config["explain_mode"]
		self.save_dir = config["shap_dir"]
		self.save_shap_file = config["save_shap_file"]
	
	def load_data_for_shap(self):
		"""
		Load and prepare data for SHAP analysis.

		Returns:
			dataset (tuple): A tuple containing the prepared dataset.
		"""
		data,label = read_dataset(self.config)
		data_np=np.array(data)
		if self.config["explain_mode"] == "all":
			X = np.expand_dims(data_np, axis=2)
			dataset =(X,label)
		else:
			X_train, X_test, y_train, y_test = train_test_split(data_np, label, test_size=0.2, random_state=123)


			X_train_3d = np.expand_dims(X_train, axis=2)
			print("X_test.shape:", X_test.shape)
			print("y_test.shape:", y_test.shape)
			y_test.reset_index(inplace=True,drop=True)
			sorted_label = y_test.sort_values(by='cell_type')
			sorted_indices = sorted_label.index
			X_test = X_test[sorted_indices]
			y_test = sorted_label.reset_index(drop=True)
			X_test_3d = np.expand_dims(X_test, axis=2)
			dataset =(X_train_3d, y_train, X_test_3d ,y_test)

		return dataset

  
	def save_shap_values(self,shap_arr,y):
		"""
		Save SHAP values to an HDF5 file.

		Args:
			shap_arr (numpy.ndarray): The SHAP values to save.
			y (pandas.DataFrame): The corresponding labels.

		Returns:
			file_path (str): The path to the saved file.
		"""
		file_path=self.config["shap_path"]
		with h5py.File(file_path, 'w') as hf:
			grp = hf.create_group(self.group)
			for i in range(shap_arr.shape[2]):
				bin_data=shap_arr[:,:,i] 
				grp.create_dataset(f'bin_{i + 1}', data=bin_data)
			label_shap=grp.create_group('label')
			label_shap.create_dataset('cell_type', data=[l.encode('utf8') for l in y['cell_type']],
								dtype=h5py.special_dtype(vlen=str))
			label_shap.create_dataset('cell_id', data=[l.encode('utf8') for l in y['cell_id']],
								dtype=h5py.special_dtype(vlen=str))
		hf.close()
		print(f"SHAP_value save to:{file_path}")
		return file_path
	
	def model_eval(self,X,y):
		"""
		Evaluate the model on a given dataset.

		Args:
			X (numpy.ndarray): The input data.
			y (pandas.DataFrame): The corresponding labels.

		Returns:
			new_labels (numpy.ndarray): The predicted labels.
		"""
		label_encoder = LabelEncoder()
		indexed_labels = label_encoder.fit_transform(y)
		onehot_encoder = OneHotEncoder(sparse=False)
		encoded_labels = onehot_encoder.fit_transform(indexed_labels.reshape(-1, 1))
		y_test =np.expand_dims(encoded_labels,axis=2)
		print("------model eval------")
		predictions = self.model.predict(X, batch_size=16)
		report = classification_report(y_test.argmax(axis=1),
		predictions.argmax(axis=1), target_names=label_encoder.classes_)
		print(str(report))
		
		y_true_labels = label_encoder.inverse_transform(y_test.argmax(axis=1))
		y_pred_labels = label_encoder.inverse_transform(predictions.argmax(axis=1))
		new_labels = np.where(y_true_labels == y_pred_labels, 1, 0)

		return new_labels
	
	def cal_shap_arr(self,background,ex_data):
		"""
		Calculate SHAP values for a given dataset.

		Args:
			background (numpy.ndarray): The background dataset.
			ex_data (numpy.ndarray): The dataset to explain.

		Returns:
			shap_arr (numpy.ndarray): The calculated SHAP values.
		"""

		e = shap.DeepExplainer(self.model, background)
		print(f"e.expected_value:{e.expected_value}")
		out_list = []
		num_samples = np.shape(ex_data)[0]
		for sample in tqdm(range(num_samples)):
			# shap
			shap_values = e.shap_values(ex_data[sample : sample + 1])
			out_list.append(shap_values)
		shap_arr = np.squeeze(np.array(out_list))
		return shap_arr

	def select_background(self,data,y,label):
		"""
		Select a background dataset for SHAP analysis.

		Args:
			data (numpy.ndarray): The input data.
			y (pandas.DataFrame): The corresponding labels.
			label (list): The labels to include in the background dataset.

		Returns:
			X (numpy.ndarray): The selected background dataset.
		"""

		y = y.reset_index(drop=True)
		X =[]
		for i in range(0,data.shape[0]):
			if y['cell_type'][i] in label and y['filter'][i] ==1:
				X.append(data[i])
		X=np.stack(X)
		return X

	def select_data(self,data,y,label):
		"""
		Select a subset of data for SHAP analysis.

		Args:
			data (numpy.ndarray): The input data.
			y (pandas.DataFrame): The corresponding labels.
			label (str): The label to include in the subset.

		Returns:
			X (numpy.ndarray): The selected subset of data.
			Y (pandas.DataFrame): The corresponding labels.
		"""

		y = y.reset_index(drop=True)
		X =[]
		Y = y[(y['cell_type'] ==label)& (y['filter']==1)]
		for i in range(0,data.shape[0]):
			if y['cell_type'][i] in label and y['filter'][i] ==1:
				X.append(data[i])
		X=np.stack(X)
		return X,Y


	def explain_test(self):
		"""
		Perform SHAP analysis on the test dataset.

		Returns:
			shap_arr (numpy.ndarray): The calculated SHAP values.
			y (pandas.DataFrame): The corresponding labels.
		"""

		X_train_3d, y_train, X_test_3d ,y_test = self.load_data_for_shap()
		y=y_test
		#添加新标签，用于过滤解释的样本中分类正确的
		label = y['cell_type']
		label_train = y_train['cell_type']

		y_train['filter'] = self.model_eval(X_train_3d,label_train)
		y['filter'] = self.model_eval(X_test_3d,label)
		if self.config["random_background"] is True:
			if X_train_3d.shape[0]<100:
				background=X_train_3d[np.random.choice(X_train_3d.shape[0], 100, replace=True)]
			else:
				
				background=X_train_3d[np.random.choice(X_train_3d.shape[0], 100, replace=False)]
	
			shap_arr = self.cal_shap_arr(background,X_test_3d)
		else:
			select_list = list(np.unique(y_train['cell_type'].values))
			out_list = []

			y_list = []

			for i, item in enumerate(select_list):
				X_test_item,y_item= self.select_data(X_test_3d,y_test,item)
				y_list.append(y_item)
				X_train_item = self.select_background(X_train_3d,y_train,select_list[:i] + select_list[i+1:])

				if X_train_item.shape[0]<100:
					background=X_train_item[np.random.choice(X_train_item.shape[0], 100, replace=True)]
				else:
				
					background=X_train_item[np.random.choice(X_train_item.shape[0], 100, replace=False)]
				e = shap.DeepExplainer(self.model, background)
				print(f"e.expected_value:{e.expected_value}")
				sample_list = []
				num_samples = np.shape(X_test_item)[0]
				for sample in tqdm(range(num_samples)):
					shap_values = e.shap_values(X_test_item[sample : sample + 1])
					sample_list.append(shap_values)
				print(np.array(sample_list).shape)
				out_list.extend(sample_list)
			print(np.array(out_list).shape)
			shap_arr = np.squeeze(np.array(out_list))
			y=pd.concat(y_list)
		return shap_arr,y

	def explain_all(self):
		"""
		Perform SHAP analysis on the entire dataset.

		Returns:
			shap_arr (numpy.ndarray): The calculated SHAP values.
			y (pandas.DataFrame): The corresponding labels.
		"""

		X, y = self.load_data_for_shap()
		label = y['cell_type']
		y['filter'] = self.model_eval(X,label)
		filtered_indices = y['filter'] == 1
		X = X[filtered_indices]
		y = y.loc[filtered_indices]
		
		if self.config["random_background"] is True:
			if X.shape[0]<100:
				background=X[np.random.choice(X.shape[0], 100, replace=True)]
			else:
				
				background=X[np.random.choice(X.shape[0], 100, replace=False)]
	
			shap_arr = self.cal_shap_arr(background,X)
		else:
			select_list = list(np.unique(y['cell_type'].values))
			out_list = []

			y_list = []

			for i, item in enumerate(select_list):
				X_train_item = self.select_background(X,y,select_list[:i] + select_list[i+1:])

				background=X_train_item
				X_test_item,y_item= self.select_data(X,y,item)
				y_list.append(y_item)
				e = shap.DeepExplainer(self.model, background)
				print(f"e.expected_value:{e.expected_value}")
				sample_list = []
				num_samples = np.shape(X_test_item)[0]
				for sample in tqdm(range(num_samples)):
					# shap
					shap_values = e.shap_values(X_test_item[sample : sample + 1])
					sample_list.append(shap_values)
				print(np.array(sample_list).shape)
				out_list.extend(sample_list)
			print(np.array(out_list).shape)
			shap_arr = np.squeeze(np.array(out_list))
			y=pd.concat(y_list)
		
		return shap_arr,y

	def run(self):
		"""
		Run the SHAP analysis and return the results.

		Returns:
			shap_obj (object): The SHAP results, either a file path or a tuple containing the SHAP values and labels.
		"""

		if self.config["explain_mode"] == "all":
			shap_arr,y = self.explain_all()
		else:
			shap_arr,y = self.explain_test()
		if self.save_shap_file is True:
			shap_path = self.save_shap_values(shap_arr,y)
			shap_obj=shap_path
		else:
			shap_arr = shap_arr.transpose(2,0,1)
			shap_obj = [shap_arr,y]
		return shap_obj

if __name__ == '__main__':
	args = common_parse_args(desc="shapley explanation")
	print(args)
	config = get_config(args.config)
	
	runshap = RunSHAP(config)
	runshap.run()