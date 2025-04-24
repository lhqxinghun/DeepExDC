import h5py
import pickle
import pandas as pd
import numpy as np
import smote_variants as sv
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def read_dataset(config):
	"""
	Reads and preprocesses the dataset from an HDF5 file.

	Args:
		config (dict): A dictionary containing configuration parameters.
			- data_path (str): Path to the HDF5 data file.
			- label_path (str): Path to the label file.
			- select_label (list): List of labels to select.
			- group (str): Group name in the HDF5 file.
			- scaler_type (str): Type of scaler to use ('standard', 'minmax', or None).
			- feature_range (tuple): Range for MinMaxScaler.
			- feature_name (str): Name of the dataset feature in the HDF5 file.
			- feature_df_path (str): Path to save the feature name DataFrame.

	Returns:
		tuple: A tuple containing the preprocessed data and label DataFrames.
	"""
	data_path = config["data_path"]
	label_path = config["label_path"]
	select_label = config["select_label"]
	group = config["group"]
	scaler_type = config["scaler_type"]
	feature_range = tuple(config["feature_range"])
	feature_name = config["feature_name"]
	feature_df_path = config["feature_df_path"]
	
	if scaler_type == 'standard':
		scaler = StandardScaler()
	elif scaler_type == 'minmax':
		scaler = MinMaxScaler(feature_range=feature_range)
	elif scaler_type is None:
		scaler = None
	X=h5py.File(data_path, 'r')
	cell_keys = list(filter(lambda key: 'cell_' in key, X[group].keys()))
	cell_list_len=len(cell_keys)
	datasets = [X[group][f'cell_{i}'][()] for i in range(cell_list_len)]
	data = pd.DataFrame({f'cell_{i}': dataset for i, dataset in enumerate(datasets)})
	
	if feature_name == "bin":
		bin = pd.DataFrame()
		bin['chr'] = [chrom.decode('utf-8') for chrom in X[group]['bin']['chrom']]
		bin['start'] = X[group]['bin']['start']
		bin['end'] = X[group]['bin']['end']
		feature_df = pd.DataFrame()
		feature_df["bin"] = bin["chr"].astype(str) + "_" + bin["start"].astype(str) + "_" + bin["end"].astype(str)
		feature_df.to_csv(feature_df_path)
	else:
		feature_df = pd.DataFrame()
		feature_df[feature_name] = [x.decode('utf-8') for x in X[feature_name]]
		feature_df.to_csv(feature_df_path)
 
	X.close()
	y=pickle.load(open(label_path, "rb"))
	label=pd.DataFrame()
	label['cell_id']= data.columns
	if isinstance(y, pd.DataFrame):
		label['cell_type'] = y.iloc[:,y.columns.get_loc('cell_type')] if 'cell_type' in y.columns else y.iloc[:,y.columns.get_loc('cell type')]
	elif isinstance(y, dict):
		label['cell_type'] = y.get('cell_type') or y.get('cell type')
	if select_label is not None and len(select_label)!=len(set(label['cell_type'])):
		# Filter the label DataFrame to include only the selected labels
		filtered_label = label[label['cell_type'].isin(select_label)]
		# Filter the data DataFrame to include only the columns corresponding to the filtered labels
		data = data[filtered_label['cell_id']]
		# Update the label DataFrame to only include the filtered labels
		label = filtered_label.reset_index(drop=True)
	if scaler is not None:
		data=scaler.fit_transform(data.T)
	print("data.max:",data.max())
	print("data.min:",data.min())
	data = pd.DataFrame(data)
	data.index = label['cell_id']
	
	return data,label


def somote_dataset(data,label,num_rows_to_append=1000):
	"""
	Creates a synthetic minority over-sampling technique (SMOTE) dataset.

	Args:
		data (pd.DataFrame): The input data.
		label (pd.DataFrame): The input label.
		num_rows_to_append (int): Number of synthetic rows to append. Defaults to 1000.

	Returns:
		tuple: A tuple containing the oversampled data and label DataFrames.
	"""
	df =data
	original_index=df.index
	columns = df.columns
	zero_rows = pd.DataFrame([[0] * len(columns)] * num_rows_to_append, columns=columns,
							index=['cell_{}'.format(i) for i in range(len(original_index), len(original_index) + num_rows_to_append)])
	df = pd.concat([df, zero_rows])
	additional_rows = pd.DataFrame({'cell_id': [f'cell_{i}' for i in range(len(label), len(label) + num_rows_to_append)],'cell_type': ['zero'] * num_rows_to_append})
	df_label = pd.concat([label, additional_rows], ignore_index=True)
	
	label_keys = df_label['cell_type'].value_counts().sort_index().index.tolist()
	print(label_keys)
	num_values = range(len(label_keys))
	label_num_dict = dict(zip(label_keys, num_values))
	X=(df)
	X.index=df_label['cell_type']
	y=(X.index).map(label_num_dict)
	oversampler = sv.MulticlassOversampling(oversampler='distance_SMOTE', oversampler_params={'n_jobs':2, 'random_state':42})
	# X_samp and y_samp contain the oversampled dataset
	X_samp, y_samp= oversampler.sample(X, y)
	df_smote = pd.DataFrame(data=X_samp, columns=df.columns)
	df_smote = df_smote.rename(index=lambda s: 'cell_' + str(s))
	label_num_dict_inv = {v: k for k, v in label_num_dict.items()}
	y_samp_str = []
	for num in y_samp:
		y_samp_str.append(label_num_dict_inv[num])
	df_smote_label=pd.DataFrame()
	df_smote_label['cell_id']=df_smote.index
	df_smote_label['cell_type']=y_samp_str
	df_filtered = df_smote.loc[~(df_smote == 0).all(axis=1)]
	df_label_filtered = df_smote_label[df_smote_label['cell_type'] != 'zero']
	data_smote = df_filtered
	label_smote = df_label_filtered
	print(label_smote['cell_type'].value_counts().sort_index()) 

	return data_smote, label_smote

def split_dataset(data,label):
	"""
	Splits the dataset into training and testing sets.

	Args:
		data (pd.DataFrame): The input data.
		label (pd.DataFrame): The input label.

	Returns:
		tuple: A tuple containing the training and testing data and labels.
	"""
	data_np=np.array(data)
	label_encoder = LabelEncoder()
	indexed_labels = label_encoder.fit_transform(label['cell_type'].values)
	onehot_encoder = OneHotEncoder(sparse=False)
	encoded_labels = onehot_encoder.fit_transform(indexed_labels.reshape(-1, 1))
	encoded_labels_df = pd.DataFrame(encoded_labels, columns=[f'_{i}' for i in range(encoded_labels.shape[1])])
	X_train, X_test, y_train, y_test = train_test_split(data_np, encoded_labels_df, test_size=0.2, random_state=123)
	X_train_3d = np.expand_dims(X_train, axis=2)
	X_test_3d = np.expand_dims(X_test, axis=2)
	print("X_train.shape:", X_train_3d.shape)
	print("y_train.shape:", y_train.shape)
	print()
	print("X_test.shape:", X_test_3d.shape)
	print("y_test.shape:", y_test.shape)
	dataset =(X_train_3d, y_train, X_test_3d ,y_test)

	return dataset

def load_shap_arr(config,shap_path=None):
	"""
	Loads SHAP values from an HDF5 file.

	Args:
		config (dict): A dictionary containing configuration parameters.
			- shap_dir (str): Directory path to save SHAP files.
			- sim_parm (str): Simulation parameter.
			- explain_mode (str): Explanation mode.
			- group (str): Group name in the HDF5 file.
		shap_path (str, optional): Path to the SHAP HDF5 file. Defaults to None.

	Returns:
		tuple: A tuple containing the SHAP data and label DataFrames.
	"""
	save_dir = config["shap_dir"]
	if shap_path is None:
		if config['sim_parm'] is not None:
			file_path= os.path.join(save_dir,f"{config['sim_parm']}_{config['explain_mode']}_shap.h5")
			
		else:
			file_path= os.path.join(save_dir,f"{config['group']}_{config['explain_mode']}_shap.h5")
	else:
		file_path = shap_path

	X = h5py.File(file_path, 'r')
	group = config['group']

	bin_keys = [key for key in X[group].keys() if 'bin_' in key]

	bin_nums=['bin_'+str(x+1) for x in range(len(bin_keys))]
	shap_data = np.array([X[group][key][()] for key in bin_nums])
	# shap_data = np.array([X[group][key][()] for key in bin_keys])

	shap_label = pd.DataFrame({
		'cell_id': [x.decode('utf-8') for x in X[group]['label']['cell_id']],
		'cell_type': [x.decode('utf-8') for x in X[group]['label']['cell_type']]
	})

	return shap_data, shap_label