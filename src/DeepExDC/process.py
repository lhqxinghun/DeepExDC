#定义对输入数据文件的处理
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
	从config配置字典取参数
	data_path:单细胞compartment score hdf5文件路径
	label_path:分类标签文件路径
	group：默认为compartment_raw，使用原始的compartment score
	scaler_type (str, optional): 选择使用的标准化方法。默认为 'standard'。
							 - 'standard': 使用 StandardScaler 进行标准化。
							 - 'minmax': 使用 MinMaxScaler 进行归一化。
	feature_range (tuple, optional): 当 scaler_type 为 'minmax' 时，指定归一化的范围。
									默认为 (-1, 1)。形状为 (min, max)。
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
	
	
	
	X=h5py.File(data_path, 'r')
	#数据转换成数据帧
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
	#排除某类样本测试
	if select_label is not None and len(select_label)!=len(set(label['cell_type'])):
		# Filter the label DataFrame to include only the selected labels
		filtered_label = label[label['cell_type'].isin(select_label)]

		# Filter the data DataFrame to include only the columns corresponding to the filtered labels
		data = data[filtered_label['cell_id']]

		# Update the label DataFrame to only include the filtered labels
		label = filtered_label.reset_index(drop=True)
	
	data=scaler.fit_transform(data.T)
	
	print("data.max:",data.max())
	print("data.min:",data.min())
	#转回数据帧
	data = pd.DataFrame(data)
	data.index = label['cell_id']
	
	return data,label


def somote_dataset(data,label,num_rows_to_append=1000):
	"""
	smote 对数据集过采样
	data:数据集数据帧
	label：数据集标签数据帧
	num_rows_to_append：默认1000，当最多类样本数少于1000时，将每一类过采样到1000

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
	划分数据集
	data:数据集数据帧
	label：数据集标签数据帧

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
	'''
	当未输入shap_path时，按config参数构造shap值路径，否则使用传递的路径
	从shap值保存路径加载shap值数组和标签
	'''
	
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
	shap_data = np.array([X[group][key][()] for key in bin_keys])

	shap_label = pd.DataFrame({
		'cell_id': [x.decode('utf-8') for x in X[group]['label']['cell_id']],
		'cell_type': [x.decode('utf-8') for x in X[group]['label']['cell_type']]
	})

	return shap_data, shap_label