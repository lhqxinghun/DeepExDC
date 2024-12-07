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

#解决在tf2.14.0 shap0.43.0环境下 某些模型定义下，如使用全局最大池化层时 shap.deep.explainers报错 gradient registry has no entry for: shap_AddV2问题
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough #添加后可以计算，但速度慢


class RunSHAP():
	def __init__(self, config):
	 
		"""
		初始化参数
		model_path: 字符串，训练好的分类模型路径
		dataset:列表，用来构建SHAP解释器和解释的样本和标签 为了方便在背景数据和前景数据上评估模型，及自定义选择数据，这里的标签为原始标签，未转为onehot编码
		shap_dir:字符串，存储SHAP值hdf5文件的目录
		sim:字符串，默认为None，仅在模拟compartment数据时使用，如group_run1，用于区分模拟数据集模型命名
		group：字符串，默认为compartment_raw,即原始的compartment score,因为采用Higashi方法计算时同时存储了zscore标准化和分位数变换的数据
		explain_mode:布尔值，默认为test,用于选择解释整个数据集或是解释测试集
		save_dir:字符串，存放shap值文件的目录
		save_shap_file：布尔值，默认为False,是否保存shap值文件，默认不保存
		"""
		# model_path,dataset,shap_dir,group='compartment_raw',mode='test'
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
		加载要送入SHAP解释的数据和标签
		data:数据集数据帧
		label:标签数据帧
		"""
		data,label = read_dataset(self.config)
		data_np=np.array(data)
		if self.config["explain_mode"] == "all":
			X = np.expand_dims(data_np, axis=2)
			dataset =(X,label)
		else:
			X_train, X_test, y_train, y_test = train_test_split(data_np, label, test_size=0.2, random_state=123)


			X_train_3d = np.expand_dims(X_train, axis=2)
			#测试集的顺序是打乱的，为了方便后续处理，重新排序后再计算SHAP值
			print("X_test.shape:", X_test.shape)
			print("y_test.shape:", y_test.shape)
			y_test.reset_index(inplace=True,drop=True)
			# 根据类型标签对数据帧进行排序
			sorted_label = y_test.sort_values(by='cell_type')
			# 获取排序后的索引
			sorted_indices = sorted_label.index

			# 根据排序后的索引重新排列数据和标签
			X_test = X_test[sorted_indices]
			y_test = sorted_label.reset_index(drop=True)

			X_test_3d = np.expand_dims(X_test, axis=2)
			dataset =(X_train_3d, y_train, X_test_3d ,y_test)

		return dataset

  
	def save_shap_values(self,shap_arr,y):
		"""
		功能：保存SHAP值结果和解释的样本的标签为hdf5文件，按bin从1开始标号，把每个Bin的shap值单独保存到文件中
		self:继承父类的参数 self下的对象可直接在子函数调用
		shap_arr: numpy多维数组，去掉多余维度的SHAP值数组，形状为 样本数，分类数，特征数
		y: pands数据帧：为过滤后(仅保留预测正确的样本)的要解释的样本的cell_id和cell_type，
		返回值：字符串，保存的结果文件路径
		"""

		# if self.sim_parm is not None:
		# 	file_path= os.path.join(self.save_dir,f"{self.sim_parm}_{self.explain_mode}_shap.h5")
			
		# else:
		# 	file_path= os.path.join(self.save_dir,f"{self.group}_{self.explain_mode}_shap.h5")

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
		评估模型，生成分类报告
		X：要预测的样本
		y：样本标签
		返回值：numpy数组，分类正确的样本位置为1，否则为0，用于过滤样本
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
		# 生成新的标签
		new_labels = np.where(y_true_labels == y_pred_labels, 1, 0)

		return new_labels
	
	def cal_shap_arr(self,background,ex_data):
		"""
		计算SHAP值，返回SHAP值数组
		background:numpy数组，背景数据
		ex_data：numpy数组,要解释的样本
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
		data:用于选择背景数据的数据集
		y:包含分类正确与否的样本标签
		lable:用作背景数据的细胞类型列表
		返回值为过滤后的数据集

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
		data:用于选择要解释的样本的数据集
		y:包含分类正确与否的样本标签
		lable:要解释的样本的细胞类型（分类标签）
		返回值为过滤后的数据集，及过滤后的标签
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
		解释测试集样本，每解释一类，背景数据从训练集其他类选择
		返回值：SHAP值数组，过滤后的测试集标签
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
		用于解释整个数据集的实现
		"""
		X, y = self.load_data_for_shap()
		label = y['cell_type']
		y['filter'] = self.model_eval(X,label)
		# 使用布尔索引来筛选数据集
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
		调用来计算和保存SHAP值的方法
		返回值：字符串，SHAP值保存结果文件路径
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