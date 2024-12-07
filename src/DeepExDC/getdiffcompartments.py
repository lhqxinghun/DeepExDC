from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
try:
	from vegdist import vegdist
	from dmanova import dmanova
	from utils import *
	from process import *
except:
	try:
		from .vegdist import vegdist
		from .dmanova import dmanova
		from .utils import *
		from .process import *
	except:
		raise EOFError


def getdiff(config,shap_data=None,shap_label=None,result_path=None):
	'''
 	通过 SHAP 值进行多元方差分析，并保存结果到指定路径。

	参数:
	- config (dict): 包含各种配置的字典，需包括以下键:
		- "shap_path" (str): SHAP 数据的路径，如果提供，将从文件中加载 SHAP 数据。
		- "feature_name" (str): 特征名称，用于结果表格中。
		- "feature_df_path" (str): 包含特征信息的 CSV 文件路径。
		- "dist_method" (str): vegdist 使用的距离计算方法。
		- "result_path" (str): 结果表格保存路径。
	- shap_data (numpy.ndarray, optional): 如果存在，直接传入的 SHAP 值数组。
	- shap_label (pandas.DataFrame, optional): 包含细胞类型等标签信息的 DataFrame。
	- result_path (str, optional): 如果提供，覆盖 config 中的 result_path 值。
	返回:
	- None: 结果将直接保存到指定路径
	'''
	if result_path is not None:
		config["result_path"] = result_path
	# 激活自动转换为pandas数据框
	if shap_data is None:
		shap_data, shap_label = load_shap_arr(config,config["shap_path"])
	print("shap_data.shape",shap_data.shape)
	print(shap_label.head(5))
	if shap_data is not None:
		pass
	else:
		raise ValueError("必须存在SHAP值对象或者其文件路径")
	feature_name = config["feature_name"]
	feature_df_path = config["feature_df_path"]
	feature_df = pd.read_csv(feature_df_path)
	result_table=pd.DataFrame(np.nan, index=range(shap_data.shape[0]), columns=range(5))
	result_table.columns=[feature_name,"pvalue","effect_size","padjust","F"]
	result_table[feature_name] = feature_df[feature_name]
 
	pvalue_list = []
	effect_size_list = []
	F_list = []
	for i in tqdm(range(shap_data.shape[0])):
		x=shap_data[i,:,:].copy(order='C')
		# 检查 x 是否全零
		if np.all(x == 0):
			pvalue_list.append(1)
			effect_size_list.append(-1)
			F_list.append(1)
			continue
		distance_matrix = vegdist(x, method = config["dist_method"], check=False)
		distance_matrix= np.nan_to_num(distance_matrix, nan=0)
		rhs = pd.DataFrame({
			'category': list(shap_label["cell_type"]),
		})
		result = dmanova(lhs=distance_matrix, rhs=rhs)
		pvalue_list.append(result["aov.tab"]["Pr(>F)"][0])
		effect_size_list.append(result["effect_size"])
		F_list.append(result["aov.tab"]["F.Model"][0])

	result_table["pvalue"] = pvalue_list
	result_table["effect_size"]= effect_size_list
	result_table["F"] = F_list
	result_table["padjust"] = multipletests(result_table["pvalue"], method='fdr_bh')[1]
	result_table.to_csv(config["result_path"])
	print("csv save to:",config["result_path"])



if __name__ == '__main__':
	args = common_parse_args(desc="determination of differential compartments")
	print(args)
	config = get_config(args.config)
	getdiff(config)