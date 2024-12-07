from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np

# Attempt to import local modules, if not found, try relative imports
try:
	from vegdist import vegdist
	from dmanova import dmanova
	from utils import *
	from process import *
except ImportError:
	try:
		from .vegdist import vegdist
		from .dmanova import dmanova
		from .utils import *
		from .process import *
	except ImportError:
		raise EOFError("Failed to import required modules.")

def getdiff(config, shap_data=None, shap_label=None, result_path=None):
	"""
	Calculate differential statistics for SHAP values.

	This function computes the p-value, effect size, and F-statistic for each feature
	based on the SHAP values and saves the results to a CSV file.

	Args:
		config (dict): Configuration dictionary containing parameters.
		shap_data (numpy.ndarray, optional): SHAP values data. Defaults to None.
		shap_label (pandas.DataFrame, optional): Labels for SHAP values. Defaults to None.
		result_path (str, optional): Path to save the result CSV file. Defaults to None.

	Returns:
		None
	"""
	if result_path is not None:
		config["result_path"] = result_path
	if shap_data is None:
		shap_data, shap_label = load_shap_arr(config, config["shap_path"])
	print("shap_data.shape", shap_data.shape)
	print(shap_label.head(5))
	if shap_data is not None:
		pass
	else:
		raise ValueError("SHAP values object or file path must exist.")
	feature_name = config["feature_name"]
	feature_df_path = config["feature_df_path"]
	feature_df = pd.read_csv(feature_df_path)
	result_table = pd.DataFrame(np.nan, index=range(shap_data.shape[0]), columns=range(5))
	result_table.columns = [feature_name, "pvalue", "effect_size", "padjust", "F"]
	result_table[feature_name] = feature_df[feature_name]

	pvalue_list = []
	effect_size_list = []
	F_list = []
	for i in tqdm(range(shap_data.shape[0])):
		x = shap_data[i, :, :].copy(order='C')
		# Check if x is all zeros
		if np.all(x == 0):
			pvalue_list.append(1)
			effect_size_list.append(-1)
			F_list.append(1)
			continue
		distance_matrix = vegdist(x, method=config["dist_method"], check=False)
		distance_matrix = np.nan_to_num(distance_matrix, nan=0)
		rhs = pd.DataFrame({
			'category': list(shap_label["cell_type"]),
		})
		result = dmanova(lhs=distance_matrix, rhs=rhs)
		pvalue_list.append(result["aov.tab"]["Pr(>F)"][0])
		effect_size_list.append(result["effect_size"])
		F_list.append(result["aov.tab"]["F.Model"][0])

	result_table["pvalue"] = pvalue_list
	result_table["effect_size"] = effect_size_list
	result_table["F"] = F_list
	result_table["padjust"] = multipletests(result_table["pvalue"], method='fdr_bh')[1]
	result_table.to_csv(config["result_path"])
	print("CSV saved to:", config["result_path"])

if __name__ == '__main__':
	args = common_parse_args(desc="determination of differential compartments")
	print(args)
	config = get_config(args.config)
	getdiff(config)