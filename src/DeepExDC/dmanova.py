import numpy as np
from scipy.stats import chi2
from scipy.linalg import pinv, cholesky, LinAlgError
import numpy as np
import pandas as pd
import patsy
'''
calculateK
dmanova
参考GUniFrac包源码，使用python转写

'''
def calculateK(G, H, df2):
	"""
	计算 K 值和中间变量 mu1, mu2.

	参数:
	- G (numpy.ndarray): 核矩阵.
	- H (numpy.ndarray): 投影矩阵.
	- df2 (int): 自由度2.

	返回:
	- dict: 包含 K, mu1, mu2 的字典.
	"""
	n = G.shape[0]
	dG = np.diag(G)
	
	mu1 = np.sum(dG) / (n - df2)
	mu2 = (np.sum(G**2) - np.sum(dG**2)) / ((n - df2)**2 + np.sum(H**4) - 2 * np.sum((np.diag(H)**2)))
	K = mu1**2 / mu2
	return {"K": K, "mu1": mu1, "mu2": mu2}

def dmanova(lhs, rhs, positify=False,
			returnG=False,
			debug=False):
	"""
	执行距离矩阵的多元方差分析 (MANOVA).

	参数:
	- lhs (numpy.ndarray): 左侧矩阵 (距离矩阵).
	- rhs (pandas.DataFrame): 右侧模型矩阵 (为因变量).
	- positify (bool, optional): 是否强制矩阵为正定矩阵, 默认为 False.
	- returnG (bool, optional): 是否返回 G 矩阵, 默认为 False.
	- debug (bool, optional): 是否启用调试模式, 默认为 False.

	返回:
	- dict: 包含统计表、K 值、效应量大小 (effect size)，以及 (如果指定) G 矩阵的字典.
	"""
	
	# Matrix operations to construct the relevant matrices
	n = rhs.shape[0]
	
	if isinstance(lhs, np.ndarray):
		D = lhs
	else:
		D = np.asarray(lhs)
	
	D = -D**2 / 2
	
	G = np.mean(D) + D - np.mean(D, axis=1).reshape(-1, 1) - np.ones((n, 1)) @ np.mean(D, axis=1).reshape(1, -1)
	
	if positify:
		try:
			# Use Cholesky decomposition to ensure positive definiteness
			G = cholesky(G, lower=True)
		except LinAlgError:
			# If not positive definite, use pseudo-inverse as fallback
			G = pinv(G)
	if debug is True:
		print("Graw")
		print(G)
	# 创建模型矩阵
	# 使用 patsy.dmatrix 来构造虚拟变量矩阵
	dmatrix = patsy.dmatrix('C(category, Sum) - 1', rhs, return_type='dataframe')

	# 获取分组信息
	grps = dmatrix.columns
	nterms = len(grps)  # 虚拟变量列的数量

	# Z: 去掉最后一列的虚拟变量矩阵 (drop the reference group)
	Z = dmatrix.iloc[:, 0:1].to_numpy()

	# XZ: 包含所有虚拟变量列的矩阵
	XZ = dmatrix.to_numpy()

	# 计算 XZ 和 Z 的逆矩阵
	XZi = np.linalg.inv(np.dot(XZ.T,XZ))
	Zi = np.linalg.inv(np.dot(Z.T,Z))

	# 计算投影矩阵 HZ 和 HXZ
	HZ = Z @ Zi @ Z.T
	HXZ = XZ @ XZi @ XZ.T

	# 计算最终的 HX 矩阵
	HX = HXZ - HZ
 
	
	HIXZ = np.eye(n) - HXZ
	HIX = np.eye(n) - HX

	# 计算 df1 和 df2
	df1 = XZ.shape[1] - Z.shape[1]
	df2 = n - XZ.shape[1]

	# 计算 MSS, RSS 和 TSS
	MSS = np.sum(G * HX)
	RSS = np.sum(G * HIXZ)
	TSS = np.sum(np.diag(G))

	# 计算 F 统计量
	f_stat = (MSS / df1) / (RSS / df2)
	

	# 计算 GXZ, XZXZi 和 GXZtXZXZi
	GXZ = G @ XZ
	XZXZi = XZ @ XZi
	GXZtXZXZi = GXZ @ XZXZi.T

	# 计算 G.tilde
	G_tilde = G + XZXZi @ (GXZ.T @ XZ) @ XZXZi.T - GXZtXZXZi - GXZtXZXZi.T
	



	obj = calculateK(G_tilde, HIXZ, n - df2)
	K = obj['K']
	if debug is True:
		print("K")
		print(K)
		print("f_stat")
		print(f_stat)
		print("df1")
		print(df1)
	
	p_value2 = chi2.sf(f_stat*K*df1, K*df1)
	
	# Sum of squares table
	sums_of_sqs = np.array([MSS, RSS, TSS])
	
	tab = {
		"Df": [df1, df2, n - 1],
		"SumsOfSqs": sums_of_sqs,
		"MeanSqs": [MSS/df1, RSS/df2, np.nan],
		"F.Model": [f_stat, np.nan, np.nan],
		"R2": [MSS/TSS, np.nan, np.nan],
		"Pr(>F)": [p_value2, np.nan, np.nan]
	}
 

	N = n
	MS_res = RSS/df2
	df = np.array(tab["Df"])
	mean_sqs = np.array(tab["MeanSqs"])

	omega = df*(mean_sqs-MS_res)/(df*mean_sqs+(N-df)*MS_res)



	
	effect_size=omega[0]

	if returnG:
		return {"aov.tab": tab, "df": K, "G": G,"effect_size":effect_size}
	else:
		return {"aov.tab": tab, "df": K,"effect_size":effect_size}


	
def format_aov_table(df, significance_levels=True):
	"""
	将输入的统计数据转换为类似于 R aov 输出的格式。

	参数:
	- df (dict): 包含 'Df', 'SumsOfSqs', 'MeanSqs', 'F.Model', 'R2', 'Pr(>F)' 的字典。
	- significance_levels (bool): 是否在 p 值列中添加显著性标记。

	返回:
	- str: 格式化后的统计数据表格。
	"""
	# 创建 DataFrame
	df = pd.DataFrame(df, index=['Model(Adjusted)', 'Residuals', 'Total'])

	# 自定义格式化函数
	def format_value(value, precision=4):
		if np.isnan(value):
			return 'NA'
		return f'{value:.{precision}f}'

	def format_significance(p_value):
		if p_value <= 0.001:
			return '***'
		elif p_value <= 0.01:
			return '**'
		elif p_value <= 0.05:
			return '*'
		elif p_value <= 0.1:
			return '.'
		else:
			return ' '

	# 将数据格式化为字符串
	df_formatted = df.copy()
	for col in df.columns:
		if col == 'Pr(>F)' and significance_levels:
			df_formatted[col] = df[col].apply(lambda x: f'{format_value(x, 4)} {format_significance(x)}')
		else:
			df_formatted[col] = df[col].apply(format_value)

	# 创建结果字符串
	result = "F stat and P value of the last term is adjusted by preceding terms!\n\n"
	result += df_formatted.to_string()
	result += "\n\n---\n"
	if significance_levels:
		result += "Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1"

	return result