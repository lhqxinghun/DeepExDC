# The following code is a Python port of functions from the GUniFrac R package.
# Original package: GUniFrac
# URL: https://CRAN.R-project.org/package=GUniFrac 
# License: GPL-3

# Copyright (c) 2024 lhqxinghun

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
from scipy.stats import chi2
from scipy.linalg import pinv, cholesky, LinAlgError
import numpy as np
import pandas as pd
import patsy

def calculateK(G, H, df2):
	"""
	Calculate K value and intermediate variables mu1, mu2.

	Args:
	- G (numpy.ndarray): Kernel matrix.
	- H (numpy.ndarray): Projection matrix.
	- df2 (int): Degrees of freedom 2.

	Returns:
	- dict: Dictionary containing K, mu1, and mu2.
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
	Perform a multivariate analysis of variance (MANOVA).

	Args:
	- lhs: Left-hand side of the model (matrix or array-like).
	- rhs: Right-hand side of the model (data frame or array-like).
	- positify (bool): Whether to ensure the matrix is positive definite. Defaults to False.
	- returnG (bool): Whether to return the kernel matrix G. Defaults to False.
	- debug (bool): Whether to print debug information. Defaults to False.

	Returns:
	- dict: Dictionary containing the ANOVA table, degrees of freedom, and effect size.
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
	dmatrix = patsy.dmatrix('C(category, Sum) - 1', rhs, return_type='dataframe')
	grps = dmatrix.columns
	nterms = len(grps)
	Z = dmatrix.iloc[:, 0:1].to_numpy()
	XZ = dmatrix.to_numpy()
	XZi = np.linalg.inv(np.dot(XZ.T,XZ))
	Zi = np.linalg.inv(np.dot(Z.T,Z))
	HZ = Z @ Zi @ Z.T
	HXZ = XZ @ XZi @ XZ.T
	HX = HXZ - HZ
	HIXZ = np.eye(n) - HXZ
	HIX = np.eye(n) - HX
	df1 = XZ.shape[1] - Z.shape[1]
	df2 = n - XZ.shape[1]
	MSS = np.sum(G * HX)
	RSS = np.sum(G * HIXZ)
	TSS = np.sum(np.diag(G))
	f_stat = (MSS / df1) / (RSS / df2)
	GXZ = G @ XZ
	XZXZi = XZ @ XZi
	GXZtXZXZi = GXZ @ XZXZi.T
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
	Format the ANOVA table for presentation.

	Args:
	- df (dict): Dictionary containing the ANOVA table data.
	- significance_levels (bool): Whether to include significance levels. Defaults to True.

	Returns:
	- str: Formatted ANOVA table as a string.
	"""
	df = pd.DataFrame(df, index=['Model(Adjusted)', 'Residuals', 'Total'])
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
	df_formatted = df.copy()
	for col in df.columns:
		if col == 'Pr(>F)' and significance_levels:
			df_formatted[col] = df[col].apply(lambda x: f'{format_value(x, 4)} {format_significance(x)}')
		else:
			df_formatted[col] = df[col].apply(format_value)
	result = "F stat and P value of the last term is adjusted by preceding terms!\n\n"
	result += df_formatted.to_string()
	result += "\n\n---\n"
	if significance_levels:
		result += "Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1"

	return result