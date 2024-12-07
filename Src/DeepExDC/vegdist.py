import warnings
import pyvegdist
import numpy as np


def vegdist(x=None,method="chord",check=True):
	"""
	Compute a distance matrix using various methods.

	This function calculates a distance matrix for the input data `x` using the specified method.
	It supports 'chord', 'manhattan', 'euclidean', and 'canberra' distance metrics.

	Args:
		x (numpy.ndarray, optional): The input data matrix. Defaults to None.
		method (str, optional): The distance metric to use. Options are 'chord', 'manhattan', 'euclidean', 'canberra'. Defaults to "chord".
		check (bool, optional): Whether to perform checks for empty rows and warn if necessary. Defaults to True.

	Returns:
		numpy.ndarray: The computed distance matrix.

	Raises:
		ValueError: If the specified method is not one of the supported methods.

	Example:
		>>> import numpy as np
		>>> x = np.array([[1, 2], [3, 4], [5, 6]])
		>>> dist_mat = vegdist(x, method="euclidean")
		>>> print(dist_mat)
		# Expected output: a distance matrix

	Note:
		The function checks for empty rows in the input data when `check` is True. If empty rows are found,
		a warning is raised because dissimilarities may be meaningless for these rows.

	See Also:
		pyvegdist.compute_distance_matrix: The underlying function used to compute the distance matrix.
	"""
	valid_methods = {
		'chord': 1,
		'manhattan': 2,
		'euclidean': 3,
		'canberra': 4
		
	}
	
	# Compute row sums
	row_sums = np.sum(x, axis=1)
	
	# Check if the method is valid
	if method not in valid_methods:
		raise ValueError("Unknown distance method,please use ['chord','manhattan','euclidean','canberra']")
	method_code = valid_methods[method]
	if check is True:
		# Check for empty rows
		if method_code in [1, 4] and np.any(row_sums == 0):
			warnings.warn(
				"You have empty rows: their dissimilarities may be meaningless in method {}".format(method)
			)
	dist_mat=pyvegdist.compute_distance_matrix(method, x)
	
	return dist_mat
