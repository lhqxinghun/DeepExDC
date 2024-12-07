import warnings
import pyvegdist
import numpy as np


def vegdist(x=None,method="chord",check=True):
    '''
    输入数据矩阵：(样本，特征)
    距离计算方法：str
    返回值：距离矩阵
    '''
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


