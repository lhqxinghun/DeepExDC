# The following code is a derivative work of two separate open-source projects,
# originally licensed under the MIT License.

# Project 1:
# [dchic]
# [https://github.com/ay-lab/dcHiC]

# Project 2:
# [Higashi]
# [https://github.com/ma-compbio/Higashi]

# This derivative work is licensed under the same terms.
# Copyright (c) [Year] [Your Name]

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
import argparse
import pickle
import subprocess
import warnings
import numpy as np
from sklearn.preprocessing import quantile_transform
from sklearn.decomposition import PCA
import h5py
import os
from tqdm import tqdm, trange
from scipy.stats import spearmanr, pearsonr, zscore
import pandas as pd
try:
	from scipy.stats import PearsonRConstantInputWarning, SpearmanRConstantInputWarning

except:
	from scipy.stats import ConstantInputWarning as PearsonRConstantInputWarning


try:
	from utils import *
except:
	try:
		from .utils import *
	except:
		raise EOFError

os.environ["OMP_NUM_THREADS"] = "20"

# Project 1: from dchic
# #######################################################################################
def download_file_with_curl(url, destination):
	"""
	Downloads a file using curl command line tool.

	:param url: URL of the file to be downloaded
	:param destination: Directory path where the file will be saved
	"""
	subprocess.run(['curl', '-O', url, '-L'], check=True, cwd=destination)


def download_and_extract(genome, anno_dir, res):
	"""
	Downloads and extracts relevant genome data for a specified genome and resolution.

	:param genome: Genome identifier (e.g., 'hg19', 'mm10')
	:param anno_dir: Directory path for annotation files
	:param res: Resolution for data extraction
	:return: Paths to the extracted golden path and cytoband files
	"""
	if anno_dir is not None:
		folder = f"{anno_dir}/{genome}"
		folder_temp = f"{anno_dir}/{genome}/{genome}_{int(res)}_goldenpathData"
		if not os.path.exists(folder_temp):
			os.makedirs(folder_temp)
	else:
		folder = f"{genome}"
		folder_temp = f"{genome}/{genome}_{int(res)}_goldenpathData"
		if not os.path.exists(folder_temp):
			os.makedirs(folder_temp)
	print("folder_temp:",folder_temp)
 
	if not os.path.exists(f"{folder}/cytoBand.txt.gz"):
		print("download cytoBand.txt.gz")
		download_file_with_curl(f"http://hgdownload.cse.ucsc.edu/goldenPath/{genome}/database/cytoBand.txt.gz", folder)
  
	if not os.path.exists(f"{folder}/cytoBand.txt"):  
		print(f"Unzipping cytoBand.txt")
		subprocess.run(f"gunzip -c {folder}/cytoBand.txt.gz > {folder}/cytoBand.txt", shell=True, check=True)
		

	if not os.path.exists(f"{folder}/{genome}.chrom.sizes"):
		print(f"download {genome}.chrom.sizes")
		download_file_with_curl(f"http://hgdownload.cse.ucsc.edu/goldenPath/{genome}/bigZips/{genome}.chrom.sizes", folder)
	if not os.path.exists(f"{folder}/{genome}.refGene.gtf.gz"):
		print(f"download {genome}.refGene.gtf.gz")
		download_file_with_curl(f"http://hgdownload.cse.ucsc.edu/goldenPath/{genome}/bigZips/genes/{genome}.refGene.gtf.gz", folder)
		

	if not os.path.exists(f"{folder}/{genome}.fa.gz"):
		genome_fa_url = f"http://hgdownload.cse.ucsc.edu/goldenPath/{genome}/bigZips/{genome}.fa.gz"
		print(f"download {genome}.fa.gz")
		download_file_with_curl(genome_fa_url, folder)
	if not os.path.exists(f"{folder}/{genome}.fa"):  
		print(f"Unzipping {genome}.fa")
		subprocess.run(f"gunzip -c {folder}/{genome}.fa.gz > {folder}/{genome}.fa", shell=True, check=True)


	tss_bed_file = f"{folder}/{genome}.tss.bed"
	if not os.path.exists(tss_bed_file):
		cmd = f"gunzip -c {folder}/{genome}.refGene.gtf.gz | awk -v OFS='\\t' '{{if($3==\"transcript\"){{if($7==\"+\"){{print $1,$4,$4+1}}else{{print $1,$5-1,$5}}}}}}' | grep -v 'alt' | grep -v 'random' | sort |uniq |sort -k 1,1 -k2,2n > {folder}/{genome}.tss.bed"
	
		print(f"Running {cmd}")
		subprocess.run(cmd, shell=True, check=True)

	binned_bed_file = f"{folder_temp}/{genome}.binned.bed"
	if not os.path.exists(binned_bed_file):
		cmd = f"bedtools makewindows -g {folder}/{genome}.chrom.sizes -w {int(res)} > {folder_temp}/{genome}.binned.bed"
		print(f"Running {cmd}")
		subprocess.run(cmd, shell=True, check=True)

	gcpt_bedgraph_file = f"{folder_temp}/{genome}.GCpt.bedGraph"
	if not os.path.exists(gcpt_bedgraph_file):
		cmd = f"bedtools nuc -fi {folder}/{genome}.fa -bed  {folder_temp}/{genome}.binned.bed | grep -v '#' | awk -v OFS='\\t' '{{print $1,$2,$3,$5}}' | grep -v 'alt' | grep -v 'random' | sort -k 1,1 -k2,2n > {folder_temp}/{genome}.GCpt.bedGraph"
		print(f"Running {cmd}")
		subprocess.run(cmd, shell=True, check=True)

	gcpt_tss_bedgraph_file = f"{folder_temp}/{genome}.GCpt.tss.bedGraph"
	if not os.path.exists(gcpt_tss_bedgraph_file):
		cmd = f"bedtools map -a {folder_temp}/{genome}.GCpt.bedGraph -b {folder}/{genome}.tss.bed -c 1 -o count -null 0 > {folder_temp}/{genome}.GCpt.tss.bedGraph"
		print(f"Running {cmd}")
		subprocess.run(cmd, shell=True, check=True)
	print("GCpt.tss.bedGraph already exists")
	goldenpath = f"{folder_temp}/{genome}.GCpt.tss.bedGraph"
 
	cytoband_path = f'{folder}/cytoBand.txt'
	return goldenpath, cytoband_path

def process_and_transpose(gcc_values):
	"""
	Processes GCC values by creating a pandas DataFrame, transposing it, and renaming columns.

	:param gcc_values: Dictionary containing GCC values to be processed
	:return: Transposed and processed DataFrame
	"""
	dfs = {key: pd.DataFrame(value) for key, value in gcc_values.items()}
	result_df = pd.concat(dfs.values(), axis=1)
	result_df = result_df.transpose()
	result_df.columns = result_df.columns + '.cor'
	result_df['name'] = result_df.index
	return result_df

def pc_flip_and_selection(df,use_rows,res,chrom,goldenpath):
	"""
	Performs PCA on the input data, flips signs based on correlation with GCC, and selects the best PCs.

	:param df: Input DataFrame containing PC values
	:param use_rows: Rows to be used for the analysis
	:param res: Resolution for data extraction
	:param chrom: Chromosome identifier
	:param goldenpath: Path to the golden path file
	:return: Flipped PC values, index of the maximum PC, sign of the maximum PC, and correlation values
	"""
	gcc = pd.read_table(goldenpath, header=None, names=["chr", "start", "end", "gcc", "tss"])
	gcc.index = gcc["chr"].astype(str) + "_" + gcc["start"].astype(str)
	lhs = pd.DataFrame()
	lhs[['pc1','pc2']] = df
	lhs.index= [chrom+'_'+str(start*res) for start in use_rows]
	lhs[["gcc", "tss"]] = gcc.loc[lhs.index, ["gcc", "tss"]]
	lhs['len'] = range(1, lhs.shape[0] + 1)
	pc_sign = []
	pc_k = 2
	for k in range(pc_k): 
		a = np.sign(pearsonr(lhs.iloc[:, k], lhs["gcc"])[0])
		lhs.iloc[:, k] = a*lhs.iloc[:, k]
		pc_sign.append(a)
	pc_flip= lhs.iloc[:,0:pc_k]

	gcc_values_j=[pd.concat([pc_flip, lhs["gcc"]],axis=1).corr().iloc[:-1, -1].round(4).transpose()]
	tss_values_j=[pd.concat([pc_flip, lhs["tss"]],axis=1).corr().iloc[:-1, -1].round(4).transpose()]
	len_values_j=[pd.concat([pc_flip, lhs["len"]],axis=1).corr().iloc[:-1, -1].round(4).transpose()]
	score = [pc1+pc2 for pc1,pc2 in zip(gcc_values_j,tss_values_j)]
	max_pc_index = score.index(max(score))
	pc_flip = np.array(pc_flip.iloc[:,max_pc_index])
	pc_sign = pc_sign[max_pc_index]

	return pc_flip,max_pc_index,pc_sign,gcc_values_j,tss_values_j,len_values_j

# Project 2: from Higashi
# #######################################################################################
def pearson(matrix):
	"""
	Calculates the Pearson correlation matrix for a given data matrix.

	:param matrix: Input data matrix
	:return: Pearson correlation matrix
	"""
	return np.corrcoef(matrix)
def sqrt_norm(matrix):
	"""
	Normalizes a matrix by dividing each element by the square root of the sum of its row elements.

	:param matrix: Input matrix to be normalized
	:return: Normalized matrix
	"""
	coverage = (np.sqrt(np.sum(matrix, axis=-1)))
	with np.errstate(divide='ignore', invalid='ignore'):
		matrix = matrix / coverage.reshape((-1, 1))
		matrix = matrix / coverage.reshape((1, -1))
	matrix[np.isnan(matrix)] = 0.0
	matrix[np.isinf(matrix)] = 0.0
	return matrix

def kth_diag_indices(a, k):
	"""
	Returns the indices of the k-th diagonal for a given matrix.

	:param a: Input matrix
	:param k: Diagonal index (0 for main diagonal, positive for upper diagonals, negative for lower diagonals)
	:return: Tuple of row and column indices for the k-th diagonal
	"""
	rows, cols = np.diag_indices_from(a)
	if k < 0:
		return rows[-k:], cols[:k]
	elif k > 0:
		return rows[:-k], cols[k:]
	else:
		return rows, cols

def oe(matrix, expected = None):
	"""
	Calculates the observed/expected (OE) ratio for a contact matrix.

	:param matrix: Input contact matrix
	:param expected: Expected values for the diagonal elements (optional)
	:return: OE matrix
	"""
	new_matrix = np.zeros_like(matrix)
	for k in range(len(matrix)):
		rows, cols = kth_diag_indices(matrix, k)
		diag = np.diag(matrix,k)
		if expected is not None:
			expect = expected[k]
		else:
			expect = np.sum(diag) / (np.sum(diag != 0.0) + 1e-15)
		if expect == 0:
			new_matrix[rows, cols] = 0.0
		else:
			new_matrix[rows, cols] = diag / (expect)
	new_matrix = new_matrix + new_matrix.T
	return new_matrix


def create_mask(cytoband_path,res,k=30, chrom="chr1",origin_sparse=None):
	"""
	Creates a mask matrix for a given cytoband path, resolution, and chromosome.

	:param cytoband_path: Path to the cytoband file
	:param res: Resolution for data extraction
	:param k: Number of diagonals to consider (default is 30)
	:param chrom: Chromosome identifier (default is 'chr1')
	:param origin_sparse: Original sparse matrix (optional)
	:return: Mask matrix and split point index
	"""
	final = np.array(np.sum(origin_sparse, axis=0).todense())
	size = origin_sparse[0].shape[-1]
	a = np.zeros((size, size))
	if k > 0:
		for i in range(min(k, len(a))):
			for j in range(len(a) - i):
				a[j, j + i] = 1
				a[j + i, j] = 1
		a = np.ones_like((a)) - a

	gap = np.sum(final, axis=-1, keepdims=False) == 0

	if cytoband_path is not None:
		gap_tab = pd.read_table(cytoband_path, sep="\t", header=None)
		gap_tab.columns = ['chrom', 'start', 'end', 'name', 'type']
		name = np.array(gap_tab['name'])
		pqarm = np.array([str(s)[0] for s in name])
		gap_tab['pq_arm'] = pqarm
		gap_tab['length'] = gap_tab['end'] - gap_tab['start']
		summarize = gap_tab.groupby(['chrom', 'pq_arm']).sum(numeric_only=True).reset_index()
		if np.sum(summarize['pq_arm'] == 'p') > 0:
			split_point = \
			np.ceil(np.array(summarize[(summarize['chrom'] == chrom) & (summarize['pq_arm'] == 'p')]['length']) / res)[0]
		else:
			split_point = -1
		
		gap_list = gap_tab[(gap_tab["chrom"] == chrom) & (gap_tab["type"] == "acen")]
		start = np.floor((np.array(gap_list['start'])) / res).astype('int')
		end = np.ceil((np.array(gap_list['end'])) / res).astype('int')
		
		for s, e in zip(start, end):
			a[s:e, :] = 1
			a[:, s:e] = 1
	else:
		split_point = -1
	a[gap, :] = 1
	a[:, gap] = 1
	return a, int(split_point)


def test_compartment(matrix, return_PCA=False, model=None, expected = None):
	"""
	Tests compartments for a given matrix and returns PCA results.

	:param matrix: Input matrix to be tested
	:param return_PCA: Flag to return PCA model (default is False)
	:param model: Pre-trained PCA model (optional)
	:param expected: Expected values for the diagonal elements (optional)
	:return: PCA results and model (if return_PCA is True)
	"""
	contact = matrix
	contact = sqrt_norm(matrix)
	contact = oe(contact, expected)
	np.fill_diagonal(contact, 1)
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore", category=PearsonRConstantInputWarning
		)
		contact = pearson(contact)
	np.fill_diagonal(contact, 1)
	contact[np.isnan(contact)] = 0.0
	if model is not None:
		y = model.transform(contact)
	else:
		pca = PCA(n_components=2)
		y = pca.fit_transform(contact)
	if return_PCA:
		return y, pca
	else:
		return y

def process_one_chrom(chrom,val,config):
	"""
	Processes a single chromosome and returns compartment results.

	:param chrom: Chromosome identifier
	:param val: Values to be processed (GCC, TSS, length)
	:param config: Configuration dictionary containing parameters
	:return: Processed compartment results
	"""
	res = config["resolution"]
	raw_dir = config["raw_dir"]
	label_path = config["label_path"]
	cytoband_path = config["cytoband_path"]
	goldenpath = config["goldenpath"]
	 
	gcc_values,tss_values,len_values = val
	# Get the raw sparse mtx list
	cell_type_info=pickle.load(open(label_path, "rb"))

	cell_type_df = pd.DataFrame(cell_type_info)
	if 'cell_age' in cell_type_df.columns:
		cell_type = cell_type_info['cell_age']
	elif 'cell type'  in cell_type_df.columns:
		cell_type=cell_type_info['cell type']
	else:
		cell_type=cell_type_info['cell_type']
	origin_sparse = np.load(os.path.join(raw_dir, "%s_sparse_adj.npy" % chrom), allow_pickle=True)
	size = origin_sparse[0].shape[0]
	print(size)
	mask, split_point = create_mask(cytoband_path,res,(int(1e5)), chrom, origin_sparse)

	bulk1 = np.array(np.sum(origin_sparse, axis=0).todense())
	print(bulk1.shape)
	mask = (np.ones_like(bulk1) - mask)
	bulk1 *= mask.astype(np.int32)
  


	use_rows_all = []

	if split_point >= 20 * 1000000 / res:
		slice_start_list, slice_end_list = [0, split_point], [split_point, len(bulk1)]
	else:
		slice_start_list, slice_end_list = [0], [len(bulk1)]

	bulk_compartment_all = []
	temp_compartment_list_zscore = []
	temp_compartment_list_quantile = []

	bulk_model_list = []
	bulk_slice_list = []
	use_rows_list = []


	for slice_start, slice_end in zip(slice_start_list, slice_end_list):
		
		bulk1_slice = bulk1[slice_start:slice_end, :]
		bulk1_slice = bulk1_slice[:, slice_start:slice_end]
		use_rows = np.where(np.sum(bulk1_slice > 0, axis=-1) > 0.01 * len(bulk1_slice))[0]
		if len(use_rows) <= 1:
			print("no reliable bins in slice:", slice_start, slice_end)
			continue
		use_rows_all.append(np.arange(slice_start, slice_end)[use_rows])
		use_rows_list.append(use_rows)
		bulk1_slice = bulk1_slice[use_rows, :]
		bulk1_slice = bulk1_slice[:, use_rows]
		# print('use_rows_list',len(use_rows_list))
		
		bulk_slice_list.append(bulk1_slice)
		bulk_expect = []
		for k in range(len(bulk1_slice)):
			diag = np.diag(bulk1_slice, k)
			bulk_expect.append(np.mean(diag))
		
		
		bulk_compartment, model = test_compartment(bulk1_slice, return_PCA=True)
			
		# reverse_flag = False
		bulk_compartment_all.append(bulk_compartment)
		bulk_model_list.append(model)
	bulk_compartment = np.concatenate(bulk_compartment_all, axis=0)
	use_rows = np.concatenate(use_rows_all, axis=0)
	real_bulk_compartment,max_pc_index,pc_sign,gcc_values_j,tss_values_j,len_values_j= pc_flip_and_selection(bulk_compartment,use_rows,res,chrom,goldenpath)
	gcc_values[chrom] = gcc_values_j
	tss_values[chrom] = tss_values_j
	len_values[chrom] = len_values_j

	print(bulk_compartment.shape)
	print(len(use_rows_list))
	temp_compartment_list_all = [[] for i in range(len(use_rows_list))]
	cell_list = trange(len(origin_sparse))
	temp = np.zeros((size, size))
	for i in cell_list:
		temp *= 0.0
		proba = np.array(origin_sparse[i].todense())
		temp+= proba
		temp *= mask
		for j in range(len(use_rows_list)):
			slice_start, slice_end = slice_start_list[j], slice_end_list[j]
			temp_slice = temp[slice_start:slice_end, :]
			temp_slice = temp_slice[:, slice_start:slice_end]
			temp_select = temp_slice[use_rows_list[j], :]
			temp_select = temp_select[:, use_rows_list[j]]
			temp_compartment = test_compartment(temp_select, False, bulk_model_list[j], None)
			temp_compartment = temp_compartment[:,max_pc_index]*pc_sign
			temp_compartment_list_all[j].append(temp_compartment.reshape((-1)))
	for j in range(len(use_rows_list)):
		temp_compartment_list_all[j] = np.stack(temp_compartment_list_all[j], axis=0)
		temp_compartment_list_quantile.append(quantile_transform(temp_compartment_list_all[j], output_distribution='uniform',
												   n_quantiles=int(temp_compartment_list_all[j].shape[-1] * 1.0), axis=1))
		
		temp_compartment_list_zscore.append(zscore(temp_compartment_list_all[j], axis=1))
	temp_compartment_list = np.concatenate(temp_compartment_list_all, axis=-1)

	temp_compartment_list_zscore = np.concatenate(temp_compartment_list_zscore, axis=-1)
	temp_compartment_list_quantile = np.concatenate(temp_compartment_list_quantile, axis=-1)

	origin_bulk_list_all=[[] for i in range(len(use_rows_list))]
	a={}
	b={}
	for j in list(set(cell_type)):
		
		indices = [index for index, value in enumerate(cell_type) if value == j]
		a[j]=indices
		b[j] = np.zeros_like(np.array(origin_sparse[0]))
		for i in a[j]:
				proba = np.array(origin_sparse[i])    
				b[j] +=proba
		temp = np.array(b[j].item().todense())
		temp *= mask
		for j in range(len(use_rows_list)):
			slice_start, slice_end = slice_start_list[j], slice_end_list[j]
			temp_slice = temp[slice_start:slice_end, :]
			temp_slice = temp_slice[:, slice_start:slice_end]
			temp_select = temp_slice[use_rows_list[j], :]
			temp_select = temp_select[:, use_rows_list[j]]
			temp_compartment = test_compartment(temp_select, False, bulk_model_list[j], None)
			temp_compartment = temp_compartment[:,max_pc_index]*pc_sign #pc值校正
			origin_bulk_list_all[j].append(temp_compartment.reshape((-1)))
		
	for j in range(len(use_rows_list)):
		origin_bulk_list_all[j] = np.stack(origin_bulk_list_all[j], axis=0)
	presudo_bulk_list = np.concatenate(origin_bulk_list_all, axis=-1)
	print (chrom, "finished")
	
	return presudo_bulk_list, temp_compartment_list,temp_compartment_list_zscore,temp_compartment_list_quantile,use_rows,size

def start_call_compartment(config):
	"""
	Starts the compartment calling process for a given configuration.

	:param config: Configuration dictionary containing parameters
	:return: Save path for the compartment results
	"""
	if isinstance(config, str):
		config = get_config(args.config)
	genome = config["genome"]
	res = config["resolution"]
	raw_dir = config["raw_dir"]
	label_path = config["label_path"]
	chrom_list =  config["chrom_list"]
	anno_dir = config.get('anno_dir',None)
	goldenpath, cytoband_path = download_and_extract(genome, anno_dir, res)
	print(raw_dir)
 
	config["goldenpath"] = goldenpath
	config["cytoband_path"] = cytoband_path
 
	gcc_values = {}
	tss_values = {}
	len_values = {}
	val = (gcc_values,tss_values,len_values)
	vals_path = os.path.join(raw_dir, "vals.txt")
	
	output = config.get('output',"scCompartment")
	if ".hdf5" not in output:
		output += ".hdf5"
	if "data_path" in config:
		save_path = config["data_path"]
	else:
		save_path = os.path.join(raw_dir, output)
	with h5py.File(save_path, "w") as output_f:
		result = {}
		for chrom in chrom_list:
			presudo_bulk_list, temp_compartment_list,temp_compartment_list_zscore,temp_compartment_list_quantile,use_rows,size = process_one_chrom(chrom,val,config)
			result[chrom] = [presudo_bulk_list, temp_compartment_list,temp_compartment_list_zscore,temp_compartment_list_quantile,use_rows,size]

		gcc_df = process_and_transpose(gcc_values)
		tss_df = process_and_transpose(tss_values)
		len_df = process_and_transpose(len_values)
		vals = pd.DataFrame()
		chr_list = [[x]*2 for x in chrom_list]
		vals['chr'] = [item for ls in chr_list for item in ls]
		vals['gcc.cor'] = gcc_df['gcc.cor'].values
		vals['tss.cor'] = tss_df['tss.cor'].values
		vals['len.cor'] = len_df['len.cor'].values
		vals['pc'] = gcc_df.index
		vals.to_csv(vals_path, sep="\t", header = False,index=False)

		bin_chrom_list = []
		bin_start_list = []
		bin_end_list = []
		presudo_bulk_all = []
		sc_cp_raw = []
		sc_cp_zscore = []
		sc_cp_quantile = []


		
		for chrom in chrom_list:
			presudo_bulk_list, temp_compartment_list,temp_compartment_list_zscore,temp_compartment_list_quantile,use_rows,size= result[chrom]
			length = size
			bin_chrom_list += [chrom] * len(use_rows)
			bin_start_list.append((np.arange(length) * res).astype('int')[use_rows])
			bin_end_list.append(((np.arange(length) + 1) * res).astype('int')[use_rows])
			presudo_bulk_all.append(presudo_bulk_list)
			sc_cp_raw.append(temp_compartment_list)
			sc_cp_zscore.append(temp_compartment_list_zscore)
			sc_cp_quantile.append(temp_compartment_list_quantile)
			
		presudo_bulk_all = np.concatenate(presudo_bulk_all, axis=-1)
		print('presudo_bulk_all.shape:',presudo_bulk_all.shape)
		sc_cp_raw = np.concatenate(sc_cp_raw, axis=-1)
		sc_cp_zscore = np.concatenate(sc_cp_zscore, axis=-1)
		sc_cp_quantile=np.concatenate(sc_cp_quantile, axis=-1)
		
		grp = output_f.create_group('compartment_raw')
		bin = grp.create_group('bin')
		bin.create_dataset('chrom', data=[l.encode('utf8') for l in bin_chrom_list],
						   dtype=h5py.special_dtype(vlen=str))
		bin.create_dataset('start', data=np.concatenate(bin_start_list))
		bin.create_dataset('end', data=np.concatenate(bin_end_list))
		for cell in range(len(sc_cp_raw)):
			grp.create_dataset("cell_%d" % cell, data=sc_cp_raw[cell])

		grp = output_f.create_group('compartment_zscore')
		bin = grp.create_group('bin')
		bin.create_dataset('chrom', data=[l.encode('utf8') for l in bin_chrom_list],
						   dtype=h5py.special_dtype(vlen=str))
		bin.create_dataset('start', data=np.concatenate(bin_start_list))
		bin.create_dataset('end', data=np.concatenate(bin_end_list))
		for cell in range(len(sc_cp_zscore)):
			grp.create_dataset("cell_%d" % cell, data=sc_cp_zscore[cell])

		grp = output_f.create_group('compartment')
		bin = grp.create_group('bin')
		bin.create_dataset('chrom', data=[l.encode('utf8') for l in bin_chrom_list],
						   dtype=h5py.special_dtype(vlen=str))
		bin.create_dataset('start', data=np.concatenate(bin_start_list))
		bin.create_dataset('end', data=np.concatenate(bin_end_list))
		for cell in range(len(sc_cp_quantile)):
			grp.create_dataset("cell_%d" % cell, data=sc_cp_quantile[cell])



		grp = output_f.create_group('compartment_presudo_bulk')
		bin = grp.create_group('bin')
		bin.create_dataset('chrom', data=[l.encode('utf8') for l in bin_chrom_list],
						   dtype=h5py.special_dtype(vlen=str))
		bin.create_dataset('start', data=np.concatenate(bin_start_list))
		bin.create_dataset('end', data=np.concatenate(bin_end_list))
		cell_type_info=pickle.load(open(label_path, "rb"))
		cell_type_df = pd.DataFrame(cell_type_info)
		if 'cell_age' in cell_type_df.columns:
			cell_type = cell_type_info['cell_age']
		elif 'cell type'  in cell_type_df.columns:
			cell_type=cell_type_info['cell type']
		else:
			cell_type=cell_type_info['cell_type']
		cell_info = list(set(cell_type))
		for j in range(len(cell_info)):
			grp.create_dataset(cell_info[j], data=presudo_bulk_all[j,:])
			
	output_f.close()
	return save_path

	
if __name__ == '__main__':
	args = common_parse_args(desc="scCompartment calling")
	print(args)
	config = get_config(args.config)
	pid = os.getpid()
	print('pid : ',pid)
	start_call_compartment(config)
