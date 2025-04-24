# DeepExDC

## 1. Introduction
DeepExDC is an interpretable one-dimensional convolutional neural network for differential analysis of A/B compartments in scHi-C data across multiple populations of individuals.

## 2. Installation & example

**2.1 OS**

- ubuntu 18.04


**2.2 Required Python Packages**

Make sure all the packages listed in the requirements.txt are installed.

-bedtools

-python>=3.9

-numpy==1.26.0

Bedtools recommends using Conda for installation

```bash
conda install -c bioconda bedtools
```


**2.3 Install from Github**

```bash
git clone ...
cd DeepExDC
pip install .
```

**2.4 Run example**

The ./example directory provides complete notebook examples for three types of data

or
```bash
cd /src/DeepEXDC
python DeepExDC.py -c /path/to/config.JSON
```
or

```bash
bash run_DeepExDC.sh /path/to/config.JSON
```

## 3. Usage
### 3.1 Config parameters

All customizable parameters are stored in a JSON config file. The path to this JSON config file will be needed when running the program.

For all parameters below, when certain parameter is marked as Optional it means you can left those parameters out when they are not applicable.

Data processing related parameters
| Parameter Name | Type | Required/Optional | Description | Example |
|----------------|------|------------------|-------------|---------|
| anno_dir | str | Required when use compartment_score | Directory for annotation files | "/path/to/annotations" |
| genome | str | Required when use compartment_score | Genome assembly used | "mm9" |
| chrom_list | list | Required when use compartment_score | List of chromosomes to include | ["chr1", "chr2", ..., "chrX"] |
| resolution | int | Required when use compartment_score | Resolution of the data in bp | 1000000 |
| raw_dir | str | Required when use compartment_score | Directory where raw data files are stored | "/path/to/raw" |
| data_path | str | Required when other types of data are used | Path to the processed data file | "/path/to/raw/scCompartment.hdf5" |
| label_path | str | Required | Path to the label information file | "/path/to/raw/label_info.pickle" |
| select_label | str | Optional | Criteria for selecting labels | ["cell_type1", "cell_type2", ..., "cell_typen"] |
| output_dir | str | Required | Directory for output files | "/path/to/output/dc_result" |
| group | str | Required when other types of data are used | Group identifier for the data | "compartment_raw" |
| scaler_type | str | Optional | Type of scaler used for normalization | "minmax" |
| feature_range | tuple | Optional | Range of values for feature scaling | (-1, 1) |
| feature_name | str | Required when other types of data are used | Name of the feature used | "bin" |
| feature_df_path | str | Optional | Path to the feature data frame file | "/path/to/output/feature_df.csv" |

Training process related parameters
| Parameter Name | Type | Required/Optional | Description | Example |
|----------------|------|------------------|-------------|---------|
| classifier_name | str | Optional | Name of the classifier model | "cnn" |
| smote | bool | Optional | Whether to use SMOTE for balancing classes | True |
| acti | str | Optional | Activation function type | "tanh" |
| batch_size | int | Optional | Number of samples per gradient update | 16 |
| epoch | int | Optional | Number of epochs to train the model | 60 |
| learning_rate | float | Optional | Learning rate for the model | 0.001 |
| optimizer | str | Optional | Select the optimizer| adam,['adam', 'rmsprop', 'sgd', 'nadam'] |
| base_filters | int | Optional | The number of convolutional layer base filters | 8 |
| num_layers | int | Optional | The number of convolution blocks in the model | 2 |
| sim_parm | str | Optional | Simulation parameters, if any | "group2_run1 |
| model_dir | str | Optional | Directory to save the model files | "/path/to/output/dc_result/cnn" |
| model_path | str | Optional | Path to save the best model file | "/path/to/output/dc_result/cnn/best_model.h5" |

Shapley explanation related parameters
| Parameter Name | Type | Required/Optional | Description | Example |
|----------------|------|------------------|-------------|---------|
| explain_mode | str | Optional | Mode for explaining the model | "test" |
| α | int | Optional | Select the number of samples to use as background data | 100 |
| β | float | Optional | Select the proportion of the sample you want to interpret | 0.2 |
| random_background | bool | Optional | Whether to use random background | False |
| shap_dir | str | Optional | Directory to save SHAP files | "/path/to/output/shap" |
| save_shap_file | bool | Optional | Whether to save SHAP files | False |

Differential analysis related parameters
| Parameter Name | Type | Required/Optional | Description | Example |
|----------------|------|------------------|-------------|---------|
| dist_method | str | Optional | Method for distance calculation | "chord" ,['chord','manhattan','euclidean','canberra']|
| result_dir | str | Optional | Directory for result files | "/path/to/output/result" |
| result_path | str | Optional | Path for the result CSV file | "/path/to/output/result/chord_test.csv" |

### 3.2 Input files
config.JSON
```json
{
    "anno_dir":"/home/work/annotation",
    "raw_dir":"/home/work/DeepEXDC/data/schic_data/raw",
    "label_path":"/home/work/DeepEXDC/data/schic_data/label_info.pickle",
    "output_dir":"/home/work/DeepEXDC/examples/scHiCEXPOUTPUT",
    "genome" : "mm9",
    "resolution":1000000,
    "group": "compartment_raw",
    "chrom_list": ["chr1","chr2","chr3","chr4","chr5",
    "chr6","chr7","chr8","chr9","chr10",
    "chr11","chr12","chr13","chr14","chr15",
    "chr16","chr17","chr18","chr19","chrX"],
    "smote":true,
    "explain_mode":"test",
    "save_shap_file":true,
    "dist_method": "chord",
    "random_background":false,
    "scaler_type": "minmax",
    "feature_range": [-1, 1]
  }
```

Single-cell HiC contact matrices stored as sparse matrix format per chromosome 
```
└── raw_dir
    ├── chr10_sparse_adj.npy
    ├── chr11_sparse_adj.npy
    ├── chr12_sparse_adj.npy
    ├── chr13_sparse_adj.npy

```
Single cell compartment score or other data stored in h5 format
```
└── group
    ├── cell_0
    ├── cell_1
└── feature_name
    ├── feature_name
```
label_info.pickle:Label file for storing cell types
```python
import pickle
output_label_file = open("label_info.pickle", "wb")
label_info = {
  'cell type': ['GM12878', 'H1ESC', 'HAP1',.....,'GM12878'],
}
pickle.dump(label_info, output_label_file)
```
