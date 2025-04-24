import json
import re
import os
from matplotlib import pyplot as plt
import pandas as pd
import argparse

def get_config(config_path = "./config.jSON"):

	c = open(config_path,"r")
	config = json.load(c)
	config["anno_dir"] = config.get("anno_dir",None)
	config["genome"] = config.get("genome",None)
	config["chrom_list"] = config.get("chrom_list",None)
	config["resolution"] = config.get("resolution",None)
	config["raw_dir"] = config.get("raw_dir",None)
	config["data_path"] = config.get("data_path", f'{config["raw_dir"]}/scCompartment.hdf5')
	config["label_path"] = config.get("label_path", f'{config["raw_dir"]}/label_info.pickle')
	config["select_label"] = config.get("select_label",None)
	config["output_dir"] = config.get("output_dir",f'{os.getcwd()}/dc_result')
	config["group"] = config.get("group","compartment_raw")
	config["scaler_type"] = config.get("scaler_type", "minmax")
	config["feature_range"] = config.get("feature_range",(-1,1))
	config["feature_name"] = config.get("feature_name","bin")
	config["feature_df_path"] = config.get("feature_df_path",f'{config["output_dir"]}/feature_df.csv')
	config["classifier_name"] = config.get('classifier_name','cnn')
	config["smote"] = config.get("smote", True)
	config["acti"] = config.get("acti","tanh")
	config["batch_size"] = config.get("batch_size",16)
	config["epoch"] = config.get("epoch",60)
	config["learning_rate"] = config.get("learning_rate",0.001)
	config["sim_parm"] = config.get("sim_parm", None)
	config["model_dir"] = config.get("model_dir", f'{config["output_dir"]}/{config["classifier_name"]}')
  
	if config["sim_parm"] is not None:
		config["model_path"] = config.get("model_path",f'{config["model_dir"]}/{config["sim_parm"]}_best_model.h5')
	else:
		config["model_path"] = config.get("model_path",f'{config["model_dir"]}/best_model.h5')



	config["optimizer"] = config.get("optimizer","adam")
	config["base_filters"] = config.get("base_filters",8)
	config["num_layers"] = config.get("num_layers",2)

	config["α"] = config.get("α", 100)
	config["β"] = config.get("β", 0.2)

	config["explain_mode"] = config.get("explain_mode","test")
	config["random_background"] = config.get("random_background", False)
	config["shap_dir"] = config.get("shap_dir",f'{config["output_dir"]}/shap')
	if config["sim_parm"] is not None:
		config["shap_path"] = config.get("shap_path",f'{config["output_dir"]}/shap/{config["sim_parm"]}_{config["group"]}_{config["explain_mode"]}_shap.h5')
			
	else:
		config["shap_path"] = config.get("shap_path",f'{config["output_dir"]}/shap/{config["group"]}_{config["explain_mode"]}_shap.h5')
	config["save_shap_file"] = config.get("save_shap_file",False)
	config["dist_method"] = config.get("dist_method","chord")
	config["result_dir"] = config.get("result_dir", f'{config["output_dir"]}/result')
	config["result_path"] = config.get("result_path", f'{config["output_dir"]}/result/{config["dist_method"]}_{config["explain_mode"]}.csv')
	create_dirs(config)
	
	
	return config

def common_parse_args(desc=None):
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('-c', '--config', type=str, default="./config.JSON")
	return parser.parse_args()


def create_dirs(config):
	
	
	temp_dir = config['output_dir']
	
	if not os.path.exists(temp_dir):
		os.makedirs(temp_dir)
	
	model_dir = config["model_dir"]
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	
	
	shap_dir = config["shap_dir"]
	if not os.path.exists(shap_dir):
		os.makedirs(shap_dir)

	result_dir = config["result_dir"]
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
  
# The following code is adapted from an open-source project licensed under the GNU General Public License v3.
# Project: dl-4-tsc
# Source: https://github.com/hfawaz/dl-4-tsc/tree/master

def plot_epochs_metric(hist, file_name, metric='loss'):
	plt.figure()
	plt.plot(hist.history[metric])
	plt.plot(hist.history['val_' + metric])
	plt.title('model ' + metric)
	plt.ylabel(metric, fontsize='large')
	plt.xlabel('epoch', fontsize='large')
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig(file_name, bbox_inches='tight')
	plt.close()

def save_logs(output_directory, hist):
	hist_df = pd.DataFrame(hist.history)
	hist_df.to_csv(output_directory + '/history.csv', index=False)
	plot_epochs_metric(hist, output_directory + '/epochs_loss.png')
	plot_epochs_metric(hist, output_directory + '/epochs_acc.png', metric='accuracy')