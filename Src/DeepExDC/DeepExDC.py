import argparse
import logging
try:
	from . import utils
except:
	import utils
def parse_args():
	parser = argparse.ArgumentParser(description="DeepExDC main program")
	parser.add_argument('-c', '--config', type=str, default="./config.JSON")
	parser.add_argument('-s', '--start', type=int, default=1)
	parser.add_argument('-e', '--end', type=int, default=4)
	
	return parser.parse_args()

class DeepExDC():
	def __init__(self, config_path):
		super().__init__()
		self.config_path = config_path
		self.config = utils.get_config(config_path)
		utils.create_dirs(self.config)

  
	def cal_comparment(self):
		try:
			from compartment_score import start_call_compartment
		except:
			from .compartment_score import start_call_compartment
		start_call_compartment(self.config)
		
  
	
	def train_model(self):
		try:
			from train import train
		except:
			from .train import train
		train(self.config)

	
	def run_shap(self):
		try:
			from deepexplation import RunSHAP
		except:
			from .deepexplation import RunSHAP

		runshap=RunSHAP(self.config)
		shap_obj=runshap.run()
		self.shap_obj = shap_obj
  
		
 
 
	def run_dc(self):
		try:
			from getdiffcompartments import getdiff
		except:
			from .getdiffcompartments import getdiff
		if "shap_path" in self.config:
			getdiff(config=self.config)
		elif isinstance(self.shap_obj, str):
			self.config["shap_path"] = self.shap_obj
			getdiff(config=self.config)
		else:
			getdiff(config=self.config,shap_data=self.shap_obj[0],shap_label=self.shap_obj[1])

if __name__ == '__main__':
	# Get parameters from config file
	args = parse_args()
	config_path = args.config
	deepexdc = DeepExDC(config_path)
	# deepexdc.cal_comparment()
	# deepexdc.train_model()
	# deepexdc.run_shap()
	# deepexdc.run_dc()
	# Define the steps
	steps = {
		1: deepexdc.cal_comparment,
		2: deepexdc.train_model,
		3: deepexdc.run_shap,
		4: deepexdc.run_dc
	}

	# Execute steps based on start and end
	for step in range(args.start, args.end + 1):
		if step in steps:
			steps[step]()
		else:
			logging.warning(f"Step {step} is not defined.")
 