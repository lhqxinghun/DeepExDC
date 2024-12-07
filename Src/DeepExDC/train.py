try:
	from utils import get_config, common_parse_args
	from process import read_dataset, somote_dataset, split_dataset
except:
	try:
		from .utils import get_config, common_parse_args
		from .process import read_dataset, somote_dataset, split_dataset
	except:
		raise EOFError
# The following code is adapted from an open-source project licensed under the GNU General Public License v3.
# Project: dl-4-tsc
# Source: https://github.com/hfawaz/dl-4-tsc/tree/master

def fit_classifier(datasets,config):
	"""
	Fits a classifier to the training data and evaluates it on the test data.

	Args:
		datasets (tuple): A tuple containing the training and testing data and labels.
		config (dict): A dictionary containing configuration parameters.

	Returns:
		str: The path to the saved model.

	Raises:
		Exception: If the input data is not in the correct format.
	"""
	x_train = datasets[0]
	y_train = datasets[1]
	x_test = datasets[2]
	y_test = datasets[3]

	nb_classes = y_test.shape[1]


	if len(x_train.shape) == 2:  # if univariate
		# add a dimension to make it multivariate with one dimension 
		x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
		x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

	input_shape = x_train.shape[1:]
	classifier = create_classifier(config["classifier_name"], input_shape, nb_classes, config)

	model_path = classifier.fit(x_train, y_train, x_test, y_test)
	return model_path

def create_classifier(classifier_name, input_shape, nb_classes, config, verbose=True):
	"""
	Creates a classifier instance based on the given name.

	Args:
		classifier_name (str): The name of the classifier to create.
		input_shape (tuple): The shape of the input data.
		nb_classes (int): The number of classes in the classification problem.
		config (dict): A dictionary containing configuration parameters.
		verbose (bool): Whether to print verbose output. Defaults to True.

	Returns:
		Classifier: An instance of the specified classifier.
	"""
	if classifier_name == 'cnn':
		try: 
			from classifiers import Classifier_CNN
		except:
			from .classifiers import Classifier_CNN
		return Classifier_CNN(config=config, input_shape=input_shape, nb_classes=nb_classes, verbose=verbose)

def train(config):
	"""
	Trains a classifier using the given configuration.

	Args:
		config (dict or str): A dictionary containing configuration parameters or a string pointing to a configuration file.

	Returns:
		str: The path to the saved model.

	Raises:
		EOFError: If the configuration file cannot be loaded.
	"""
	if isinstance(config, str):
		config = get_config(args.config)
	smote = config.get("smote",True)
	if "data_path "in config:
		data_path = config["data_path"]
	else:
		data_path = f'{config["raw_dir"]}/scCompartment.hdf5'
	label_path = config["label_path"]
	classifier_name = config.get('classifier_name','cnn')
	config["classifier_name"] = classifier_name
	data,label=read_dataset(config)
	
	if smote is True:
		data_smote,label_smote=somote_dataset(data,label)
		datasets = split_dataset(data_smote,label_smote)
	else:
		datasets = split_dataset(data,label,config)
	
	model_path=fit_classifier(datasets,config)
	return model_path

if __name__ == '__main__':    
	args = common_parse_args(desc="classification of single cells")
	print(args)
	config = get_config(args.config)
	train(config)