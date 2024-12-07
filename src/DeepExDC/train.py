try:
	from utils import get_config, common_parse_args
	from process import read_dataset, somote_dataset, split_dataset
except:
	try:
		from .utils import get_config, common_parse_args
		from .process import read_dataset, somote_dataset, split_dataset
	except:
		raise EOFError

def fit_classifier(datasets,config):
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
	if classifier_name == 'cnn':  # 1D-CNN （单通道）
		try: 
			from classifiers import Classifier_CNN
		except:
			from .classifiers import Classifier_CNN
		return Classifier_CNN(config=config, input_shape=input_shape, nb_classes=nb_classes, verbose=verbose)

def train(config):
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