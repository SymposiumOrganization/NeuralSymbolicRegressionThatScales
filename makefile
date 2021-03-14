SHELL := /bin/bash


data/raw_datasets/${NUM}:
	python3 scripts/data_creation/dataset_creation.py --number_of_equations $${NUM:0:$${#NUM}-1}000 --no-debug

data/datasets/${NUM}/.dirstamp: data/raw_datasets/${NUM}
	python3 scripts/data_creation/split_train_val.py --data_path $?

data/datasets/${NUM}/${NUM}_subset: data/datasets/${NUM}/.dirstamp
	python3 scripts/data_creation/filter_validation.py --val_path data/datasets/${NUM}/${NUM}_val