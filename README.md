# NeuralSymbolicRegressionThatScales
## The repo documentation is currently under development, please check back soon for more information.

Pytorch implementation and pretrained models for the paper "Neural Symbolic Regression That Scales" 
For details, see **Neural Symbolic Regression That Scales**.  
[[`arXiv`](https://arxiv.org/pdf/2106.06427.pdf)] 


## Installation
Please clone and install this repository via

```
git clone https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales.git
cd NeuralSymbolicRegressionThatScales/
pip3 install . -e
```

This library requires python>3.7



## Pretrained models
We offers two models "10M" and "100M". The first, trained on a dataset of 10M and without constant prediction, is the one used for experiements in our paper. The second is trained on 100M of equations and with constant prediction enabled.
For both models, the equations included in data/tests.csv are held out during training.

If you want to try the models out, look at **jupyter/fit_func.ipynb**.


## Dataset Generation
Before training, you need a dataset of equations. Here the steps to follow

### Raw training dataset generation
There are two ways to generate this dataset:

* If you are running on linux, you use makefile in terminal as follows:
```
export NUM=${NumberOfEquations} #Export num variable
make data/raw_datasets/${NUM}: #Launch make file command
```
NumberOfEquations can be defined in two formats with K or M suffix. For instance 100K is equal to 100'000 while 10M is equal to 10'0000000
For example, if you want to create a 10M dataset simply:

```
export NUM=10M #Export num variable
make data/raw_datasets/10M: #Launch make file command
```

* Run this script: 
```
python3 scripts/data_creation/dataset_creation.py --number_of_equations NumberOfEquations --no-debug #Replace NumberOfEquations with the number of equations you want to generate
```

After this command you will have a folder named **data/raw_data/NumberOfEquations** containing .h5 files. By default, each of this h5 files contains a maximum of 5e4 equations. The code generators scripts are based on Code for generating the dataset is largely based on [https://github.com/facebookresearch/SymbolicMathematics]


### Raw test dataset generation
This step is optional. You can skip it if you want to use our test set used for the paper (located in **test_set/nc.csv**).
Use the same commands as before for generating a validation dataset. All equations in this dataset will be remove from the training dataset in the next stage, 
hence this validation dataset should be **small**. For our paper it constisted of 200 equations.

```
#Code for generating a 150 equation dataset 
python3 scripts/data_creation/dataset_creation.py --number_of_equations 150 --no-debug 
```

Now you convert the newly created validation dataset in the csv format. First in **scripts/config.yaml** replace the entry of *raw_test_path* with the path to your test set. For instance if you have created a dataset equations it would be **data/raw_datasets/150**) then run: `python3 scripts/csv_handling/dataload_format_to_csv.py`

This command will create csv file named test.csv in the test_set folder.

### Remove test equations from the training dataset

python3 scripts/data_creation/filter_from_already_existing.py --data_path data/raw_datasets/${NUM} --csv_path pathToValidate equations #You can leave csv_path empty if you want to create a validation set
python3 scripts/data_creation/filter_validation.py --val_path data/datasets/${NUM}/${NUM}_val
python3 scripts/data_creation/to_h5.py --folder_dataset data/datasets/${NUM} 
```




## Training
Once you have created your training and validation datasets run 
```
python3 scripts/train.py
```
You can configure the config.yaml with the necessary options
