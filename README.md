# NeuralSymbolicRegressionThatScales
Pytorch implementation and pretrained models for the paper "Neural Symbolic Regression That Scales" 
For details, see **Emerging Properties in Self-Supervised Vision Transformers**.  
[[`arXiv`](Missing Link] 


## Pretrained models
We offers two models "10M" and "100M". The first, trained on a dataset of 10M and with constant prediction set to false, is the one used for experiements in our paper. The second is trained on 100M of equations and with constant prediction set to true.
For both models, the equations included in data/tests.csv are held out during training


## Dataset Generation
Before training, you need to create the training and validation sets. Code for generating the dataset is largely based on [https://github.com/facebookresearch/SymbolicMathematics]
If you are running on linux, you can use makefile as follows:
First define and export a variable command export NUM=${NumberOfEquationsYouWant}. 
NumberOfEquationsYouWant can be defined in two formats with K or M suffix. For instance 100K is equal to 100 000 while 10M is equal to 10 000 000



Alternatevely run the following commands:
```
python3 scripts/data_creation/dataset_creation.py --number_of_equations 100000 --no-debug #Replace 100000 with the number of equations you want to generate
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
