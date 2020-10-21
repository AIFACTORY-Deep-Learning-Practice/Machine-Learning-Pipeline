# Machine-Learning-Pipeline

This repository was made for providing basic machine learning process and pipeline system. I made machine learning process by using `scikit-learn`. I hope you can get help from this repo.

# Environments

**Requirment**

```bash
scikit-learn == 0.22.1
```

**ML task**

- Regression
- Classification

# ML in scikit-learn

## Models

**Linear Models**

- Linear Regression
- Logistic Regression
- Ridge
- Lasso
- ElasticNet

**Tree-based Models**

- Decision Tree
- Random Forest
- Adaboost
- Gradient Tree Boosting

**Support Vector Machine**

- SVC (classification)
- SVR (regression)

**Naighbors**

- K-Nearest Neighbors

## Evaluation

**Regression**

- Mean Squared Error, MSE
- Mean Absolued Error, MAE

**Classification**

- Accuracy
- Recall
- Precision
- F1-score
- AUC

## Save & Load

- pickle

# Process

**Basic process for machine learning (ML)**

1. Load Data
2. Preprocessing Data
3. Model Building
4. Model Training
5. Evaluation
6. Save

Let's try coding with examples

- [Regression Notbook](https://github.com/DataNetworkAnalysis/Machine-Learning-Pipeline/blob/main/notebook/Process%20(Regression).ipynb)
- [Classification Notbook](https://github.com/DataNetworkAnalysis/Machine-Learning-Pipeline/blob/main/notebook/Process%20(Classification).ipynb)

# Pipeline

**Scripts**

```bash
.
├── dataload.py
├── evaluate.py
├── main.py
├── model.py
├── train.py
└── utils.py
```

## main.py

**Argument**

```
Machine Learning Pipeline

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Set seed
  --task {regression,classification}
                        Choice machine learning task
  --datagen DATAGEN     Generate train and test set
  --datadir DATADIR     Set data directory
  --logdir LOGDIR       Set log directory
  --val_size VAL_SIZE   Set validation size
  --kfold KFOLD         Number of cross validation
  --modelname {OLS,Logistic,Ridge,Lasso,ElasticNet,DT,RF,ADA,GT,SVM,KNN}
                        Choice machine learning model
```

**Code**

```python
# 1. load data
train, test = dataloader(task=args.task, datadir=args.datadir)

# 2. preprocessing
x_train, y_train = preprocessing(data=train, data_type='train')
x_test = preprocessing(data=test, data_type='test')

# 3. model setting
model = SklearnModels(task=args.task, modelname=args.modelname, random_state=args.seed)

# 4. training
if args.kfold:
    cross_validation(K=args.kfold, model=model, train=[x_train, y_train], test=x_test, args=args)
else:
    training(model=model, train=[x_train, y_train], test=x_test, args=args)
```

## model.py

```python
class SklearnModels:
    def __init__(self, **kwargs):
        pass

    def build(self, **kwargs):
        pass

    def fit(self, **kwargs):
        pass

    def eval(self, **kwargs):
        pass
        
    def predict(self, **kwargs):
        pass

    def load(self,, **kwargs):
        pass
```


## dataload.py

```python
def data_generator(**kwargs):
    pass

def dataloader(**kwargs):
    pass

def preprocessing(**kwargs):
    pass
```

## train.py

```python
def training(model, train: list, test, args):
    pass

def cross_validation(K, model, train: list, test, args):
    pass
```

## evaluate.py

```python
def evaluation_all(task, y_true, y_pred, threshold=0.5):
    pass
```

## utils.py

```python
def results_save(**kwargs):
    pass

def results_comparison(**kwargs):
    pass

def results_bar(**kwargs):
    pass
```