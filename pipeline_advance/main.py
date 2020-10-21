import argparse

from dataload import dataloader, preprocessing
from model import SklearnModels
from utils import data_generator
from train import training, cross_validation

import os
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')


if __name__=='__main__':
    # config
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline")
    parser.add_argument('--seed', type=int, default=223, help='Set seed')
    parser.add_argument('--task', type=str, choices=['regression','classification'], help='Choice machine learning task')
    parser.add_argument('--datagen', type=bool, default=True, help='Generate train and test set')
    parser.add_argument('--datadir', type=str, default='../data', help='Set data directory')
    parser.add_argument('--logdir', type=str, default='../logs', help='Set log directory')
    parser.add_argument('--val_size', type=float, default=None, help='Set validation size')
    parser.add_argument('--kfold', type=int, default=None, help='Number of cross validation')
    parser.add_argument('--modelname', type=str, 
                        choices=['OLS','Logistic','Ridge','Lasso','ElasticNet','DT','RF','ADA','GT','SVM','KNN'],
                        help='Choice machine learning model')
    args = parser.parse_args()

    # generate train and test
    if args.datagen:
        # hard coding data generate parameters
        n_samples = 10000
        n_features = 10
        test_size = 0.3
        noise = 5
        
        data_generator(task=args.task, n_samples=n_samples, n_features=n_features, noise=noise, test_size=test_size, random_state=args.seed)
    
    # load data
    train, test = dataloader(task=args.task, datadir=args.datadir)
    
    # preprocessing
    x_train, y_train = preprocessing(data=train, data_type='train')
    x_test = preprocessing(data=test, data_type='test')

    # model setting
    model = SklearnModels(task=args.task, modelname=args.modelname, random_state=args.seed)

    # training
    if args.kfold:
        cross_validation(K=args.kfold, model=model, train=[x_train, y_train], test=x_test, args=args)
    else:
        training(model=model, train=[x_train, y_train], test=x_test, args=args)
    

        
