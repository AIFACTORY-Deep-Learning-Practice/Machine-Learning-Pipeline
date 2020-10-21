import argparse

from dataload import dataloader
from model import SklearnModels

import os

import warnings
warnings.filterwarnings(action='ignore')

if __name__=='__main__':
    # config
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline")
    parser.add_argument('--seed', type=int, default=223, help='Set seed')
    parser.add_argument('--task', type=str, choices=['regression','classification'], help='Choice machine learning task')
    parser.add_argument('--modelname', type=str, 
                        choices=['OLS','Ridge','Lasso','ElasticNet','DT','RF','ADA','GT','SVM','KNN'],
                        help='Choice machine learning model')
    args = parser.parse_args()

    # hard coding
    n_samples = 1000
    n_features = 10
    test_size = 0.3
    noise = 5

    # dataload
    x_train, x_test, y_train, y_test = dataloader(task=args.task, 
                                                  n_samples=n_samples, 
                                                  n_features=n_features, 
                                                  noise=noise,
                                                  test_size=test_size,
                                                  random_state=args.seed)

    # build model
    model = SklearnModels(task=args.task, modelname=args.modelname)
    model.build(random_state=args.seed)

    # training
    model.fit(X=x_train, y=y_train)

    # prediction 
    model.eval(X=x_test, y=y_test)


        
