from sklearn.model_selection import KFold 

from utils import results_save
from evaluate import evaluation_all

import pandas as pd
import os 


def training(model, train: list, test, args):
    # data
    x_train, y_train = train
    x_test = test 
    y_test = pd.read_csv(os.path.join(args.datadir, f'y_true({args.task}).csv'))['y']
    submission = pd.read_csv(os.path.join(args.datadir, f'submission({args.task}).csv'))

    # build model
    model.build()

    # training
    val_results = model.fit(X=x_train, y=y_train, validation_size=args.val_size)

    # prediction 
    y_pred = model.predict(X=x_test)

    # evaluation
    test_results = evaluation_all(task=args.task, y_true=y_test, y_pred=y_pred)

    # save results
    submission['y'] = y_pred
    results_save(args=args,
                 model=model, 
                 validation_results=val_results,
                 test_results=test_results,
                 submission=submission)


def cross_validation(K, model, train: list, test, args):
    # data
    x_train, y_train = train
    x_test = test 
    y_test = pd.read_csv(os.path.join(args.datadir, f'y_true({args.task}).csv'))['y']
    submission = pd.read_csv(os.path.join(args.datadir, f'submission({args.task}).csv'))

    # Set K-fold
    cv = KFold(n_splits=K, random_state=model.random_state, shuffle=False)

    # define prediction of validation set
    val_pred_df = y_train.copy()
    val_pred_df['y'] = 0

    for i, (train_idx, val_idx) in enumerate(cv.split(x_train, y_train)):
        # i-th print
        print('{0:2d}-Fold'.format(i))
        # split train and validation set 
        x_train_i, y_train_i = x_train.iloc[train_idx], y_train.iloc[train_idx]
        x_val_i, y_val_i = x_train.iloc[val_idx], y_train.iloc[val_idx]

        # build model
        model.build()

        # training
        _ = model.fit(X=x_train_i, y=y_train_i, validation_set=[x_val_i, y_val_i])

        # prediction 
        y_pred_val = model.predict(X=x_val_i)
        val_pred_df['y'].iloc[val_idx] = y_pred_val

        y_pred = model.predict(X=x_test)
        submission['y'] += y_pred / K

        # evaluation 
        val_results = evaluation_all(task=args.task, y_true=y_train, y_pred=val_pred_df['y'].values)
        test_results = evaluation_all(task=args.task, y_true=y_test, y_pred=submission['y'].values)

        # save results
        results_save(args=args,
                     model=model, 
                     validation_results=val_results, 
                     test_results=test_results,
                     submission=submission,
                     kfold_i=i)

