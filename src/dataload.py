from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

import pandas as pd 
import os 

def data_generator(task, n_samples, n_features, noise, test_size, random_state):
    if task == 'classification':
        X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=random_state)
    elif task == 'regression':
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # define train and test set 

    # train set
    train = pd.DataFrame(x_train, columns=[f'f{i}' for i in range(x_train.shape[1])])
    train = train.reset_index().rename(columns={'index':'id'})
    train['y'] = y_train
    
    # test set 
    test = pd.DataFrame(x_test, columns=[f'f{i}' for i in range(x_train.shape[1])])
    test = test.reset_index().rename(columns={'index':'id'})
    test['id'] = range(7000, 10000)

    # define y of test set 
    y_test = pd.DataFrame(y_test,columns=['y'])
    y_test = y_test.reset_index().rename(columns={'index':'id'})
    y_test['id'] = range(7000, 10000)

    # define prediction set 
    y_pred = y_test.copy()
    y_pred['y'] = 0

    # save to csv
    train.to_csv(f'../data/train({task}).csv', index=False)
    test.to_csv(f'../data/test({task}).csv', index=False)
    y_test.to_csv(f'../data/y_true({task}).csv', index=False)
    y_pred.to_csv(f'../data/submission({task}).csv', index=False)

def dataloader(task, datadir):
    train = pd.read_csv(os.path.join(datadir, f'train({task}).csv'))
    test = pd.read_csv(os.path.join(datadir, f'test({task}).csv'))

    return train, test

def preprocessing(data, data_type):
    '''
    (preprocessing)
    '''

    drop_features = ['id']

    if data_type == 'train':
        drop_features.append('y')
        X = data.drop(drop_features, axis=1)
        y = data[['y']]

        return X, y

    elif data_type == 'test':
        X = data.drop(drop_features, axis=1)
        
        return X