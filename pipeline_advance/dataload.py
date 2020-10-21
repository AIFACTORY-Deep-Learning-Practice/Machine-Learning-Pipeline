import pandas as pd 
import os 


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