from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

import pickle 
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
    

def results_save(task, logdir, model, modelname, validation_results, test_results, submission, kfold_i=None):
    # make logs folder
    savedir = logdir
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    # make task folder
    savedir = os.path.join(savedir, f'{task}')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    # check save version
    version = len(os.listdir(savedir))

    # make save folder
    if (kfold_i == None) | (kfold_i == 0):
        savedir = os.path.join(savedir, f'version{version}')
        os.mkdir(savedir)
    else:
        version -= 1
        savedir = os.path.join(savedir, f'version{version}')
        
    # update save directory of model and results 
    modeldir = os.path.join(savedir, f'{modelname}.pkl' if kfold_i==None else f'{modelname}_{kfold_i}.pkl')
    resultsdir = os.path.join(savedir, f'validation_results.pkl')
    testdir = os.path.join(savedir, 'test_results.pkl')
    subdir = os.path.join(savedir, 'prediction.csv')

    # save model and results
    pickle.dump(model.model, open(modeldir,'wb'))
    pickle.dump(validation_results, open(resultsdir,'wb'))
    pickle.dump(test_results, open(testdir,'wb'))
    submission.to_csv(subdir, index=False)


def results_comparison(task, datatype):
    savedir = f'../logs/{task}'
    versions = os.listdir(savedir)
    
    results_lst = []
    for v in versions:
        # define results directory
        resultsdir = os.path.join(savedir, v, f'{datatype}_results.pkl') 

        # load results and add to results list
        results = pickle.load(open(resultsdir, 'rb'))
        results_lst.append(pd.Series(results, name=v))

    compare_df = pd.concat(results_lst, axis=1).T

    return compare_df

def results_bar(results, description):
    r = ''
    for k, v in results.items():
        r += '{0:}: {1:.4f} '.format(k,v)    
    print('[Results]{0:10s}: '.format(description.upper()),r)


