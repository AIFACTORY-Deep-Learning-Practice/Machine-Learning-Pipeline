import pickle 
import pandas as pd
import os

def results_comparison(task):
    savedir = f'../logs/{task}'
    versions = os.listdir(savedir)
    
    results_lst = []
    for v in versions:
        # define results directory
        resultsdir = os.path.join(savedir, v, 'results.pkl') 

        # load results and add to results list
        results = pickle.load(open(resultsdir, 'rb'))
        results_lst.append(pd.Series(results, name=v))

    compare_df = pd.concat(results_lst, axis=1).T

    return compare_df