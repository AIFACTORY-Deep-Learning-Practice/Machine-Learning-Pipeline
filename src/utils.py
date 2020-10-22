import pickle 
import pandas as pd
import os
import sys

def results_save(args, model, validation_results, test_results, submission, kfold_i=None):
    # make logs folder
    savedir = args.logdir
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    # make task folder
    savedir = os.path.join(savedir, f'{args.task}')
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
        
    # update save directory of model, results, and commendline
    modeldir = os.path.join(savedir, f'{args.modelname}.pkl' if kfold_i==None else f'{args.modelname}_{kfold_i}.pkl')
    resultsdir = os.path.join(savedir, f'validation_results.pkl')
    testdir = os.path.join(savedir, 'test_results.pkl')
    subdir = os.path.join(savedir, 'prediction.csv')
    cmddir = os.path.join(savedir, 'commenline.txt')

    # save model, results, and commendline
    pickle.dump(model.model, open(modeldir,'wb'))
    pickle.dump(validation_results, open(resultsdir,'wb'))
    pickle.dump(test_results, open(testdir,'wb'))
    submission.to_csv(subdir, index=False)
    f = open(cmddir, 'w')
    f.write('python '+' '.join(sys.argv))


def results_comparison(task, datatype, logdir):
    savedir = os.path.join(logdir, task)
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


