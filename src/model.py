from sklearn import linear_model, tree, ensemble, svm, neighbors
from sklearn import metrics

import pickle
import os 


class SklearnModels:
    def __init__(self, task, modelname):
        self.task = task
        self.modelname = modelname

    def build(self, random_state, **params):
        if self.task == 'regression':
            # regression
            if self.modelname == 'OLS':
                # Ordinary Least Square
                self.model = linear_model.LinearRegression(**params)
            elif self.modelname == 'Ridge':
                # Ridge
                self.model = linear_model.Ridge(random_state=random_state, **params)
            elif self.modelname == 'Lasso':
                # Lasso 
                self.model = linear_model.Lasso(random_state=random_state, **params)
            elif self.modelname == 'ElasticNet':
                # Elastic-Net
                self.model = linear_model.ElasticNet(random_state=random_state, **params)
            elif self.modelname == 'DT':
                # Decision Tree
                self.model = tree.DecisionTreeRegressor(random_state=random_state, **params)
            elif self.modelname == 'RF':
                # Random Forest
                self.model = ensemble.RandomForestRegressor(random_state=random_state, **params)
            elif self.modelname == 'ADA':
                # Adaboost
                self.model = ensemble.AdaBoostRegressor(random_state=random_state, **params)
            elif self.modelname =='GT':
                # Gradient Tree Boosting
                self.model = ensemble.GradientBoostingRegressor(random_state=random_state, **params)
            elif self.modelname == 'SVM':
                # Support Vector Machine
                self.model = svm.SVR(**params)
            elif self.modelname == 'KNN':
                # K-Nearest Neighbors
                self.model = neighbors.KNeighborsRegressor(**params)

        elif self.task == 'classification':
            # classification
            if self.modelname == 'OLS':
                # Ordinary Least Square
                self.model = linear_model.LogisticRegression(**params)
            elif self.modelname == 'Ridge':
                # Ridge
                self.model = linear_model.RidgeClassifier(random_state=random_state, **params)
            elif self.modelname == 'DT':
                # Decision Tree
                self.model = tree.DecisionTreeClassifier(random_state=random_state, **params)
            elif self.modelname == 'RF':
                # Random Forest
                self.model = ensemble.RandomForestClassifier(random_state=random_state, **params)
            elif self.modelname == 'ADA':
                # Adaboost
                self.model = ensemble.AdaBoostClassifier(random_state=random_state, **params)
            elif self.modelname == 'GT':
                # Gradient Tree Boosting
                self.model = ensemble.GradientBoostingClassifier(random_state=random_state, **params)
            elif self.modelname == 'SVM':
                # Support Vector Machine 
                self.model = svm.SVC(probability=True, **params)
            elif self.modelname == 'KNN':
                # K-Nearest Neighbors
                self.model = neighbors.KNeighborsClassifier(**params)
 
    def fit(self, X, y):
        self.model.fit(X, y)

    def eval(self, X, y):
        y_pred = self.model.predict(X)
        
        if self.task == 'regression':
            # Mean Squared Error, MSE 
            mse = metrics.mean_squared_error(y, y_pred)
            # Mean Absolute Error, MAE
            mae = metrics.mean_absolute_error(y, y_pred)

            self.results = {'MSE':mse, 'MAE':mae}

        elif self.task == 'classification':
            if self.modelname == 'Ridge':
                y_prob = y_pred
            else:
                y_prob = self.model.predict_proba(X)[:,1]

            # Accuracy = (TP + FP) / (TP + TN + FP + FN)
            acc = metrics.accuracy_score(y, y_pred)
            # Precision = TP / (TP + FP)
            precision = metrics.precision_score(y, y_pred)
            # Recall = TP / (TP + FN)
            recall = metrics.recall_score(y, y_pred)
            # F1 score = 2 * (Precision * Recall) / (Predicion + Recall)
            f1_score = metrics.f1_score(y, y_pred)
            # Area Under Curve of Receive Operating Characteristic
            auc = metrics.roc_auc_score(y, y_prob)

            self.results = {'Acc':acc, 'Precision':precision, 'Recall':recall, 'F1 score':f1_score, 'AUC':auc}

        # results save
        self.save()

    def save(self):
        # make save folder
        savedir = '../logs'
        if not os.path.isdir(savedir):
            os.mkdir(savedir)

        # make task folder
        savedir = os.path.join(savedir, f'{self.task}')
        if not os.path.isdir(savedir):
            os.mkdir(savedir)

        # check save version
        version = len(os.listdir(savedir))

        # update save directory
        savedir = os.path.join(savedir, f'version{version}')
        os.mkdir(savedir)

        # update save directory of model and results 
        modeldir = os.path.join(savedir, f'{self.modelname}.pkl')
        resultsdir = os.path.join(savedir, 'results.pkl')

        # save model and results
        pickle.dump(self.model, open(modeldir,'wb'))
        pickle.dump(self.results, open(resultsdir,'wb'))


    def load(self, modeldir):
        self.model = pickle.load(open(modeldir, 'rb'))