from sklearn import linear_model, tree, ensemble, svm, neighbors
from sklearn.model_selection import train_test_split

from utils import results_bar
from evaluate import evaluation_all

import pickle
import os 


class SklearnModels:
    def __init__(self, task, modelname, random_state):
        self.task = task
        self.modelname = modelname
        self.random_state = random_state

    def build(self, **params):
        if self.task == 'regression':
            # regression
            if self.modelname == 'OLS':
                # Ordinary Least Square
                self.model = linear_model.LinearRegression(**params)
            elif self.modelname == 'Ridge':
                # Ridge
                self.model = linear_model.Ridge(random_state=self.random_state, **params)
            elif self.modelname == 'Lasso':
                # Lasso 
                self.model = linear_model.Lasso(random_state=self.random_state, **params)
            elif self.modelname == 'ElasticNet':
                # Elastic-Net
                self.model = linear_model.ElasticNet(random_state=self.random_state, **params)
            elif self.modelname == 'DT':
                # Decision Tree
                self.model = tree.DecisionTreeRegressor(random_state=self.random_state, **params)
            elif self.modelname == 'RF':
                # Random Forest
                self.model = ensemble.RandomForestRegressor(random_state=self.random_state, **params)
            elif self.modelname == 'ADA':
                # Adaboost
                self.model = ensemble.AdaBoostRegressor(random_state=self.random_state, **params)
            elif self.modelname =='GT':
                # Gradient Tree Boosting
                self.model = ensemble.GradientBoostingRegressor(random_state=self.random_state, **params)
            elif self.modelname == 'SVM':
                # Support Vector Machine
                self.model = svm.SVR(**params)
            elif self.modelname == 'KNN':
                # K-Nearest Neighbors
                self.model = neighbors.KNeighborsRegressor(**params)

        elif self.task == 'classification':
            # classification
            if self.modelname == 'Logistic':
                # Ordinary Least Square
                self.model = linear_model.LogisticRegression(**params)
            elif self.modelname == 'Ridge':
                # Ridge
                self.model = linear_model.RidgeClassifier(random_state=self.random_state, **params)
            elif self.modelname == 'DT':
                # Decision Tree
                self.model = tree.DecisionTreeClassifier(random_state=self.random_state, **params)
            elif self.modelname == 'RF':
                # Random Forest
                self.model = ensemble.RandomForestClassifier(random_state=self.random_state, **params)
            elif self.modelname == 'ADA':
                # Adaboost
                self.model = ensemble.AdaBoostClassifier(random_state=self.random_state, **params)
            elif self.modelname == 'GT':
                # Gradient Tree Boosting
                self.model = ensemble.GradientBoostingClassifier(random_state=self.random_state, **params)
            elif self.modelname == 'SVM':
                # Support Vector Machine 
                self.model = svm.SVC(probability=True, random_state=self.random_state, **params)
            elif self.modelname == 'KNN':
                # K-Nearest Neighbors
                self.model = neighbors.KNeighborsClassifier(**params)
 

    def fit(self, X, y, validation_set:list=None, validation_size=None, threshold=0.5):

        if validation_size:
            # split train and validation set
            x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, random_state=self.random_state)
        elif validation_set:
            x_train, y_train = X, y 
            x_val, y_val = validation_set

        # model training
        self.model.fit(X=x_train, y=y_train)
        # evaluate train and validation set
        train_results = self.eval(X=x_train, y=y_train, threshold=threshold)
        val_results = self.eval(X=x_val, y=y_val, threshold=threshold)

        # print results
        results_bar(results=train_results, description='train')
        results_bar(results=val_results, description='validation')

        return val_results


    def eval(self, X, y, threshold=0.5):
        # prediction 
        y_pred = self.predict(X)
            
        # eval
        results = evaluation_all(task=self.task, y_true=y, y_pred=y_pred, threshold=threshold)
            
        return results

    def predict(self, X):
        if self.task == 'regression':
            y_pred = self.model.predict(X)
            y_pred = y_pred[:,0] if len(y_pred.shape) == 2 else y_pred 
            
        elif self.task == 'classification':
            if self.modelname == 'Ridge':
                y_pred = self.model.predict(X)
            else:
                y_pred = self.model.predict_proba(X)[:,1]

        return y_pred

    def load(self, modeldir):
        self.model = pickle.load(open(modeldir, 'rb'))