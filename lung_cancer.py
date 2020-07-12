#Accuracy:100%
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

dataset1=pd.read_csv("lung_cancer.csv")
dataset1=dataset1.drop_duplicates()
dataset1=dataset1.sample(frac=1)

X = dataset1.iloc[:, :10]
y = dataset1.iloc[:, 10]
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
X.iloc[:,2]= labelencoder_x.fit_transform(X.iloc[:,2])

from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=np.nan, strategy='mean')
X.iloc[:,[5,6]]=imputer.fit_transform(X.iloc[:,[5,6]])
X.drop(X.columns[[0,1]],axis=1,inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=3)

def bayesian_optimization(dataset1, function, parameters):
   X_train, y_train, X_test, y_test = dataset1
   n_iterations = 5
   gp_params = {"alpha": 0.07}
   BO = BayesianOptimization(function, parameters)
   BO.maximize(n_iter=n_iterations, **gp_params)
   return BO.max
def rfc_optimization(cv_splits):
    def function(n_estimators, max_depth, min_samples_split):
        return cross_val_score(
               RandomForestClassifier(
                   n_estimators=int(max(n_estimators,0)),                                                               
                   max_depth=int(max(max_depth,1)),
                   min_samples_split=int(max(min_samples_split,2)), 
                   n_jobs=-1, 
                   random_state=32,   
                   class_weight="balanced"),  
               X=X_train, 
               y=y_train, 
               cv=cv_splits,
               scoring="roc_auc",
               n_jobs=-1).mean()
    parameters = {"n_estimators": (10, 1000),
                  "max_depth": (1, 150),
                  "min_samples_split": (2, 10)}
    return function, parameters

def train(X_train, y_train, X_test, y_test, function, parameters):
    dataset = (X_train, y_train, X_test, y_test)
    cv_splits = 4
    best_solution = bayesian_optimization(dataset, function, parameters)      
    params = best_solution["params"]
    print(params)
    model = RandomForestClassifier (
             n_estimators=int(max(params["n_estimators"], 0)),
             max_depth=int(max(params["max_depth"], 1)),
             min_samples_split=int(max(params["min_samples_split"], 2)), 
             criterion="entropy",
             n_jobs=-1, 
             random_state=32,   
             class_weight="balanced")
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    from sklearn.metrics import precision_score
    print(precision_score(y_pred,y_test,average=None).mean())
    return model

function,parameters=rfc_optimization(4)
model=train(X_train, y_train, X_test, y_test, function, parameters)