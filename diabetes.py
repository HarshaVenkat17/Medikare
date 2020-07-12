#1 Accuracy:82.03
import pandas as pd
import numpy as np
import lightgbm as lgb

dataset1=pd.read_csv("diabetes.csv")
X = dataset1.iloc[:, :8]
y = dataset1.iloc[:,8]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'early_stopping_rounds': 40, 
        'feature_fraction_seed': 1,
        'lambda_rank':3.0,
        'tree_learner':'voting',
        'min_sum_hessian_in_leaf':0,
        'learning_rate': 0.072,
        'num_threads':4,
        'device':'cpu',
        'metric' : 'auc',
        'bagging_fraction':0.1,
        'feature_fraction':1,
        'lambda_l1':  0, 
        'lambda_l2':  3,
        'max_depth':30,
        'min_child_weight':6,
        'min_split_gain':0.0010000000824852966,
        'num_leaves': 10,
        'min_data_in_leaf':1,
        'bagging_freq': 5,
        'bagging_seed': 6,
        'subsample_for_bin': 1000,
        'min_child_samples': 40,
        'max_bin':81
       }
lgb_train=lgb.Dataset(X_train,y_train)
lgb_val=lgb.Dataset(X_test,y_test)
lgbm_model = lgb.train(params, train_set = lgb_train, valid_sets = lgb_val, verbose_eval=4)

y_pred = lgbm_model.predict(X_test)
c_pred=y_pred
y_pred[y_pred<0.5] = 0
y_pred[y_pred>0.5] = 1

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import precision_score
print(precision_score(y_pred,y_test,average=None).mean())