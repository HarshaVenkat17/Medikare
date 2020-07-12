#Accuracy:73.91
import pandas as pd
import numpy as np
import lightgbm as lgb

dataset1=pd.read_csv("cardiovascular.csv",sep=";")
dataset1=np.array(dataset1.drop(['id'],axis=1))
X = dataset1[:, :11]
y = dataset1[:, 11]
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(categories=[[1,2,3],[1,2,3]]),[6,7])],remainder='passthrough')#,("gluc",OneHotEncoder(),[0])
X=ct.fit_transform(X)
X=np.delete(X,[0,3],1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'early_stopping_rounds': 80, 
         'boost_from_average':True,
        'feature_fraction_seed': 1,
        'lambda_rank':3.0,
        'tree_learner':'voting',
        'min_sum_hessian_in_leaf':20,
        'learning_rate': 0.082,
        'num_threads':4,
        'device':'cpu',
         'metric' : 'auc',                   
        'bagging_fraction': 0.8833876361825719,
        'feature_fraction':  0.807891724509068,
        'lambda_l1':0.7544241204224678,
        'lambda_l2':0.6763031288535707,
        'min_child_weight':5.1319716479846,
         'min_split_gain': 0.09145973736604322,
        'bagging_freq': 5,
        'bagging_seed': 15,
        'subsample_for_bin': 2000,
        'min_child_samples': 20,
        'num_leaves': 18,
        'min_data_in_leaf':60,
        'max_depth':15,
        'max_bin':80  
    }
lgb_train=lgb.Dataset(X_train,y_train)
lgb_val=lgb.Dataset(X_test,y_test)
lgbm_model = lgb.train(params, train_set = lgb_train, valid_sets = lgb_val, verbose_eval=5)

y_pred = lgbm_model.predict(X_test)
c_pred=y_pred
y_pred[y_pred<0.5] = 0
y_pred[y_pred>0.5] = 1

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)