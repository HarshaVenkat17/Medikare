#Accuracy: 88.75%-97.5%
import pandas as pd
import numpy as np
import lightgbm as lgb

dataset1=pd.read_csv("kidney.csv")
l1= ['age','bp','bgr','bu','sod','pcv','wc']
l2=['sg','al','su','rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
l3=['sc','pot','hemo','rc']

from sklearn.impute import SimpleImputer
imputer1= SimpleImputer(missing_values=np.nan, strategy='mean')
imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
for i in l1:
    dataset1[i] = imputer1.fit_transform(dataset1[[i]]).astype('int')
for i in l3:
    dataset1[i] = imputer1.fit_transform(dataset1[[i]])
for i in l2:
    dataset1[i] = imputer2.fit_transform(dataset1[[i]])
dataset1=dataset1.sample(frac=1)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(categories=[[1.005, 1.01, 1.015, 1.02, 1.025],[0,1,2,3,4,5],[0,1,2,3,4,5],['normal','abnormal'],['normal','abnormal'],['notpresent','present'],['notpresent','present'],['no','yes'],['no','yes'],['no','yes'],['poor','good'],['no','yes'],['no','yes'],['notckd','ckd']]),[3,4,5,6,7,8,9,19,20,21,22,23,24,25])],remainder='passthrough')#,("gluc",OneHotEncoder(),[0])
dataset1=ct.fit_transform(dataset1)

X = dataset1[:, :]
y = dataset1[:, 38]
X=np.delete(X,[0,5,11,17,19,21,23,25,27,29,31,33,35,37,38,39],1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'early_stopping_rounds': 80,
        'feature_fraction_seed': 3,
        'lambda_rank':6.0,
        'tree_learner':'voting',
        'min_sum_hessian_in_leaf':4.847881380318586,
        'learning_rate':0.098804745206693,
        'num_threads':4,
        'device':'cpu',
        'metric' : 'auc',
        'bagging_fraction':0.4135816085580124,
        'feature_fraction':0.6771287146747047,
        'lambda_l1': 0.3120012518081733,
        'lambda_l2':  1.6594846881756342,
        'max_depth':18,
        'min_child_weight': 6.179812922752215,
         'min_split_gain':0.012279857339053388,
         'num_leaves': 15,
        'verbose':200,
        'min_data_in_leaf':47,
        'bagging_freq': 5,
        'bagging_seed': 3,
        'subsample_for_bin': 2000,
        'min_child_samples': 40,
        'max_bin':81,
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

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
