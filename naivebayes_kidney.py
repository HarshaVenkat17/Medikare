#Accuracy:98%-100%
import pandas as pd
import numpy as np

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
print("Accuracy: "+str(accuracy_score(y_test,y_pred)*100))
