#Accuracy: 95%-100%
import pandas as pd
import numpy as np

dataset1=pd.read_csv("Common_diseases.csv")
dataset1=dataset1.drop(dataset1.columns[[133]],axis=1)
dataset1=dataset1.drop_duplicates()
dataset1.reset_index(drop=True,inplace=True)
dataset1.sort_values('prognosis',inplace=True)
X = dataset1.iloc[:, :132]
y = dataset1.iloc[:, 132]

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import precision_score
print(precision_score(y_pred,y_test,average=None).mean())
