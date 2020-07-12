#Accuracy:77.95%
import pandas as pd
import numpy as np

dataset1=pd.read_csv("liver.csv")
dataset1=dataset1.drop_duplicates()

X = dataset1.iloc[:, :10]
y = dataset1.iloc[:, 10]

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
X.iloc[:,1]= labelencoder_x.fit_transform(X.iloc[:,1])
from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=np.nan, strategy='mean')
X.iloc[:,:]=imputer.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=3)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=4,min_samples_split=7,n_estimators=11,class_weight='balanced',random_state=32)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import precision_score
print(precision_score(y_pred,y_test,average=None).mean())
