import pandas as pd

dataset1=pd.read_csv("Training.csv")
dataset1=dataset1.drop(dataset1.columns[[133]],axis=1)
dataset1=dataset1.drop_duplicates()
dataset1.reset_index(drop=True,inplace=True)

X = dataset1.iloc[:, :132]
y = dataset1.iloc[:, 132]

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 5)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy',random_state=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

"""from sklearn.metrics import precision_score
print(precision_score(y_pred,y_test,average=None).mean())"""
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

X_test2=   [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
y_pred2 = classifier.predict(X_test2)
y_pred2=list(labelencoder_y.inverse_transform(y_pred2))
print(*y_pred2)
