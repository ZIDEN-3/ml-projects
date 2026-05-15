import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_Titanic = train_df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'])
X_Titanic['Age'] = X_Titanic['Age'].fillna(X_Titanic['Age'].median())

y_Titanic = train_df['Survived']


Xtrain, Xtest, ytrain, ytest = train_test_split(X_Titanic, y_Titanic, train_size=0.5)


le = LabelEncoder()
Xtrain['Sex'] = le.fit_transform(Xtrain['Sex'])
Xtest['Sex'] = le.transform(Xtest['Sex'])
print(X_Titanic.shape)
print(y_Titanic.shape)


model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)


print(accuracy_score(ytest, y_model))

