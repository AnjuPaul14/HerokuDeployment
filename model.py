import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('./iris.csv')

print(df.head())

from  sklearn.preprocessing  import LabelEncoder
le = LabelEncoder()

df['variety'] = le.fit_transform(df['variety'])
print(df.head())

# Import the feature values:
x = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values


# Split the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)


print("x_train : ")
print(X_train)

print("y_train : ")
print(X_test)

log_reg = LogisticRegression(max_iter = 1000)
log_reg.fit(X_train, y_train)

prediction = log_reg.predict(X_test)
print(prediction)

import pickle
pickle.dump(log_reg, open('model.pickle', 'wb'))

