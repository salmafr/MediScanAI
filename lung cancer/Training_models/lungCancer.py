import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix


df = pd.read_csv('H:\YaneCode\PP\Datasets\cancer patient data sets.csv')

df.drop(columns=['index'], inplace=True)
df.drop(columns=['Patient Id'], inplace=True)

df.drop_duplicates()

df['Level']=df['Level'].map({'High':1,'Low':0, 'Medium':2})
print (df['Level'])

X = df.iloc[:,:-1]
y = df['Level']
X_train, X_test, y_train, y_test =train_test_split(X ,y ,test_size=0.3)


#scale=MinMaxScaler()
#X_train=pd.DataFrame(scale.fit_transform(X_train),columns=X_train.columns)
#X_train
#X_test = pd.DataFrame(scale.fit_transform(X_test),columns=X_test.columns)
#X_test

print(df)

def models(X_train, y_train):

  #Decision Tree
  from sklearn import tree
  model1 = tree.DecisionTreeClassifier()
  model1 = model1.fit(X_train,y_train)

  #Random Forest
  from sklearn.ensemble import RandomForestClassifier
  model2 = RandomForestClassifier()
  model2 = model2.fit(X_train,np.ravel(y_train))

  #Naive Bayes
  from sklearn.naive_bayes import GaussianNB
  gnb = GaussianNB()
  gnb=gnb.fit(X_train,np.ravel(y_train))

  #Print the models accuracy on the training data
  print('[0]Decision tree training accuracy: ', model1.score(X_train, y_train))
  print('[1]Random Forest training accuracy: ', model2.score(X_train, y_train))
  print('[2]Naive Bayes training accuracy: ', gnb.score(X_train, y_train))

  return model1, model2, gnb

model= models(X_train, y_train)

for i in range(len(model)):
  print('model', i)
  cm = confusion_matrix(y_test, model[i].predict(X_test))

  TP=cm[0][0]
  TN=cm[1][1]
  FP= cm[0][1]
  FN= cm[1][0]

  print (cm)
  print('Testing Accuracy = ', (TP+TN)/(TP+TN+FN+FP))
  print()


import pickle 
file_path = 'H:\YaneCode\PP\lung cancer\decisiontree.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(model[0], file)

pred= model[0].predict(X_test)
print('prediction=', pred)
print()
print('y_test=', y_test.values)

import os

file_path = os.path.join(os.getcwd(), 'decisiontree.pkl')
print("Pickle file is located at:", file_path)