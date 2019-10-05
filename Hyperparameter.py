# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 04:22:28 2019

@author: Dilip
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#import the data set
chun_dataframe=pd.read_csv('churn_modelling.csv')

#analysing data
sns.countplot(x="Exited", hue="Gender", data=chun_dataframe)
sns.countplot(x="Exited", hue="Geography", data=chun_dataframe)
chun_dataframe["Age"].plot.hist()
chun_dataframe["HasCrCard"].plot.hist()

#data wrangling
chun_dataframe.isnull()
chun_dataframe.isnull().sum()
sns.heatmap(chun_dataframe.isnull(), yticklabels=False,cmap="viridis" )


x= chun_dataframe.iloc[:,3:13]
y= chun_dataframe.iloc[:,13]


#creating dummie variables
Sex=pd.get_dummies(x ["Gender"], drop_first=True)
geography=pd.get_dummies(x["Geography"], drop_first=True)


#concate the data frame
x=pd.concat([x, Sex, geography], axis=1)

#drop unwanted coloumns
x.drop(["Gender", "Geography"], axis=1, inplace=True)

#spliting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#importing the keras libraries
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid

#perform hyperparameter optimization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def create_model(layers, activation):
    model=Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes, input_dim=x_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
          
            model.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
            
            model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            return model
    

model = KerasClassifier(build_fn=create_model, verbose= 0)

layers=[[20], [40, 20], [45, 30, 15]]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size= [128, 256], epochs= [30])
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)


grid_result = grid.fit(x_train, y_train)

#print model result
print(grid_result.best_score_,grid_result.best_params_)

#predicting the test result
y_pred=grid.predict(x_test)
y_pred=(y_pred > 0.5)

#calculate the accuracy
from sklearn.metrics import confusion_matrix, accuracy_score 
cm = confusion_matrix(y_test, y_pred)
print('\n')
ac = accuracy_score(y_test, y_pred)



