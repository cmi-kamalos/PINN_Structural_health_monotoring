<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 23:12:24 2022

@author: choueb

installations of the following librairies
    -pandas
    -numpy
    -tensorflow
    -keras
    -scikit(sklearn)
"""

import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# suppress tensorflow warnings (must be called before importing tensorflow)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



#reading our dataset
df=pd.read_csv("mat.CSV")

#making a copy of the dataset to work with
beamer_df=df.copy()

#splitting the dataset in 1/3,2/3 test and training data
beamer_train,beamer_test=train_test_split(beamer_df, test_size=0.33)

#beamers labels are Wn columns
beamer_labels=beamer_train

beamer_labels_columns_names=beamer_labels.iloc[:,0:8].columns.values.tolist()

#beamer_features our features or the proprieties of a beamer, the 8 first columns
beamer_features=pd.DataFrame([beamer_labels.pop(x) for x in beamer_labels_columns_names]).T

print(beamer_features.head())
print(beamer_labels.head())


#Converting them into numpy array
beamer_feature=np.array(beamer_features)
beamer_label=np.array(beamer_labels)

#define model for making prediction(Ann algorithm)

def get_model(n_inputs, n_outputs):
    model = Sequential()
    #using just 2 hidden layers, the hidden layers are hyperparameters to be adjusted
    model.add(Dense(2, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
    model.compile(loss='mae', optimizer='adam')
    """
    We implemented the sequential algorithm from keras for Ann
    model.add (1st): the input parameters with the hidden layers associated with, an activation layer,
    kernel initialiser and taking an input as parameter for training
    
    model.add (2nd): taking the the output of the first layers as input. It's outputs as a parameter
    mixed initialiser function
    
    At the end, when compile it's compute with mae(mean absolute error) with a gradient descent
    stochastic function known as adam optimizer

    """
    return model




input_model,output_model=beamer_features.shape[1],beamer_labels.shape[1]

model=get_model(input_model,output_model)


#fitting the model

model.fit(beamer_feature,beamer_label,epochs=100)

#evaluating the model
model.evaluate(x=beamer_feature,y=beamer_label)


test=np.array(beamer_test.iloc[2:3,0:8])

"""
Testing the model. For that we take an entry with from column Length to Dump in the dataset
and passed to the model.
"""
testBeforePrediction=np.array(beamer_test.iloc[2:3,8:])

print("result of a raw...........\\\\\................ Wn before prediction")
print(testBeforePrediction)
print("Making the prediction of the raw .............\\\.......... Wn entry")
print(model.predict(test))





=======
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 23:12:24 2022

@author: choueb

installations of the following librairies
    -pandas
    -numpy
    -tensorflow
    -keras
    -scikit(sklearn)
"""

import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# suppress tensorflow warnings (must be called before importing tensorflow)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



#reading our dataset
df=pd.read_csv("mat.CSV")

#making a copy of the dataset to work with
beamer_df=df.copy()

#splitting the dataset in 1/3,2/3 test and training data
beamer_train,beamer_test=train_test_split(beamer_df, test_size=0.33)

#◘beamers labels are Wn columns
beamer_labels=beamer_train

beamer_labels_columns_names=beamer_labels.iloc[:,0:8].columns.values.tolist()

#beamer_features our features or the proprieties of a beamer, the 8 first columns
beamer_features=pd.DataFrame([beamer_labels.pop(x) for x in beamer_labels_columns_names]).T

print(beamer_features.head())
print(beamer_labels.head())


#Converting them into numpy array
beamer_feature=np.array(beamer_features)
beamer_label=np.array(beamer_labels)

#define model for making prediction(Ann algorithm)

def get_model(n_inputs, n_outputs):
    model = Sequential()
    #using just 2 hidden layers, the hidden layers are hyperparameters to be adjusted
    model.add(Dense(2, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
    model.compile(loss='mae', optimizer='adam')
    """
    We implemented the sequential algorithm from keras for Ann
    model.add (1st): the input parameters with the hidden layers associated with, an activation layer,
    kernel initialiser and taking an input as parameter for training
    
    model.add (2nd): taking the the output of the first layers as input. It's outputs as a parameter
    mixed initialiser function
    
    At the end, when compile it's compute with mae(mean absolute error) with a gradient descent
    stochastic function known as adam optimizer

    """
    return model




input_model,output_model=beamer_features.shape[1],beamer_labels.shape[1]

model=get_model(input_model,output_model)


#fitting the model

model.fit(beamer_feature,beamer_label,epochs=100)

#evaluating the model
model.evaluate(x=beamer_feature,y=beamer_label)


test=np.array(beamer_test.iloc[2:3,0:8])

"""
Testing the model. For that we take an entry with from column Length to Dump in the dataset
and passed to the model.
"""
testBeforePrediction=np.array(beamer_test.iloc[2:3,8:])

print("result of a raw...........\\\\\................ Wn before prediction")
print(testBeforePrediction)
print("Making the prediction of the raw .............\\\.......... Wn entry")
print(model.predict(test))





>>>>>>> 3c0a3c5 (ann beamer)
