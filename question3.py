import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam


#Read data
data=pd.read_csv("data.csv")


#Label encoding for categorical data
from sklearn.preprocessing import LabelEncoder

def label_encoding(data):
    column_list = list(data.columns.values)
    for cur_col in column_list:
        if cur_col!='Age' and cur_col!='Class Duration':
            le=LabelEncoder()
            data[cur_col+'_n']=le.fit_transform(data[cur_col])
            data=data.drop(cur_col,axis='columns')
    return data

def scale_data(data):
    data['Age'] = data['Age'] / (0.5 * max(data['Age']))
    data['Class Duration'] = data['Class Duration'] / (0.5 * max(data['Class Duration']))
    return data

#Test code: convert categorical data into numerical ones
data=label_encoding(data)


#Function for converting ranges into numerical values
def convert_range_into_num(range: str)->float:
    if range=='0':
        return float(range)
    else:
        tokens = range.split('-')
        return (float(tokens[0])+float(tokens[1]))/2


#Test Code: Convert Age and Class Duration to numerical values
data['Age']=data['Age'].apply(convert_range_into_num)
data['Class Duration']=data['Class Duration'].apply(convert_range_into_num)


# Construct test and training sets
X=data.drop('Adaptivity Level_n',axis='columns')
Y=data['Adaptivity Level_n']
X_scaled=scale_data(X) #Scale the training dataset



#Model Selection
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

print("Result of Cross Validation: Decision Tree / Random Forest /Support Vector / K Neighbors Classifier")
print(cross_val_score(tree.DecisionTreeClassifier(),X,Y))
print(cross_val_score(RandomForestClassifier(),X,Y))
print(cross_val_score(SVC(),X,Y))
print(cross_val_score(KNeighborsClassifier(n_neighbors=3),X,Y))

print("Result of Cross Validation: Decision Tree / Random Forest /Support Vector / K Neighbors Classifier (Scaled)")
print(cross_val_score(tree.DecisionTreeClassifier(),X_scaled,Y))
print(cross_val_score(RandomForestClassifier(),X_scaled,Y))
print(cross_val_score(SVC(),X_scaled,Y))
print(cross_val_score(KNeighborsClassifier(n_neighbors=3),X_scaled,Y))


#Splitting training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)
X_train_scaled=scale_data(X_train)
X_test_scaled=scale_data(X_test)


#Model training:
random_forest_model=RandomForestClassifier()
random_forest_model.fit(X_train,Y_train)

#Performance Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


print("Result: Random Forest")
print("Score: %f" % random_forest_model.score(X_test,Y_test))
predictions_random_forest=random_forest_model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(Y_test,predictions_random_forest))
print("Classification Report: ")
print(classification_report(Y_test,predictions_random_forest, target_names=['High', 'Low', 'Moderate']))


plt.figure(1)
feature_label=['Age', 'Class Duration', 'Gender', 'Edu. Level',
       'Inst. Type', 'IT Student', 'Location', 'Load-shedding',
       'Fin. Condition', 'Internet Type', 'Network Type',
       'Self Lms', 'Device']
plt.bar(feature_label, random_forest_model.feature_importances_)
plt.title('Feature Importance: Random Forest')
#plt.show()


#Hyper Parameter Search
Y_train_nn=to_categorical(Y_train)
Y_test_nn=to_categorical(Y_test)



from tensorflow.keras.regularizers import L2

def define_model(hp):
    nn_model = Sequential()
    hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])
    hp_layer_1 = hp.Int('layer_1', min_value=1, max_value=1000)
    hp_layer_2 = hp.Int('layer_2', min_value=1, max_value=500)
    hp_layer_3 = hp.Int('layer_3', min_value=1, max_value=250)
    hp_layer_4 = hp.Int('layer_4', min_value=1, max_value=125)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    nn_model.add(Dense(units=hp_layer_1,input_dim=13, activation=hp_activation))
    nn_model.add(Dropout(0.05))
    nn_model.add(BatchNormalization())
    nn_model.add(Dense(units=hp_layer_2, activation=hp_activation))
    nn_model.add(Dropout(0.05))
    nn_model.add(BatchNormalization())
    nn_model.add(Dense(units=hp_layer_3, activation=hp_activation))
    nn_model.add(Dropout(0.05))
    nn_model.add(BatchNormalization())
    nn_model.add(Dense(units=hp_layer_4, activation=hp_activation))
    nn_model.add(Dropout(0.05))
    nn_model.add(BatchNormalization())
    nn_model.add(Dense(3, activation='softmax'))
    nn_model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return nn_model

import keras_tuner as kt
tuner=kt.Hyperband(define_model,objective='val_accuracy',max_epochs=50,factor=3, directory='dir_four',project_name='xfour')
tuner.search(X_train_scaled, Y_train_nn, epochs = 50, validation_split=0.2)
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
#Hyper Parameter
print("The optimal hyperparameters are as follows: ")
print("Activation Function: ", best_hps.get('activation'))
print("Number of neurons for layer 1: ", best_hps.get('layer_1'))
print("Number of neurons for layer 2: ", best_hps.get('layer_2'))
print("Number of neurons for layer 3: ", best_hps.get('layer_3'))
print("Number of neurons for layer 4: ", best_hps.get('layer_4'))
print("Learning rate for Adam: ", best_hps.get('learning_rate'))


#Neural network training
nn_model=Sequential()
nn_model.add(BatchNormalization())
nn_model.add(Dense(542,input_dim=13,activation='relu'))
nn_model.add(Dropout(0.05))
nn_model.add(BatchNormalization())
nn_model.add(Dense(149, activation='relu'))
nn_model.add(Dropout(0.05))
nn_model.add(BatchNormalization())
nn_model.add(Dense(212, activation='relu'))
nn_model.add(BatchNormalization())
nn_model.add(Dense(86, activation='relu'))
nn_model.add(BatchNormalization())
nn_model.add(Dense(3, activation='softmax'))
nn_model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
nn_model.fit(X_train_scaled,Y_train_nn,epochs=1,batch_size=64)
score=nn_model.evaluate(X_test_scaled,Y_test_nn)
print("Summary of the neural network model...")
print(nn_model.summary())

#Performance Comparsion
print("Result 1: Random Forest")
print("Score: %f" % random_forest_model.score(X_test,Y_test))

print("Result 2: Neural Network")
print("Score: %f" % score[1])

#save model
import pickle
with open('neural_network_model_crude.pickle','wb') as f:
    pickle.dump(nn_model,f)

import json
columns={
    'data_columns' : [col.lower() for col in feature_label]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))

print("TEST")
adaptivity_type=['High', 'Low', 'Moderate']
ex=X.values[:1]
pred_example=nn_model.predict(X.values[:1])
idx=np.argmax(pred_example)
print(adaptivity_type[idx])