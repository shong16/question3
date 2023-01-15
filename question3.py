import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib


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
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)

#Model 1: Decision Tree Model
from sklearn import tree
decision_tree_model=tree.DecisionTreeClassifier()
decision_tree_model.fit(X_train,Y_train)

#Model 2:
from sklearn.ensemble import RandomForestClassifier
random_forest_model=RandomForestClassifier()
random_forest_model.fit(X_train,Y_train)


#Performance Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print("Result 1: Decision Tree")
print("Score: %f" % decision_tree_model.score(X_test,Y_test))
predictions_decision_tree=decision_tree_model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(Y_test,predictions_decision_tree))
print("Classification Report: ")
print(classification_report(Y_test,predictions_decision_tree, target_names=['High', 'Low', 'Moderate']))

plt.subplot(211)
feature_label=['Age', 'Class Duration', 'Gender', 'Edu. Level',
       'Inst. Type', 'IT Student', 'Location', 'Load-shedding',
       'Fin. Condition', 'Internet Type', 'Network Type',
       'Self Lms', 'Device']
plt.bar(feature_label, decision_tree_model.feature_importances_)
plt.title('Feature Importance: Decision Tree')




print("Result 2: Random Forest")
print("Score: %f" % random_forest_model.score(X_test,Y_test))
predictions_random_forest=random_forest_model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(Y_test,predictions_random_forest))
print("Classification Report: ")
print(classification_report(Y_test,predictions_random_forest, target_names=['High', 'Low', 'Moderate']))

plt.subplot(212)
plt.bar(feature_label, random_forest_model.feature_importances_)
plt.title('Feature Importance: Random Forest')
plt.show()

#histogram of columns of interest

#matplotlib.rcParams["figure.figsize"] = (5,5)
#plt.hist(data.Age,rwidth=0.8)
#plt.xlabel("Age")
#plt.ylabel("Count")
#plt.title('Age Histogram')
#plt.show()





