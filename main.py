import numpy as np # linear algebra
import pandas as pd

data = pd.read_csv('creditcard.csv')
data.head()
data.isnull().sum()
data.describe()
print('Valid transaction',len(data[data['Class']==0]))
print('fraud transaction',len(data[data['Class']==1]))
y= data['Class']
x= data.drop(columns=['Class'],axis=1)
#splitting the data into train test 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.3,random_state=0)
# fitting randomforest model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
#model_1
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20,criterion='entropy', random_state=0,max_depth=10)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import  classification_report, confusion_matrix
print('Classifcation report:\n', classification_report(y_test, y_pred))
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)
#model_2
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=30,criterion='entropy', random_state=0,max_depth=10)
classifier.fit(x_train,y_train)
y_pred_2 = classifier.predict(x_test)
from sklearn.metrics import  classification_report, confusion_matrix
print('Classifcation report:\n', classification_report(y_test, y_pred_2))
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred_2)
print('Confusion matrix:\n', conf_mat)
#trying with undersmapling technique
# This is the pipeline module we need from imblearn for Undersampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# Define which resampling method and which ML model to use in the pipeline
resampling = RandomUnderSampler()
model = RandomForestClassifier(n_estimators=30,criterion='entropy', random_state=0,max_depth=10)


# Define the pipeline and combine sampling method with the RF model
pipeline = Pipeline([('RandomUnderSampler', resampling), ('RF', model)])
pipeline.fit(x_train, y_train) 
predicted = pipeline.predict(x_test)


# Obtain the results from the classification report and confusion matrix 
print('Classifcation report:\n', classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)
# This is the pipeline module we need from imblearn for Oversampling
from imblearn.over_sampling import RandomOverSampler
# Define which resampling method and which ML model to use in the pipeline
resampling = RandomOverSampler()
model = RandomForestClassifier(n_estimators=30,criterion='entropy', random_state=0,max_depth=10)

# Define the pipeline and combine sampling method with the model
pipeline = Pipeline([('RandomOverSampler', resampling), ('RF', model)])
pipeline.fit(x_train, y_train) 
predicted = pipeline.predict(x_test)


# Obtain the results from the classification report and confusion matrix 
print('Classifcation report:\n', classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)
# This is the pipeline module we need from imblearn for SMOTE
from imblearn.over_sampling import SMOTE
# Define which resampling method and which ML model to use in the pipeline
resampling = SMOTE(sampling_strategy='auto',random_state=0)
model = RandomForestClassifier(n_estimators=30,criterion='entropy', random_state=0,max_depth=10)

# Define the pipeline and combine sampling method with the model
pipeline = Pipeline([('SMOTE', resampling), ('RF', model)])
pipeline.fit(x_train, y_train) 
predicted = pipeline.predict(x_test)


# Obtain the results from the classification report and confusion matrix 
print('Classifcation report:\n', classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)

import matplotlib.pyplot as plt
import seaborn as sns
#visulalizing the confusion matrix
LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(y_test, y_pred) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 
