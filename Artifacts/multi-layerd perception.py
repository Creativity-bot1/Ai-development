import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score ,confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv(r"M:\DATA SET for final year\train_test_network.csv")

print("old database")
df.info()

print("head output:\n" , df.head(), "\n=====================\n")  ## using the variable explorer can see the whole form of dataset.

#first task is dropping the unneedeed colums.
# make a second copy of the database
newdf = df.copy(deep=True)
newdf = newdf.drop(df.columns.difference(['src_ip', "src_port", "dst_ip", "dst_port", "proto", "service","type",
"src_bytes","dst_bytes", "conn_state", "src_pkts", "dst_pkts", "label", "type"]), axis = 1)


print("second fixture")
newdf.info()

#the following below shows total duplicates values in dataset.
print(" duplicate output: \n", newdf.duplicated().sum(),"\n==================\n")

print(newdf['type'].value_counts())

#Replacing empty values with 'Null'
newdf = newdf.replace("-", "Null")

#Optional : combining the nine attack types into one name.
newdf = newdf.replace(['backdoor','ddos','dos','injection','mitm','password','ransomware','scanning','xss'], "Attack")

newdf[['src_ip','dst_ip','proto','service','conn_state']] = newdf[['src_ip','dst_ip','proto','service','conn_state']].apply(LabelEncoder().fit_transform)




#--------------------------------- training of model below-------------------------------------------------------------

#define the features x and the target vairbale y 
X = newdf[["src_port", "dst_port", "proto", "service",
"src_bytes","dst_bytes", "conn_state", "src_pkts", "dst_pkts","label"]]
y = newdf['type']


#splitting data to train and test use. 20 for test and 80 for training.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

mlp_clf = MLPClassifier(hidden_layer_sizes=(149), max_iter=1000,learning_rate_init=0.01,random_state=42,early_stopping=True,validation_fraction=0.15,n_iter_no_change=10 ) # paramters are changed here
mlp_clf.fit(X_train,y_train) # train model

y_pred = mlp_clf.predict(X_test)


print(classification_report(y_test, y_pred))



#calculate and print the accuracy 
y_train_pred = mlp_clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
accuracy = accuracy_score (y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)  
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
print("Accuracy on the training set:", train_accuracy)
print("Accuracy on the testing set:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("f1:", f1)

#---------------- -------------------
def plot_roc(model, X, y):
   
    # comparing the given model with a Random Forests model
    random_forests_model = RandomForestClassifier(random_state=42)
    random_forests_model.fit(X, y) # trains random forest model on same dataset 
    

    rfc_disp = RocCurveDisplay.from_estimator(random_forests_model, X, y) # displays the random forest reults 
    model_disp = RocCurveDisplay.from_estimator(model, X, y, ax=rfc_disp.ax_) # display results of both models
    model_disp.figure_.suptitle("ROC curve: Multilayer Perceptron vs Random Forests")

    plt.show()

# using perceptron model as input
plot_roc(mlp_clf,X, y)
         