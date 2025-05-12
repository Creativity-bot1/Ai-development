from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score ,confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


df = pd.read_csv(r"M:\DATA SET for final year\train_test_network.csv")

print("old database")
df.info()

print("head output:\n" , df.head(), "\n=====================\n")  ## using the variable explorer can see the whole form of dataset.

#first task is dropping the unneedeed colums.
# make a second copy of the database
newdf = df.copy(deep=True)
newdf = newdf.drop(df.columns.difference(['src_ip', "src_port", "dst_ip", "dst_port", "proto", "service",
"src_bytes","dst_bytes", "conn_state", "src_pkts", "dst_pkts", "label", "type"]), axis = 1)


print("second fixture")
newdf.info()

#the following below shows total duplicates values in dataset.
print(" duplicate output: \n", df.duplicated().sum(),"\n==================\n")

print(newdf['type'].value_counts())

#Replacing empty values with 'Null'
newdf = newdf.replace("-", "Null")

#Optional : combining the nine attack types into one name.
#newdf = newdf.replace(['backdoor','ddos','dos','injection','mitm','password','ransomware','scanning','xss'], "Attack")

newdf[['src_ip','dst_ip','proto','service','conn_state']] = newdf[['src_ip','dst_ip','proto','service','conn_state']].apply(LabelEncoder().fit_transform)


#--------------------------------- training of model below-------------------------------------------------------------

#define the features x and the target vairbale y 
X = newdf[["src_port", "dst_port", "proto", "service",
"src_bytes","dst_bytes", "conn_state", "src_pkts", "dst_pkts", "label"]]
y = newdf ["type"]


#splitting data to train and test use. 20 for test and 80 for training.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

param_grid = {'criterion':["gini","entropy"],'min_samples_leaf': [10,2,50],'min_samples_split': [100, 2000 , 30000 , 4000000],'max_depth':[12,13,14,15]} 

#creating a decisontreeclassififer with a maximum depth of 12
tree_clf = DecisionTreeClassifier()

grid_search = GridSearchCV(tree_clf, param_grid, cv=5)

grid_search.fit(X_train, y_train)

tree_clf.fit(X_train, y_train)

print("Best hyperparameters: ", grid_search.best_params_)

#defining column names and unique class values 
column_names = newdf.columns
class_unique_values = ['normal','backdoor','ddos','dos','injection','mitm','password','ransomware','scanning','xss']        
#class_unique_values = ['normal','atack']   

#plot the decison tree - 1,2,7 are the x features that it wants to show and the class unique values is hte traget variable so for me label
plot_tree(tree_clf,feature_names=column_names[[0,1,2,3,4,5,6,7,8,9,10,11]].tolist(),class_names=class_unique_values, filled=True, rounded=True)
plt.show()

#make predictions on the test datset 
y_pred = tree_clf.predict(X_test)

# Predict on the training set
y_train_pred = tree_clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Compare the accuracy on the training set and the testing set
print("Accuracy on the training set:", train_accuracy)

#calculate and print the accuracy 
accuracy = accuracy_score (y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)  
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
print("Accuracy on the testing set:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("f1:", f1)


#Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tree_clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=90)
plt.show()


report = classification_report(y_test, y_pred, target_names=class_unique_values)
print(report)

