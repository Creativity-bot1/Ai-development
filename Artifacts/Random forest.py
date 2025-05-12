from sklearn.ensemble import RandomForestClassifier
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,ConfusionMatrixDisplay,f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import randint


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


#Replacing empty values with 'Null'
newdf = newdf.replace("-", "Null")

#Optional : combining the nine attack types into one name.
#newdf = newdf.replace(['backdoor','ddos','dos','injection','mitm','password','ransomware','scanning','xss'], "Attack")

# transformng data from categroical tp numerical 
newdf[['src_ip','dst_ip','proto','service','conn_state']] = newdf[['src_ip','dst_ip','proto','service','conn_state']].apply(LabelEncoder().fit_transform)

#--------------------------------- training of model below-------------------------------------------------------------

#define the features x and the target vairbale y 
X = newdf[['src_ip', "src_port", "dst_ip", "dst_port", "proto", "service",
"src_bytes","dst_bytes", "conn_state", "src_pkts", "dst_pkts", "label"]]
y = newdf ["type"]

                       
#splitting data to train and test use. 20 for test and 80 for training.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

# setting the vairbale that contains paramter to be tuned 
param_dist = {'min_samples_leaf': [1,50],'max_features': ['sqrt','log2',None],'min_samples_split': [2,100,500],'n_estimators': randint(50,500),'max_depth': randint(1,20)}

#creating a Rf classsifer 
rf = RandomForestClassifier(oob_score = True)

# Set up k-fold cross-validation (using 5-fold cross-validation)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform k-fold cross-validation using cross_val_score
cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')

# Print the accuracy for each fold
print(f"Cross-validation scores for each fold: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")


# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf,param_distributions = param_dist,n_iter=5, cv=5) # increasing the cv effects how long it takes to train

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

#training the model
rf.fit(X_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# displaying oob_score
print('Out-of-Bag score:', rf.oob_score_)

# Generates predictions with the best model
y_pred = best_rf.predict(X_test)

# displaying the accuracy on training results 
y_train_pred = rf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Compare the accuracy on the training set and the testing set
print("Accuracy on the training set:", train_accuracy)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)  
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)

print("Precision:", precision)
print("Recall:", recall)
print("f1:", f1)


#Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=90)
plt.show()










