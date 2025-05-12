import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score , precision_score, recall_score, f1_score ,confusion_matrix,ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier

#reads the dataset from my file 
df = pd.read_csv(r"M:\DATA SET for final year\Highway-Rail_Grade_Crossing_Accident_Data.csv") 

#prints the database as it is 
print("OLD DATABASE")
df.info() 

#make a second copy of the database
newdf = df.copy(deep=True) 
# drops all columns besides the ones mentioned
newdf = newdf.drop(df.columns.difference(['Visibility','Weather Condition','Weather Condition Code','Visibility Code',
                                          'Temperature','Total Killed Form 57','Total Injured Form 57',]), axis = 1) 
# this shows the detail of the new database 
print("NEW DATABSE:")
newdf.info()

# drops duplicates 
#newdf = newdf.drop_duplicates() 

#This displays the total amount of duplicates in our new database
print(" duplicate output: \n", newdf.duplicated().sum(),"\n==================\n") 

#Replacing Missing values with 'N/A''
#newdf = newdf.replace("NaN", "N/A") 

# drops mishandled or NaN data completely
newdf.dropna(inplace=True)

# Encode categorical variables to numerical values
newdf[['Visibility','Weather Condition']] = newdf[['Visibility','Weather Condition']].apply(LabelEncoder().fit_transform)

#--- MLP model creation ---
# this helps me create binary targets 
newdf['Killed_Binary'] = (newdf['Total Killed Form 57'] > 0).astype(int)
newdf['Injured_Binary'] = (newdf['Total Injured Form 57'] > 0).astype(int)

# X are my features and y is my target variable
X = newdf[['Visibility', 'Weather Condition', 'Temperature', 'Weather Condition Code', 'Visibility Code']]
y_killed = newdf['Killed_Binary']
y_injured = newdf['Injured_Binary']

# Train-test split for both of my Y target features 
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X, y_killed, test_size=0.2, random_state=42)
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X, y_injured, test_size=0.2, random_state=42)

# Apply SMOTE to both target sets
smote = SMOTE(random_state=42)
X_train_k_smote, y_train_k_smote = smote.fit_resample(X_train_k, y_train_k)
X_train_i_smote, y_train_i_smote = smote.fit_resample(X_train_i, y_train_i)

# Train classifiers
MLP_killed = MLPClassifier(hidden_layer_sizes=(185), max_iter=1000, activation='relu', solver='adam', learning_rate_init = 0.01, 
                           early_stopping=True,validation_fraction=0.1, random_state=42)
MLP_injured = MLPClassifier(hidden_layer_sizes=(185), max_iter=1000, activation='relu', solver='adam',learning_rate_init = 0.01,
                           early_stopping=True,validation_fraction=0.1, random_state=42)

MLP_killed.fit(X_train_k_smote, y_train_k_smote)
MLP_injured.fit(X_train_i_smote, y_train_i_smote)

# this is for prediciting 
y_pred_killed = MLP_killed.predict(X_test_k)
y_pred_injured = MLP_injured.predict(X_test_i)

# Code for Evaluation report
print("\n--- Classification Report: Killed_Binary ---")
print(classification_report(y_test_k, y_pred_killed, zero_division=1))

print("\n--- Classification Report: Injured_Binary ---")
print(classification_report(y_test_i, y_pred_injured, zero_division=1))

# triaining and testing Accuracy scores for Killed_Binary
train_acc_killed = MLP_killed.score(X_train_k_smote, y_train_k_smote)
test_acc_killed = MLP_killed.score(X_train_k_smote, y_train_k_smote)

# training and testing Accuracy scores for Injured_Binary
train_acc_injured = MLP_injured.score(X_train_i_smote, y_train_i_smote)
test_acc_injured = MLP_injured.score(X_train_i_smote, y_train_i_smote)

# Print accuracy results on training and testing 
print("\n--- Accuracy: Killed_Binary ---")
print(f"Training Accuracy: {train_acc_killed:.4f}")
print(f"Testing Accuracy:  {test_acc_killed:.4f}")

print("\n--- Accuracy: Injured_Binary ---")
print(f"Training Accuracy: {train_acc_injured:.4f}")
print(f"Testing Accuracy:  {test_acc_injured:.4f}")

# the following below allows me to print out the scores individualy on the testing---

# Evaluate predictions for Killed_Binary
precision_killed = precision_score(y_test_k, y_pred_killed, zero_division=1)
recall_killed = recall_score(y_test_k, y_pred_killed, zero_division=1)
f1_killed = f1_score(y_test_k, y_pred_killed, zero_division=1)

# Print out the evaluation for Killed_Binary
print("\n--- Evaluation Metrics: Killed_Binary ---")
print(f"Precision: {precision_killed:.4f}")
print(f"Recall: {recall_killed:.4f}")
print(f"F1 Score: {f1_killed:.4f}")

# Evaluate predictions for Injured_Binary
precision_injured = precision_score(y_test_i, y_pred_injured, zero_division=1)
recall_injured = recall_score(y_test_i, y_pred_injured, zero_division=1)
f1_injured = f1_score(y_test_i, y_pred_injured, zero_division=1)

# Prints out the evaluation for Injured_Binary
print("\n--- Evaluation Metrics: Injured_Binary ---")
print(f"Precision: {precision_injured:.4f}")
print(f"Recall: {recall_injured:.4f}")
print(f"F1 Score: {f1_injured:.4f}")


 #Confusion Matrix for Injured_Binary
cm_injured = confusion_matrix(y_test_i, y_pred_injured)
disp_injured = ConfusionMatrixDisplay(confusion_matrix=cm_injured, display_labels=[0, 1])
disp_injured.plot(cmap='Blues')
plt.title('Confusion Matrix: Injured_Binary')
plt.show()

 #Confusion Matrix for Killed_Binary
cm_killed = confusion_matrix(y_test_k, y_pred_killed)
disp_killed = ConfusionMatrixDisplay(confusion_matrix=cm_killed, display_labels=[0, 1])
disp_killed.plot(cmap='Blues')
plt.title('Confusion Matrix: Killed_Binary')
plt.show()

# ROC Curve for Killed_Binary
fpr_killed, tpr_killed, thresholds_killed = roc_curve(y_test_k, MLP_killed.predict_proba(X_test_k)[:, 1])
roc_auc_killed = roc_auc_score(y_test_k, y_pred_killed)

# Plotting the ROC curve for killed
plt.figure(figsize=(10, 6))
plt.plot(fpr_killed, tpr_killed, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc_killed))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # this formats it in a Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for killed')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# ROC Curve for Injured_Binary
fpr_injured, tpr_injured, thresholds_injured = roc_curve(y_test_i, MLP_injured.predict_proba(X_test_i)[:, 1])
roc_auc_injured = roc_auc_score(y_test_i, y_pred_injured)

# Plotting the ROC curve for injured
plt.figure(figsize=(10, 6))
plt.plot(fpr_injured, tpr_injured, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc_injured))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Injured')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# ---------------------------------------- below is the code to creat the random forest to be used to comapre with my MLP --------------------------------------#
#--- plotting random forset againts killed roc curve  -------------------
def plot_roc(model, X_train_k_smote, y_train_k_smote):
   
    # comparing my model againts random forest 
    random_forests_model = RandomForestClassifier(random_state=42)
    random_forests_model.fit(X_train_k_smote, y_train_k_smote)
    
    y_pred_killed = random_forests_model.predict(X_test_k)
    
    acc_killed = random_forests_model.score(X_test_k, y_test_k)
    rf_precision_killed = precision_score(y_test_k, y_pred_killed, zero_division=1)
    rf_recall_killed = recall_score(y_test_k, y_pred_killed, zero_division=1)
    rf_f1_killed = f1_score(y_test_k, y_pred_killed, zero_division=1)
    print("\n--- Random forest Evaluation Metrics: Killed_Binary ---")
    print(f"Precision: {rf_precision_killed:.4f}")
    print(f"Recall: {rf_recall_killed:.4f}")
    print(f"F1 Score: {rf_f1_killed:.4f}")
    print(f"Accuracy for random forest : {acc_killed:.4f}")
    

    rfc_disp = RocCurveDisplay.from_estimator(random_forests_model, X_train_k_smote, y_train_k_smote)
    model_disp = RocCurveDisplay.from_estimator(model, X_train_k_smote, y_train_k_smote, ax=rfc_disp.ax_)
    model_disp.figure_.suptitle("ROC curve: Multilayer Perceptron vs Random Forests on Killed ")

    plt.show()

# using perceptron model as input
plot_roc(MLP_killed, X_train_k_smote, y_train_k_smote)



#--- plotting random forest againts injured roc curve --------------------
def plot_roc(model, X_train_i_smote, y_train_i_smote):
   
    # comparing my model againts random forest 
    random_forests_model = RandomForestClassifier(random_state=42)
    random_forests_model.fit(X_train_i_smote, y_train_i_smote)
    
    y_pred_injured = random_forests_model.predict(X_test_k)
    
    acc_injured = random_forests_model.score(X_test_i, y_test_i)
    rf_precision_injured = precision_score(y_test_i, y_pred_injured, zero_division=1)
    rf_recall_injured = recall_score(y_test_i, y_pred_injured, zero_division=1)
    rf_f1_injured = f1_score(y_test_i, y_pred_injured, zero_division=1)
    print("\n--- Random forest Evaluation Metrics: Injured_Binary ---")
    print(f"Precision: {rf_precision_injured:.4f}")
    print(f"Recall: {rf_recall_injured:.4f}")
    print(f"F1 Score: {rf_f1_injured:.4f}")
    print(f"Accuracy for random forest : {acc_injured:.4f}")

    rfc_disp = RocCurveDisplay.from_estimator(random_forests_model, X_train_i_smote, y_train_i_smote)
    model_disp = RocCurveDisplay.from_estimator(model, X_train_i_smote, y_train_i_smote, ax=rfc_disp.ax_)
    model_disp.figure_.suptitle("ROC curve: Multilayer Perceptron vs Random Forests on Injured ")

    plt.show()

# using perceptron model as input
plot_roc(MLP_killed, X_train_i_smote, y_train_i_smote)




