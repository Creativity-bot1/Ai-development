import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score ,confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report



df = pd.read_csv(r"M:\DATA SET for final year\Highway-Rail_Grade_Crossing_Accident_Data.csv") #reads the dataset from my file 

print("OLD DATABASE")
df.info() #prints the database as it is 


newdf = df.copy(deep=True) #make a second copy of the database
newdf = newdf.drop(df.columns.difference(['Visibility','Weather Condition','Weather Condition Code','Visibility Code',
                                          'Temperature','Total Killed Form 57','Total Injured Form 57',]), axis = 1) # drops all columns besides the ones mentioned


print("NEW DATABSE:")
newdf.info()

#newdf = newdf.drop_duplicates() # drops duplicates 

#This displays the total amount of duplicates in our new database
print(" duplicate output: \n", newdf.duplicated().sum(),"\n==================\n") 

#Replacing Missing values with 'N/A''
#newdf = newdf.replace("NaN", "N/A") 

# drops mishandled or NaN data completely
newdf = newdf.dropna()

newdf[['Visibility','Weather Condition']] = newdf[['Visibility','Weather Condition']].apply(LabelEncoder().fit_transform)


