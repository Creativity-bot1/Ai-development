import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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
newdf = newdf.replace(['backdoor','ddos','dos','injection','mitm','password','ransomware','scanning','xss'], "Attack")

# type shoudl be the target variable 
# have evrything else be features 

# look up how to distingush true adn fasle psotiive 