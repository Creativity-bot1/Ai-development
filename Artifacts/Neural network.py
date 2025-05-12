import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score ,confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


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
print(" duplicate output: \n", df.duplicated().sum(),"\n==================\n")

print(newdf['type'].value_counts())

#Replacing empty values with 'Null'
newdf = newdf.replace("-", "Null")

#Optional : combining the nine attack types into one name.
newdf = newdf.replace(['backdoor','ddos','dos','injection','mitm','password','ransomware','scanning','xss'], "Attack")

newdf[['src_ip','dst_ip','proto','service','conn_state','type']] = newdf[['src_ip','dst_ip','proto','service','conn_state','type']].apply(LabelEncoder().fit_transform)


#--------------------------------- training of model below-------------------------------------------------------------

#define the features x and the target vairbale y 
X = newdf[["src_port", "dst_port", "proto", "service",
"src_bytes","dst_bytes", "conn_state", "src_pkts", "dst_pkts","type"]]
y = newdf['label']


#splitting data to train and test use. 20 for test and 80 for training.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

tf.random.set_seed(42) 

initializer = tf.keras.initializers.GlorotUniform(seed=42) # ensures that if any weight inilization or daat shuffling occur the same results will still be prodocued 

model_1 = tf.keras.Sequential([tf.keras.layers.Dense(1, kernel_initializer=initializer)]) # creating the model with its hidden layers and the initilzer functions 

model_1.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['accuracy'])

# Train the model using the training data
model_1.fit(X_train, y_train, epochs=3 ) # training the model.  verbose = 0 removes the training procedure 

#----------------model 2 -------------------
print("model 2 here:")
model_2 = tf.keras.Sequential([ tf.keras.layers.Dense(1,kernel_initializer=initializer),
                               tf.keras.layers.Dense(1,kernel_initializer=initializer)]) # adding a second layer 

model_2.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer = tf.keras.optimizers.SGD(),
                metrics = ['accuracy'])

model_2.fit(X_train, y_train, epochs = 3 )

#----------------model 3 -------------------
print("model 3 here:")
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100,kernel_initializer=initializer), # add 100 dense neurons
    tf.keras.layers.Dense(10,kernel_initializer=initializer), # add another layer with 10 neurons
    tf.keras.layers.Dense(1,kernel_initializer=initializer)])

model_3.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer = tf.keras.optimizers.SGD(),
                metrics = ['accuracy'])

model_3.fit(X_train, y_train, epochs = 3 )

#----------------model 4 -------------------
print("model 4 here:")
model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(128,activation = 'relu', kernel_initializer=initializer), # chaningg neurons and furthemore adding an activation function 
    tf.keras.layers.Dense(64,activation = 'relu', kernel_initializer=initializer),
    tf.keras.layers.Dense(1,activation = 'sigmoid', kernel_initializer=initializer)])

model_4.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer = tf.keras.optimizers.Adam(learning_rate = 0.005),   # adding paramters such as learning rate
                metrics = ['accuracy'])

model_4.fit(X_train, y_train, epochs = 5 ) # epochs is how many time it looks through the dataset 

train_loss, Train_accuracy = model_4.evaluate(X_train, y_train)
print(f' Model loss on the train set: {train_loss}')
print(f' Model accuracy on the train set: {100*Train_accuracy}')

loss, accuracy = model_4.evaluate(X_test, y_test)
print(f' Model loss on the test set: {loss}')
print(f' Model accuracy on the test set: {100*accuracy}')

y_pred_probs = model_4.predict(X_test)
y_pred_classes = (y_pred_probs > 0.5).astype(int)

classification_rep = classification_report(y_test, y_pred_classes)
print("\nClassification Report:\n", classification_rep)