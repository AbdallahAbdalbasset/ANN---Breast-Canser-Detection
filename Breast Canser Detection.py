import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler




# Getting the dataset and apllying a preprocessing to match our purpose

def prepareData():
    
    dataset = pd.read_csv('breast-cancer.csv')

    dataset.drop(['id'],axis = 1,inplace = True)

    inputdata = dataset.drop(['diagnosis'],axis=1)

    outputdata = dataset.diagnosis

    lb = LabelEncoder()

    outputdata = lb.fit_transform(outputdata)

    
    return inputdata,outputdata


# Building the ANN model using the dataset

def buildTheModel(traingingData,traingingDtaLabels):
    
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(9,activation = 'relu',input_dim = 30),
    tf.keras.layers.Dense(9,activation = 'relu'),
    tf.keras.layers.Dense(1,activation = 'sigmoid')])

    model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(trainingData,trainingDataLabels,batch_size = 100,epochs = 100)

    return model


# Calculating the accuracy of our model
    
def accuracy(testData,testDataLabels):
    
    predectedLabels = model.predict(testData)

    predectedLabels = (predectedLabels>0.5)

    accuracy = accuracy_score(testDataLabels,predectedLabels)
    
    return accuracy


# Using the neural network after comleting its training and testing

def runTheApplication():
    
    flag = True

    again = ""

    while(flag):
        
        try:
            
            evaluationdata = list(map(float, input(again + "Enter the data of the person to predict separated by \",\" : \n").strip().split(',')))
            
            evaluationdata = np.array(evaluationdata).reshape(1,30)

            predectedLabel = model.predict(evaluationdata)

            print('\nPercentage of beinng Malignant is : ',predectedLabel[0]*100,'% \n')
    
            predectedLabel = (predectedLabel>0.5)
    
            if predectedLabel == 1:
            
                print('The diagnosis is \"M\" for Malignant \n')
            else:
                print('The diagnosis is \"B\" for Benign \n')
        except:
             print('Invalid input \n')
        again = "again, "


# Split the data labels
    
inputData,outputData = prepareData()

# Divide the data into training and testing data

trainingData,testData,trainingDataLabels,testDataLabels = train_test_split(inputData,outputData,test_size = 0.2,random_state = 40)

# Building the ANN

model = buildTheModel(trainingData,trainingDataLabels)

# Calculating the accuracy over the test data

accuracy = accuracy(testData,testDataLabels)

print('The accuracy is : ',accuracy,'\n')

runTheApplication()






