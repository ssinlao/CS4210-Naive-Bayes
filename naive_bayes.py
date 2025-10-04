#-------------------------------------------------------------------------
# AUTHOR: Stella Sinlao
# FILENAME: naive_bayes.py
# SPECIFICATION: output the classification of each of the 10 instances
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour 20 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here

X = []

outlook = {"Sunny": 1, "Overcast": 2, "Rain":3}
temperature = {"Hot": 1, "Mild": 2, "Cool": 3}
humidity = {"High": 1, "Normal": 2}
wind = {"Weak": 1, "Strong": 2}
playtennis = {"Yes": 1, "No": 2}
playtennis_output = {1: "Yes", 2: "No"}

for row in dbTraining:
    X.append([
        outlook[row[1]],
        temperature[row[2]],
        humidity[row[3]],
        wind[row[4]]
    ])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
for row in dbTraining:
    Y.append(playtennis[row[5]])
#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
print("Day          Outlook      Temperature  Humidity     Wind         PlayTennis   Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for i, row in df.iterrows():
    day = row['Day']
    features = [outlook[row['Outlook']], temperature[row['Temperature']], humidity[row['Humidity']], wind[row['Wind']]]
    probs = clf.predict_proba([features])[0]
    max_prob = max(probs)
    prediction = clf.predict([features])[0]
    if max_prob >= 0.75:
        print(f"{day:12} {row['Outlook']:12} {row['Temperature']:12} {row['Humidity']:12} {row['Wind']:12} {playtennis_output[prediction]:12} {max_prob:.2f}")