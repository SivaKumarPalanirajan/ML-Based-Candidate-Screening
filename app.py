import pandas as pd 
import pickle 
from flask import Flask,render_template,url_for,request
import pymongo as pym

client=pym.MongoClient('mongodb://127.0.0.1:27017')

db=client['Resume_Screening']
Model_Results=db.Model_Results
classifier=pickle.load(open('Models/Classifier.sav','rb'))

input=[10,1]
model_prediction=classifier.predict([input])
Model_Results.insert_one({'Years of Experience':input[0],'Number of Projects':input[1],'Recruiter Decision':model_prediction[0]})
print(model_prediction[0])