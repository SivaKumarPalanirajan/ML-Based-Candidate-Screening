import pandas as pd 
import pickle 
from flask import Flask,render_template,url_for,request,redirect
import pymongo as pym
import requests
import os 
import numpy as np
from dotenv import load_dotenv 
import mlflow 

load_dotenv()

uri=os.environ.get('MONGO_URI')

app=Flask(__name__)

client=pym.MongoClient(uri)
db=client.Resume_Screening
Model_results=db.Model_Results

buffer={'Experience':None,'Projects':None,'Type of Processing':None}

@app.route('/',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        if request:
            experience=request.form.get('Experience in Years')
            projects=request.form.get('No of projects')
            if int(experience)<0 or int(projects)<0:
                return render_template('index.html',error='PLEASE ENTER VALID INPUTS')
            buffer['Type of Processing']=request.form.get('type_of_prediction')
            buffer['Experience']=request.form.get('Experience in Years')
            buffer['Projects']=request.form.get('No of projects')
            return redirect(url_for('processing'))
        
    return render_template('index.html')

@app.route('/processing')
def processing():
    Result_not_found_in_db=False
    experience=int(buffer.get('Experience'))
    projects=int(buffer.get('Projects'))
    type=buffer.get('Type of Processing')


    input_to_model=[[experience,projects]]
    data_from_db=Model_results.find_one({'Years of Experience':experience,'Number of Projects':projects},['Result'])

    if data_from_db is not None:
        if ('AI' in data_from_db['Result'].split(' ')) and (type=='AI Score'):
            result=data_from_db['Result']
        elif ('predict' in data_from_db['Result'].split(' ')) and (type=="Recruiter Decision"):
            result=data_from_db['Result']
        else:
            Result_not_found_in_db=True
        
    if Result_not_found_in_db:  
        if type=='AI Score':
            # result=requests.post('http://regression:5000/make_prediction',json={'exp':experience,'projs':projects})
            # prediction=result.json()['prediction']
            regressor=mlflow.sklearn.load_model('models:/Regression-Resume-Screening-LR/Latest')
            ai_score=regressor.predict(input_to_model)[0]
            prediction=np.exp(-ai_score)
            prediction=round(float(prediction),0)
            if prediction>100:
                prediction=100
            elif prediction<0:
                prediction=0
            result=f'The AI Score for you is {prediction}'
        else:
            # result=requests.post('http://classification:5000/make_prediction',json={'exp':experience,'projs':projects})
            # prediction=result.json()['prediction']
            classifier=mlflow.sklearn.load_model('models:/Classification-Resume-Screening-Stacking/Latest')
            prediction=classifier.predict(input_to_model)
            if prediction[0]=='Hire':
                result=f'We predict that you will be selected'
            else:
                result=f'We predict that you will be rejected'


        Model_results.insert_one({'Years of Experience':experience,'Number of Projects':projects,'Result':result})
    return render_template('result.html',prediction=result)   


if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)
