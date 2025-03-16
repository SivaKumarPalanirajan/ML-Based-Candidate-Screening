import pandas as pd 
import pickle 
from flask import Flask,render_template,url_for,request,redirect
import pymongo as pym

file=open('mongodb_url.txt')
url=file.readlines()[0]

app=Flask(__name__)
client=pym.MongoClient(url)
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
    experience=int(buffer.get('Experience'))
    projects=int(buffer.get('Projects'))
    input_to_model=[experience,projects]

    if buffer.get('Type of Processing')=='AI Score':
        regressor=pickle.load(open('Models/Regressor.sav','rb'))
        prediction=round(regressor.predict([input_to_model])[0],0)
        if prediction>100:
            prediction=100
        elif prediction<0:
            prediction=0
        result=f'The AI Score for you is {prediction}'
    else:
        classifier=pickle.load(open('Models/Classifier.sav','rb'))
        prediction=classifier.predict([input_to_model])[0]
        result=f'The predicted Recruiter decision is {prediction}'
    Model_results.insert_one({'Years of Experience':experience,'Number of Projects':projects,'Result':result})
    return render_template('result.html',prediction=result)   


if __name__=="__main__":
    app.run(debug=True)
