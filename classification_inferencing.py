import pandas as pd 
from flask import Flask,request,jsonify
import pickle 

app=Flask(__name__)

classifier=pickle.load(open('C_model.pkl','rb'))

@app.route('/make_prediction',methods=['POST','GET'])
def make_prediction():
    data=request.get_json()
    exp=data['exp']
    proj=data['projs']
    prediction=classifier.predict([[exp,proj]])[0]
    return jsonify({'prediction':prediction})


if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)
