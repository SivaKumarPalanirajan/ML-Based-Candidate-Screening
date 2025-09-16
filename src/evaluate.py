import pandas as pd 
import mlflow 
import yaml 
from dotenv import load_dotenv
import pickle
from sklearn.metrics import r2_score,accuracy_score,f1_score

load_dotenv()

required_features=['Experience (Years)','Projects Count','Recruiter Decision','AI Score (0-100)']
def evaluate(test_data_path:str,regressor_model_path:str,classifier_model_path:str)->None:
    with mlflow.start_run(run_name='Regression-Evaluation'):
        data=pd.read_csv(test_data_path)[required_features]

        data_for_regression=data.copy()
        regressor=pickle.load(open(regressor_model_path,'rb'))

        x_regression=data.drop(columns=['Recruiter Decision','AI Score (0-100)'])
        y_regression=data['AI Score (0-100)']

        y_pred_regression=regressor.predict(x_regression)

        r2=r2_score(y_regression,y_pred_regression)
    
        print('R2 Score',r2)
        mlflow.log_metric('R2 Score',r2)
    
    with mlflow.start_run(run_name='Classification-Evaluation'):
        data_for_classification=data.copy()
        classifier=pickle.load(open(classifier_model_path,'rb'))

        x_classification=data.drop(columns=['Recruiter Decision','AI Score (0-100)'])
        y_classification=data['Recruiter Decision']

        y_pred_classification=classifier.predict(x_classification)

        f1=f1_score(y_classification,y_pred_classification,pos_label='Hire')
        accuracy=accuracy_score(y_classification,y_pred_classification)
        print('F1 Score',f1)
        print('Accuracy',accuracy)

        mlflow.log_metric('F1 Score',f1)
        mlflow.log_metric('Accuracy',accuracy)

if __name__=='__main__':

    mlflow.set_experiment('Resume-Screening-Evaluation')
    params=yaml.safe_load(open('params.yaml'))['evaluate']

    test_data_path=params['test_data_path']
    regressor_model_path=params['regressor_model_path']
    classifier_model_path=params['classifier_model_path']

    evaluate(test_data_path,regressor_model_path,classifier_model_path)