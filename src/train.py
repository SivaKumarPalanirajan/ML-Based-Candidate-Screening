import pandas as pd 
from sklearn.ensemble import StackingClassifier 
from sklearn.linear_model import LinearRegression ,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score,classification_report,precision_score,recall_score,f1_score
import os 
import mlflow
from mlflow.models import infer_signature
import yaml 
import pickle 
import optuna 
from dotenv import load_dotenv
from typing import Dict
load_dotenv()


required_features=['Experience (Years)','Projects Count','Recruiter Decision','AI Score (0-100)']


def train_regressor(train_data_path:str,regressor_model_path:str)->None:
    with mlflow.start_run(run_name='Resume-Checker-AI-Score-LR'):
        data=pd.read_csv(train_data_path)[required_features]

        x=data.drop(columns=['AI Score (0-100)','Recruiter Decision'])
        y=data['AI Score (0-100)']

        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1)

        regressor=LinearRegression()
        regressor.fit(X_train,y_train)
        print('Model trained')

        prediction=regressor.predict(X_test)

        training_score=regressor.score(X_train,y_train)
        print(f"Model Training score: {round(training_score*100,2)}%")

        r2=r2_score(y_test,prediction)
        print(f"R2 Score: {r2}")

        mse=mean_squared_error(y_test,prediction)
        print(f"MSE: {mse}")

        signature=infer_signature(X_train,y_train)

        mlflow.log_params(regressor.get_params())
        mlflow.log_metrics({'R2 Score':r2,'Model Training Score':training_score,'MSE':mse})
        
        mlflow.sklearn.log_model(regressor,'Regression-Resume-Screening',
                                 signature=signature,
                                 registered_model_name='Regression-Resume-Screening-LR',
                                 input_example=X_train)
        
        os.makedirs(os.path.dirname(regressor_model_path),exist_ok=True)
        pickle.dump(regressor,open(regressor_model_path,'wb'))
        print('Saved Model')
        
def train_classifier(train_data_path:str,classifier_model_path:str,classifier_HPs:Dict)->None:
    with mlflow.start_run(run_name='Resume-Checker-Status-Stacking'): 
        data=pd.read_csv(train_data_path)[required_features]

        x=data.drop(columns=['AI Score (0-100)','Recruiter Decision'])
        y=data['Recruiter Decision']

        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1)

        regularization_params=classifier_HPs['C']

        def objective(trail):
            param={'C':trail.suggest_categorical('C',regularization_params)} 

            log_reg=LogisticRegression()
            svc=SVC(**param)

            nb=GaussianNB()

            model=StackingClassifier(
                estimators=[('logistic_reg',log_reg),('svc',svc)],
                final_estimator=nb
            )
            model.fit(X_train,y_train)

            y_pred=model.predict(X_test)

            return accuracy_score(y_test,y_pred)
        

        optuna_hp_tuning=optuna.create_study()
        optuna_hp_tuning.optimize(objective,n_trials=5)
        
        print('Finished HP Tuning')
        best_params=optuna_hp_tuning.best_trial.params

        for i,trial in enumerate(optuna_hp_tuning.trials):
            with mlflow.start_run(run_name=f'HP RUN {i}',nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric('Accuracy',trial.value)
        
        log_reg=LogisticRegression()
        svc=SVC(**best_params)

        nb=GaussianNB()

        best_model=StackingClassifier(
            estimators=[('logistic_reg',log_reg),('svc',svc)],
            final_estimator=nb
        )

        best_model.fit(X_train,y_train)
        y_pred=best_model.predict(X_test)
        print('Model trained')

        training_score=best_model.score(X_train,y_train)
        accuracy=accuracy_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred,pos_label='Hire')
        precision=precision_score(y_test,y_pred,pos_label='Hire')
        f1=f1_score(y_test,y_pred,pos_label='Hire')
        class_report=classification_report(y_test,y_pred)

        print('Training Score: ',training_score)
        print('Accuracy: ',accuracy)
        print('Recall: ',recall)
        print('Precision: ',precision)
        print('F1 Score: ',f1)

        mlflow.log_metrics({'Training Score':training_score,
                            'Accuracy':accuracy,
                            'Recall':recall,
                            'Precision':precision,
                            'F1 Score':f1})
        
        signature=infer_signature(X_train,y_train)

        mlflow.log_params(best_params)
        mlflow.log_text(str(class_report),'Classification_Report.txt')
        mlflow.sklearn.log_model(best_model,
                                 'Classification-Resume-Screening',
                                 signature=signature,
                                 input_example=X_train,
                                 registered_model_name='Classification-Resume-Screening-Stacking'
                                 )
        
        os.makedirs(os.path.dirname(classifier_model_path),exist_ok=True)
        pickle.dump(best_model,open(classifier_model_path,'wb'))
        print('Saved Model')
        


if __name__=='__main__':
    params=yaml.safe_load(open('params.yaml'))['train']

    mlflow.set_experiment('Resume-Checker')

    train_data_path=params['train_data_path'] 
    classifier_model_path=params['classifier_model_path'] 
    classifier_HPs=params['classifier_HPs']
    regressor_model_path=params['regressor_model_path']

    train_regressor(train_data_path,regressor_model_path)

    train_classifier(train_data_path,classifier_model_path,classifier_HPs)
