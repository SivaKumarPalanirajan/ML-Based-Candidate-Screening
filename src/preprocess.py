import pandas as pd 
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split 
import yaml 
import os 

def preprocess(input_path,train_data_path,test_data_path,preprocessor_path):

    data=pd.read_csv(input_path)

    data.drop_duplicates(inplace=True)
    print('Dropped duplicates if found')

    # fix null values
    data['Certifications']=data['Certifications'].fillna(data['Certifications'].mode()[0])
    print('Checked and filled null values if found')

    # remove outliers
    max_limit=data['AI Score (0-100)'].mean()+3*data['AI Score (0-100)']
    min_limit=data['AI Score (0-100)'].mean()-3*data['AI Score (0-100)']

    data=data[(data['AI Score (0-100)']<max_limit) & (data['AI Score (0-100)']>min_limit)]

    print('Checked and removed Outliers if found')

    # fix class imbalance
    class1_cnt=data[data['Recruiter Decision']=='Hire'].shape[0]
    class2_cnt=data[data['Recruiter Decision']=='Reject'].shape[0]

    if class1_cnt!=class2_cnt:
        print(f'Class imbalance found: {class1_cnt} - {class2_cnt}')
        x=data.drop(columns=['Recruiter Decision'])
        y=data['Recruiter Decision']
        over_sampler=SMOTENC(sampling_strategy={'Hire':1000,'Reject':1000},categorical_features=['Name','Skills','Education','Certifications','Job Role'])
        x_resampled,y_resampled=over_sampler.fit_resample(x,y)
        data=pd.concat([x_resampled,y_resampled],axis=1)
        print('Fixed class imbalance')

    train,test=train_test_split(data,test_size=0.2)

    os.makedirs(os.path.dirname(train_data_path),exist_ok=True)
    
    train.to_csv(train_data_path,index=False)
    print('Saved Train set')

    test.to_csv(test_data_path,index=False)
    print('Saved Test set')


if __name__=='__main__':
    params=yaml.safe_load(open('params.yaml'))['preprocess']

    input_path=params['input_path']
    train_data_path=params['train_data_path']
    test_data_path=params['test_data_path']
    preprocessor_path=params['preprocessor_path']

    preprocess(input_path,train_data_path,test_data_path,preprocessor_path)
