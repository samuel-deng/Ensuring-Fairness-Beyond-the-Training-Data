import pandas as pd
import time
from meta_algo import MetaAlgorithm
import pickle

dataset_used = 'compas'

if(dataset_used == 'compas'):
    compas_train = pd.read_csv('./../../data/compas_train.csv')
    compas_val = pd.read_csv('./../../data/compas_val.csv')
    compas_test = pd.read_csv('./../../data/compas_test.csv')

    y_train = compas_train.pop('two_year_recid') 
    y_test = compas_test.pop('two_year_recid')
    sensitive_features_train = compas_train['race']
    sensitive_features_test = compas_test['race']
    X_train = compas_train
    X_test = compas_test
    
    sensitive_features_train = sensitive_features_train.replace(0, 'African-American')
    sensitive_features_train = sensitive_features_train.replace(1, 'Caucasian')
    sensitive_features_test = sensitive_features_test.replace(0, 'African-American')
    sensitive_features_test = sensitive_features_test.replace(1, 'Caucasian')
    
elif(dataset_used == 'adult'):
    adult_train = pd.read_csv('./../../data/adult_train.csv')
    adult_val = pd.read_csv('./../../data/adult_val.csv')
    adult_test = pd.read_csv('./../../data/adult_test.csv')

    y_train = adult_train.pop('Income Binary') 
    y_test = adult_test.pop('Income Binary')
    sensitive_features_train = adult_train['sex']
    sensitive_features_test = adult_test['sex']
    X_train = adult_train
    X_test = adult_test
    
    sensitive_features_train = sensitive_features_train.replace(0, 'Female')
    sensitive_features_train = sensitive_features_train.replace(1, 'Male')
    sensitive_features_test = sensitive_features_test.replace(0, 'Female')
    sensitive_features_test = sensitive_features_test.replace(1, 'Male')
    
else:
    print('Invalid dataset_used variable.')

algo = MetaAlgorithm(B = 100, T=2, T_inner=2, gamma_1 = 0.05, gamma_2 = 0.3, eta=0.05)
list_hypotheses, final_ensemble = algo.meta_algorithm(X_train, y_train, sensitive_features_train)

with open('list_hypotheses.pkl', 'wb') as f:
    pickle.dump(list_hypotheses, f)

with open("final_ensemble.pkl", "wb") as f:
    pickle.dump(final_ensemble, f)


'''
loaded_list = pickle.load(open('list_hypotheses.pkl', 'rb'))
print(loaded_list)
final_ensemble = pickle.load(open('final_ensemble.pckl', 'rb'))
print(final_ensemble.predict(X_test))
'''