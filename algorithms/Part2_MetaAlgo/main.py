import pandas as pd
import time
from meta_algo import MetaAlgorithm
import pickle
import argparse
import datetime

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

    X_train = X_train.drop('Unnamed: 0', axis=1)
    X_test = X_test.drop('Unnamed: 0', axis=1)
    
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

    X_train = X_train.drop('Unnamed: 0', axis=1)
    X_test = X_test.drop('Unnamed: 0', axis=1)
    
    sensitive_features_train = sensitive_features_train.replace(0, 'Female')
    sensitive_features_train = sensitive_features_train.replace(1, 'Male')
    sensitive_features_test = sensitive_features_test.replace(0, 'Female')
    sensitive_features_test = sensitive_features_test.replace(1, 'Male')
    
else:
    print('Invalid dataset_used variable.')

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", help="upper bound on the Lambda value")
    parser.add_argument("--T", help="number of iterations for outer loop")
    parser.add_argument("--T_inner", help="number of iterations for inner loop")
    parser.add_argument("--gamma_1", help="gamma_1 param for weight discretization")
    parser.add_argument("--gamma_2", help="gamma_2 param for LP buckets")
    parser.add_argument("--eta", help="eta param")
    parser.add_argument("--num_cores", help="number of cores for multiprocessing")
    parser.add_argument("--solver", help="solver for the LPs: [ECOS, OSQP, SCS, GUROBI]")
    parser.add_argument("--output_list", help="output file name for list of hypotheses")
    parser.add_argument("--output", help="output file name for final ensemble")

    now = datetime.datetime.now()
    args = parser.parse_args()
    if(args.B):
        arg_B = float(args.B)
    else:
        arg_B = 10
    if(args.T):
        arg_T = int(args.T)
    else:
        arg_T = 25
    if(args.T_inner):
        arg_T_inner = int(args.T_inner)
    else:
        arg_T_inner = 200
    if(args.gamma_1):
        arg_gamma_1 = float(args.gamma_1)
    else:
        arg_gamma_1 = 0.01
    if(args.gamma_2):
        arg_gamma_2 = float(args.gamma_2)
    else:
        arg_gamma_2 = 0.05
    if(args.eta):
        arg_eta = float(args.eta)
    else:
        arg_eta = 0.05
    if(args.num_cores):
        arg_num_cores = int(args.num_cores)
    else:
        arg_num_cores = 2
    if(args.solver):
        arg_solver = args.solver
    else:
        arg_solver = 'ECOS'
    if(args.output_list):
        arg_output_list = args.output_list
    else:
        arg_output_list = 'list_hypotheses_' + now.strftime("%Y-%m-%d_%H:%M:%S") + '.pkl'
    if(args.output):
        arg_output = args.output
    else:
        arg_output = 'hypotheses_ensemble_' + now.strftime("%Y-%m-%d_%H:%M:%S") + '.pkl'

    print("=== OUTPUT FILES ===")
    print("List of Hypotheses: " + str(arg_output_list))
    print("Ensemble Classifier: " + str(arg_output))
    algo = MetaAlgorithm(B = arg_B, T = arg_T, T_inner = arg_T_inner, 
                        gamma_1 = arg_gamma_1, gamma_2 = arg_gamma_2, 
                        eta = arg_eta, num_cores = arg_num_cores, solver = arg_solver)

    list_hypotheses, final_ensemble = algo.meta_algorithm(X_train, y_train, sensitive_features_train)

    with open(arg_output_list, 'wb') as f:
        pickle.dump(list_hypotheses, f)

    with open(arg_output, "wb") as f:
        pickle.dump(final_ensemble, f)

'''
loaded_list = pickle.load(open('list_hypotheses.pkl', 'rb'))
print(loaded_list)
final_ensemble = pickle.load(open('final_ensemble.pckl', 'rb'))
print(final_ensemble.predict(X_test))
'''