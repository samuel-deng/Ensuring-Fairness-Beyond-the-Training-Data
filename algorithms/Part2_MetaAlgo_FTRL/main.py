import pandas as pd
import numpy as np
import time
from meta_algo import MetaAlgorithm
import pickle
import argparse
import datetime
from sklearn.metrics import accuracy_score


dataset_used = 'compas'

def evaluate_fairness(y_pred, sensitive_features):
        """
        Evaluates fairness of the final majority vote classifier over T_inner hypotheses
        on the test set.
        #TODO: add equalized odds option
        #NOTE: defined in the meta_algo file, but we chose:
        a0 := African-American (COMPAS), Female (Adult)
        a1 := Caucasian (COMPAS), Male (Adult)

        :return: list. subgroups in sensitive_features.
        :return: dict. recidivism_pct for each group.
        """
        groups = np.unique(sensitive_features.values)
        indices = {}
        recidivism_count = {}
        recidivism_pct = {}
        for index, group in enumerate(groups):
            indices[group] = sensitive_features.index[sensitive_features == group]
            recidivism_count[group] = sum(y_pred[indices[group]])
            recidivism_pct[group] = recidivism_count[group]/len(indices[group])
        
        gap = abs(recidivism_pct[groups[0]] - recidivism_pct[groups[1]])
        return groups, recidivism_pct, gap

if(dataset_used == 'compas'):
    compas_train = pd.read_csv('./data/compas_train_subset_2.csv')
    compas_val = pd.read_csv('./data/compas_val.csv')
    compas_test = pd.read_csv('./data/compas_test.csv')

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
    parser.add_argument("--epsilon", help="epsilon fairness constraint")
    parser.add_argument("--gamma_1", help="gamma_1 param for weight discretization")
    parser.add_argument("--gamma_2", help="gamma_2 param for LP buckets")
    parser.add_argument("--eta", help="eta param")
    parser.add_argument("--eta_inner", help="eta param for inner loop")
    parser.add_argument("--num_cores", help="number of cores for multiprocessing")
    parser.add_argument("--solver", help="solver for the LPs: [ECOS, OSQP, SCS, GUROBI]")
    parser.add_argument("--name", help="output file name for final ensemble")
    parser.add_argument("--constraint", help="constraint (dp or eo)")
    parser.add_argument("--lbd_group_weights", help="lower bound on group weights")
    parser.add_argument("--no_output", help="disable outputting pkl files")

    now = datetime.datetime.now()
    args = parser.parse_args()
    if(args.B):
        arg_B = float(args.B)
    else:
        arg_B = 1
    if(args.T):
        arg_T = int(args.T)
    else:
        arg_T = 40
    if(args.T_inner):
        arg_T_inner = int(args.T_inner)
    else:
        arg_T_inner = 400
    if(args.epsilon):
        arg_epsilon = float(args.epsilon)
    else:
        arg_epsilon = 0.05
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
        arg_eta = float(1/np.sqrt(2*arg_T))
    if(args.eta_inner):
        arg_eta_inner = float(args.eta_inner)
    else:
        arg_eta_inner = float(1/np.sqrt(2*arg_T_inner))
    if(args.num_cores):
        arg_num_cores = int(args.num_cores)
    else:
        arg_num_cores = 2
    if(args.solver):
        arg_solver = args.solver
    else:
        arg_solver = 'ECOS'
    if(args.constraint):
        arg_constraint = args.constraint
    else:
        arg_constraint = 'dp'
    if(args.lbd_group_weights):
        arg_lbd_g_wt = args.lbd_group_weights
    else:
        arg_lbd_g_wt = 0.35
    if(args.no_output):
        arg_no_output = True
    else:
        arg_no_output = False

    algo = MetaAlgorithm(B = arg_B, T = arg_T, T_inner = arg_T_inner, eta = arg_eta, eta_inner = arg_eta_inner,
                         epsilon=arg_epsilon, gamma_1 = arg_gamma_1, gamma_2 = arg_gamma_2, 
                        num_cores = arg_num_cores, solver = arg_solver, constraint_used=arg_constraint, lbd_g_wt = arg_lbd_g_wt)

    list_hypotheses, final_ensemble = algo.meta_algorithm(X_train, y_train, sensitive_features_train, 
                                                            X_test, y_test, sensitive_features_test)

    if (args.name):
        arg_output = 'ensemble_' + args.name + '.pkl'
        arg_output_list = 'list_' + args.name + '.pkl'
    else:
        arg_output = 'ensemble_B{}_Tinner{}_etainner{}.pkl'.format(arg_B, arg_T_inner, arg_eta_inner) 
        arg_output_list = 'list_B{}_Tinner{}_etainner{}.pkl'.format(arg_B, arg_T_inner, arg_eta_inner)

    y_pred = final_ensemble.predict(X_test)
    groups, recidivism_pct, gap = evaluate_fairness(y_pred, sensitive_features_test)
    print("Accuracy = {}".format(accuracy_score(y_test, y_pred)))
    for group in groups:
        print("P[h(X) = 1 | {}] = {}".format(group, recidivism_pct[group]))
    print("Delta_DP = {}".format(gap))

    # output_file = 'final_y_pred' + now.strftime("%Y-%m-%d_%H:%M:%S") + '.pkl'
    # pickle.dump(y_pred, open(output_file, 'wb') )

    if(not arg_no_output):
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
