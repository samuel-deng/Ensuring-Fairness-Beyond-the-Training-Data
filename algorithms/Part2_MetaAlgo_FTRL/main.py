import pandas as pd
import numpy as np
import time
from meta_algo import MetaAlgorithm
import pickle
import argparse
import datetime
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def evaluate_fairness(y_true, y_pred, sensitive_features):
        """
        Evaluates fairness of the final majority vote classifier over T_inner hypotheses
        on the test set.
        #NOTE: defined in the meta_algo file, but we chose:
        a0 := African-American (COMPAS), Female (Adult)
        a1 := Caucasian (COMPAS), Male (Adult)

        :return: list. subgroups in sensitive_features.
        :return: list, dict, dict. groups is a list of the sensitive features in the dataset. 
        group_metrics is a dictionary containing dictionaries that have Delta_dp, Delta_eoy0, 
        and Delta_eoy1 for each group. gaps is a dictionary that contains the fairness gap
        for dp, eo_y0 and eo_y1.
        """
        groups = np.unique(sensitive_features.values)
        pos_count = {}
        dp_pct = {}
        eo_y0_pct = {}
        eo_y1_pct = {}
        
        for index, group in enumerate(groups):
            # Demographic Parity
            indices = {}
            indices[group] = sensitive_features.index[sensitive_features == group]
            dp_pct[group] = sum(y_pred[indices[group]])/len(indices[group])

            # Equalized Odds
            y1_indices = {}
            y0_indices = {}
            y1_indices[group] = sensitive_features.index[(sensitive_features == group) & (y_true == 1)]
            y0_indices[group] = sensitive_features.index[(sensitive_features == group) & (y_true == 0)]
            eo_y0_pct[group] = sum(y_pred[y0_indices[group]])/len(y0_indices[group])   
            eo_y1_pct[group] = sum(y_pred[y1_indices[group]])/len(y1_indices[group])
        
        gaps = {}
        group_metrics = {} # a dictionary of dictionaries

        gaps['dp'] = abs(dp_pct[groups[0]] - dp_pct[groups[1]])
        gaps['eo_y0'] = abs(eo_y0_pct[groups[0]] - eo_y0_pct[groups[1]])
        gaps['eo_y1'] = abs(eo_y1_pct[groups[0]] - eo_y1_pct[groups[1]])
        group_metrics['dp'] = dp_pct
        group_metrics['eo_y0'] = eo_y0_pct
        group_metrics['eo_y1'] = eo_y1_pct
        
        return groups, group_metrics, gaps

def pick_dataset(dataset_used):
    if(dataset_used == 'compas'):
        X_train = pd.read_csv('./../../data/processed/compas/compas_train1_X.csv')
        X_test = pd.read_csv('./../../data/processed/compas/compas_test1_X.csv')
        y_train = pd.read_csv('./../../data/processed/compas/compas_train1_y.csv')
        y_train = y_train['two_year_recid']
        y_test = pd.read_csv('./../../data/processed/compas/compas_test1_y.csv')
        y_test = y_test['two_year_recid']

        sensitive_features_train = X_train['race']
        sensitive_features_test = X_test['race']        

    elif(dataset_used == 'adult'):
        X_train = pd.read_csv('./../../data/processed/adult/adult_train1_X.csv')
        X_test = pd.read_csv('./../../data/processed/adult/adult_test1_X.csv')
        y_train = pd.read_csv('./../../data/processed/adult/adult_train1_y.csv')
        y_train = y_train['income']
        y_test = pd.read_csv('./../../data/processed/adult/adult_test1_y.csv')
        y_test = y_test['income']

        sensitive_features_train = X_train['sex']
        sensitive_features_test = X_test['sex']

        sensitive_features_train[sensitive_features_train <= 0] = 0
        sensitive_features_train[sensitive_features_train > 0] = 1
        sensitive_features_train = sensitive_features_train.reset_index(drop=True)

        sensitive_features_test[sensitive_features_test <= 0] = 0
        sensitive_features_test[sensitive_features_test > 0] = 1
        sensitive_features_test = sensitive_features_test.reset_index(drop=True)

    elif(dataset_used == 'lawschool'):
        X_train = pd.read_csv('./../../data/processed/lawschool/lawschool_train1_X.csv')
        X_test = pd.read_csv('./../../data/processed/lawschool/lawschool_test1_X.csv')
        y_train = pd.read_csv('./../../data/processed/lawschool/lawschool_train1_y.csv')
        y_train = y_train['bar1']
        y_test = pd.read_csv('./../../data/processed/lawschool/lawschool_test1_y.csv')
        y_test = y_test['bar1']

        sensitive_features_train = X_train['race7']
        sensitive_features_test = X_test['race7']

        sensitive_features_train[sensitive_features_train <= 0] = 0
        sensitive_features_train[sensitive_features_train > 0] = 1
        sensitive_features_train = sensitive_features_train.reset_index(drop=True)

        sensitive_features_test[sensitive_features_test <= 0] = 0
        sensitive_features_test[sensitive_features_test > 0] = 1
        sensitive_features_test = sensitive_features_test.reset_index(drop=True)

    elif(dataset_used == 'communities'):
        X_train = pd.read_csv('./../../data/processed/communities/communities_train1_X.csv')
        X_test = pd.read_csv('./../../data/processed/communities/communities_test1_X.csv')
        y_train = pd.read_csv('./../../data/processed/communities/communities_train1_y.csv')
        y_train = y_train['ViolentCrimesPerPop']
        y_test = pd.read_csv('./../../data/processed/communities/communities_test1_y.csv')
        y_test = y_test['ViolentCrimesPerPop']

        sensitive_features_train = X_train['majority_white']
        sensitive_features_test = X_test['majority_white']
        sensitive_features_train[sensitive_features_train <= 0] = 0
        sensitive_features_train[sensitive_features_train > 0] = 1
        sensitive_features_train = sensitive_features_train.reset_index(drop=True)

        sensitive_features_test[sensitive_features_test <= 0] = 0
        sensitive_features_test[sensitive_features_test > 0] = 1
        sensitive_features_test = sensitive_features_test.reset_index(drop=True)
        
    else:
        raise ValueError("Invalid dataset. Please designate a correct dataset.")

    return X_train, X_test, y_train, y_test, sensitive_features_train, sensitive_features_test

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
    parser.add_argument("--no_output", help="disable outputting pkl files")
    parser.add_argument("--dataset", help="dataset in use for the experiment")
    parser.add_argument("--gp_wt_bd", help="group weight bound on the marginal distributions")

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
        arg_gamma_1 = 0.001
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
        arg_constraint = 'eo'
    if(args.no_output):
        arg_no_output = True
    else:
        arg_no_output = False
    if(args.dataset):
        arg_dataset = args.dataset
    else:
        arg_dataset = 'adult'
    if(args.gp_wt_bd):
        arg_gp_wt_bd = float(args.gp_wt_bd)
    else:
        arg_gp_wt_bd = 0.0331
   
    algo = MetaAlgorithm(B = arg_B, T = arg_T, T_inner = arg_T_inner, eta = arg_eta, eta_inner = arg_eta_inner,
                         epsilon=arg_epsilon, gamma_1 = arg_gamma_1, gamma_2 = arg_gamma_2, num_cores = arg_num_cores, 
                         solver = arg_solver, fair_constraint=arg_constraint, gp_wt_bd = arg_gp_wt_bd)

    X_train, X_test, y_train, y_test, sensitive_features_train, sensitive_features_test = pick_dataset(arg_dataset)
    list_hypotheses, final_ensemble = algo.meta_algorithm(X_train, y_train, sensitive_features_train, 
                                                            X_test, y_test, sensitive_features_test)

    if (args.name):
        arg_output = 'ensemble_' + args.name + '.pkl'
        arg_output_list = 'list_' + args.name + '.pkl'
    else:
        arg_output = 'ensemble_B{}_Tinner{}_etainner{}.pkl'.format(arg_B, arg_T_inner, arg_eta_inner) 
        arg_output_list = 'list_B{}_Tinner{}_etainner{}.pkl'.format(arg_B, arg_T_inner, arg_eta_inner)

    print("=== FINAL ENSEMBLE FAIRNESS EVALUATION ===")
    y_pred = final_ensemble.predict(X_test)
    groups, group_metrics, gaps = evaluate_fairness(y_test, y_pred, sensitive_features_test)
    print("Test Accuracy = {}".format(accuracy_score(y_test, y_pred)))
    if(arg_constraint == 'dp'):
        for group in groups:
            print("P[h(X) = 1 | A = {}] = {}".format(group, group_metrics['dp'][group]))
        print("Delta_dp = {}".format(gaps['dp']))
    elif(arg_constraint == 'eo'):
        for group in groups:
            print("P[h(X) = 1 | A = {}, Y = 0] = {}".format(group, group_metrics['eo_y0'][group]))
            print("P[h(X) = 1 | A = {}, Y = 1] = {}".format(group, group_metrics['eo_y1'][group]))
        print("Delta_eo1 = {}".format(gaps['eo_y1']))
        print("Delta_eo0 = {}".format(gaps['eo_y0']))
    else:
        raise ValueError("Invalid fairness constraint. Choose dp or eo.")

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
