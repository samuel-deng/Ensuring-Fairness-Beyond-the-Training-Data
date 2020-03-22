import pandas as pd
import numpy as np
import itertools
import cvxpy as cp
import math
import pulp
from joblib import dump, load
from tqdm import tqdm
import time

class LambdaBestResponse:
    """ 
    The Lambda Best Response step that solves an LP to give a single tuple
    Lambda back to the h player. This tuple represents the entry in the Lambda
    vector that gets updated.
    
    :param  
    :type 
    """
    def __init__(self, h_pred, X, y, weights, sensitive_features, a_indices, 
                card_A, nu, M, B, T_1, gamma_1, gamma_1_buckets, gamma_2_buckets, epsilon, eta):
        self.h_pred = np.asarray(h_pred)
        self.X = X
        self.y = y 
        self.weights = weights
        self.sensitive_features = sensitive_features
        self.a_indices = a_indices
        self.card_A = card_A
        self.nu = nu
        self.M = M
        self.B = B
        self.T_1 = T_1
        self.gamma_1 = gamma_1
        self.gamma_1_buckets = gamma_1_buckets
        self.gamma_2_buckets = gamma_2_buckets
        self.epsilon = epsilon
        self.eta = eta

    def _discretize_weights_bsearch(self, w):
        for i, w_i in enumerate(w):
            bucket = self._binary_search_buckets(w_i)
            w[i] = bucket[1] # upper endpoint
        
        return w 
    
    def _binary_search_buckets(self, x):
        first = 0
        last = len(self.gamma_1_buckets) - 1
        found = False
        while(first<=last and not found):
            mid = (first + last)//2
            if self.gamma_1_buckets[mid][0] <= x <= self.gamma_1_buckets[mid][1]:
                found = True
                bucket = self.gamma_1_buckets[mid]
            else:
                if x < self.gamma_1_buckets[mid][0]:
                    last = mid - 1
                elif x > self.gamma_1_buckets[mid][1]:
                    first = mid + 1	
    
        if not found:
            raise(ValueError("Discretization FAILED."))
        
        return bucket


    def _best_response_LP_cvxpy(self, pi, a, a_p):
        # w variable to solve over
        w = cp.Variable(len(self.X))

        # Problem constants
        h_xi_a = self.h_pred.copy()
        h_xi_a[self.a_indices[a]] = 0
        h_xi_ap = self.h_pred.copy()
        h_xi_ap[self.a_indices[a_p]] = 0

        # Constraints
        constraints = [
            cp.sum(w[self.a_indices[a]]) == pi[0],
            cp.sum(w[self.a_indices[a_p]]) == pi[1],
            cp.sum(w) == pi[0] + pi[1],
            0 <= w
        ]

        # Objective Function
        objective = cp.Maximize((1/pi[0] * (w @ h_xi_a)) - (1/pi[1] * (w @ h_xi_ap)))
        prob = cp.Problem(objective, constraints)        
        print(prob.is_dcp())
        prob.solve(solver='GUROBI', verbose=False, warm_start=True)
        
        return prob.value, w.value

    def best_response(self):
        w_dict = dict()
        val_dict = dict()

        N_gamma_2_A = self.gamma_2_buckets 
        a_a_p = list(itertools.permutations(['a0', 'a1']))

        start = time.time()
        total_LP = 0
        num_LP = 0
        for pi in N_gamma_2_A:
            for (a, a_p) in a_a_p:
                start_LP = time.time()
                max_lp, argmax_lp = self._best_response_LP_cvxpy(pi, a, a_p)
                end_LP = time.time()
                total_LP += end_LP - start_LP
                num_LP += 1

                # some < 0 entries in the argmax
                argmax_lp[argmax_lp < 0] = 0
                
                dict_key = (a, a_p, pi)
                w_dict[dict_key] = argmax_lp
                if max_lp != None:
                    val_dict[dict_key] = max_lp
                else: # when objective value is null 
                    raise ValueError("LP Failed.")
        end = time.time()
        print("LP TIME: " + str(end - start))
        print("Total LP TIME: " + str(total_LP))
        print("num LPs: " + str(num_LP))

        optimal_tuple = max(val_dict, key=val_dict.get)

        # Violation of fairness
        if(val_dict[optimal_tuple] > self.epsilon - 4*self.gamma_1):
            optimal_w = w_dict[optimal_tuple]
            optimal_w = self._discretize_weights_bsearch(optimal_w)
            
            lambda_w_a_ap = self.B
            lambda_entry = (optimal_tuple[0], optimal_tuple[1], tuple(optimal_w)) # lists aren't hashable
            # return of form ('a0', 'a1', weight_vector)
        else:
            lambda_w_a_ap = 0
            lambda_entry = (0, 0, 0)

        return lambda_entry

    