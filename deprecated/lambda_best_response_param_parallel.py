import pandas as pd
import numpy as np
import itertools
import cvxpy as cp
import math
from joblib import dump, load
from tqdm import tqdm
import time
from multiprocessing import Pool
import multiprocessing

class LinearProgram():
    def __init__(self, n, h_pred, a_indices, a, a_p):
        self._w = cp.Variable(n)
        self._a = a
        self._a_p = a_p

        # Problem constants
        self._h_xi_a = h_pred.copy()
        self._h_xi_a[a_indices[a_p]] = 0 # (we only want indices of a)
        self._h_xi_ap = h_pred.copy() 
        self._h_xi_ap[a_indices[a]] = 0 # (we only want indices of a_p)
        self.pi_0 = cp.Parameter(nonneg=True)
        self.pi_1 = cp.Parameter(nonneg=True)

        # Constraints
        self._constraints = [
            cp.sum(self._w[a_indices[a]]) == self.pi_0,
            cp.sum(self._w[a_indices[a_p]]) == self.pi_1,
            cp.sum(self._w) == self.pi_0 + self.pi_1, # don't exactly sum to 1 sometimes
            0 <= self._w
            #cp.sum(self._w[a_indices[a]]) >= 0.1,
            #cp.sum(self._w[a_indices[a]]) >= 0.1
        ]

        # Objective Function
        self._objective = cp.Maximize((1/self.pi_0 * (self._w @ self._h_xi_a)) - (1/self.pi_1 * (self._w @ self._h_xi_ap)))
        self._prob = cp.Problem(self._objective, self._constraints) 

    def solve(self, pi):
        self.pi_0.value = pi[0]
        self.pi_1.value = pi[1]
        self._prob.solve(solver='GUROBI', verbose=False, warm_start = True)
        return self._prob.value, self._w.value, (self._a, self._a_p, (pi[0], pi[1]))

class LambdaBestResponse:
    """ 
    The Lambda Best Response step that solves an LP to give a single tuple
    Lambda back to the h player. This tuple represents the entry in the Lambda
    vector that gets updated.
    
    :param  
    :type 
    """
    def __init__(self, h_pred, X, y, weights, sensitive_features, a_indices, 
                card_A, nu, M, B, T_inner, gamma_1, gamma_1_buckets, gamma_2_buckets, epsilon, eta):
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
        self.T_inner = T_inner
        self.gamma_1 = gamma_1
        self.gamma_1_buckets = gamma_1_buckets
        self.gamma_2_buckets = gamma_2_buckets
        self.epsilon = epsilon
        self.eta = eta

    def _discretize_weights_bsearch(self, w):
        for i, w_i in enumerate(w):
            bucket = self._binary_search_buckets(w_i)
            w[i] = bucket[1]
        
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

    def best_response(self):
        num_cores = multiprocessing.cpu_count()
        print(num_cores)

        w_dict = dict()
        val_dict = dict()
        N_gamma_2_A = self.gamma_2_buckets
        a_a_p = list(itertools.permutations(['a0', 'a1']))
        print("Solving " + str(len(N_gamma_2_A)) + " LPs...")

        start = time.time()
        solved_results = []
        for (a, a_p) in a_a_p:
            problem = LinearProgram(len(self.X), self.h_pred, self.a_indices, a, a_p)
            pool = Pool(processes = num_cores)
            solved_results.extend(pool.map(problem.solve, N_gamma_2_A))
        end = time.time()
        print("LP TIME: " + str(end - start))

        # max over the objective values
        max_lp = -1e5
        argmax_lp = np.zeros
        for result in solved_results:
            if result[0] > max_lp:
                max_lp = result[0]
                argmax_lp = result[1]
                argmax_lp[argmax_lp < 0] = 0 # might be some slightly < 0 entries for argmax
                optimal_tuple = result[2]
        
        # Violation of fairness
        if(max_lp > self.epsilon - 4*self.gamma_1):
            optimal_w = argmax_lp
            optimal_w[optimal_w < 0] = 0
            optimal_w = self._discretize_weights_bsearch(optimal_w) # let w_i be the upper end-point of bucket
            
            lambda_w_a_ap = self.B
            lambda_entry = (optimal_tuple[0], optimal_tuple[1], tuple(optimal_w)) # lists aren't hashable
            # return of form ('a0', 'a1', weight_vector)
        else:
            lambda_w_a_ap = 0
            lambda_entry = (0, 0, 0)

        return lambda_entry

    