import numpy as np
import itertools
import cvxpy as cp
import math
import time
from multiprocessing import Pool
import multiprocessing

"""
The LP we want to solve for the Lambda player. Written like this so we can take
advantage of multiprocessing via Pool.map()

Returns:
(1) prob.value. the solved objective value of the LP.
(2) w.value. the argmax w that solved the LP.
(3) (a, a_p, (pi[0], pi[1])). the fixed a, a_p, and pi proportions 
for the linear program. these are constants passed through the loop
in Algorithm 2.
"""
class DpLinearProgram():
    def __init__(self, n, h_pred, a_indices, a, a_p, solver):
        self._w = cp.Variable(n)
        self._a = a
        self._a_p = a_p
        self.solver = solver

        # Problem constants
        self._h_xi_a = h_pred.copy()
        self._h_xi_a[a_indices[a_p]] = 0 # we only want the indices where a_i = a after dotting, so set rest to 0
        self._h_xi_ap = h_pred.copy() 
        self._h_xi_ap[a_indices[a]] = 0 # we only want the indices where a_i = a_p after dotting, so set rest to 0
        self.pi_a = cp.Parameter(nonneg=True)
        self.pi_a_p = cp.Parameter(nonneg=True)

        # Constraints
        self._constraints = [
            cp.sum(self._w[a_indices[a]]) == self.pi_a,
            cp.sum(self._w[a_indices[a_p]]) == self.pi_a_p,
            cp.sum(self._w) == 1, # don't EXACTLY sum to 1 sometimes
            0 <= self._w,
        ]

        # Objective Function
        # NOTE: @ is dot product between w and the h prediction vector with all a or a_p zero'd out
        self._objective = cp.Maximize(((1/self.pi_a) * (self._w @ self._h_xi_a)) - ((1/self.pi_a_p) * (self._w @ self._h_xi_ap)))
        self._prob = cp.Problem(self._objective, self._constraints) 

    def solve(self, pi):
        if(self._a == 'a0'):
            self.pi_a.value = pi[0] # fixed as constant from the Algorithm 2 loop
            self.pi_a_p.value = pi[1] # fixed as constant from the Algorithm 2 loop
        else:
            self.pi_a.value = pi[1] # fixed as constant from the Algorithm 2 loop
            self.pi_a_p.value = pi[0] # fixed as constant from the Algorithm 2 loop

        #print("OBJECTIVE : 1/{} (w * h({})) - 1/{} (w * h({}))".format(self.pi_0.value, self._a, self.pi_1.value, self._a_p))
        self._prob.solve(solver = self.solver, verbose=False, warm_start = True)
        return self._prob.value, self._w.value, (self._a, self._a_p, (pi[0], pi[1]))

class EoLinearProgram():
    def __init__(self, n, h_pred, a_indices, a, a_p, solver):
        self._w = cp.Variable(n)
        self._a = a
        self._a_p = a_p
        self._a_1 = "{}_{}".format(a, 'y1') 
        self._a_0 = "{}_{}".format(a, 'y0')
        self._a_p_1 = "{}_{}".format(a_p, 'y1')
        self._a_p_0 = "{}_{}".format(a_p, 'y0')
        self.solver = solver

        # Problem constants
        self._h_xi_a_0 = h_pred.copy()
        self._h_xi_a_0[a_indices[self._a_1]] = 0 
        self._h_xi_a_0[a_indices[self._a_p_0]] = 0
        self._h_xi_a_0[a_indices[self._a_p_1]] = 0

        self._h_xi_a_1 = h_pred.copy()
        self._h_xi_a_1[a_indices[self._a_0]] = 0 
        self._h_xi_a_1[a_indices[self._a_p_0]] = 0
        self._h_xi_a_1[a_indices[self._a_p_1]] = 0

        self._h_xi_ap_0 = h_pred.copy()
        self._h_xi_ap_0[a_indices[self._a_0]] = 0 
        self._h_xi_ap_0[a_indices[self._a_1]] = 0
        self._h_xi_ap_0[a_indices[self._a_p_1]] = 0

        self._h_xi_ap_1 = h_pred.copy()
        self._h_xi_ap_1[a_indices[self._a_0]] = 0
        self._h_xi_ap_1[a_indices[self._a_1]] = 0 
        self._h_xi_ap_1[a_indices[self._a_p_0]] = 0

        self.pi_a_0 = cp.Parameter(nonneg=True)
        self.pi_a_1 = cp.Parameter(nonneg=True)
        self.pi_ap_0 = cp.Parameter(nonneg=True)
        self.pi_ap_1 = cp.Parameter(nonneg=True)

        # Constraints
        self._constraints = [
            cp.sum(self._w[a_indices[self._a_0]]) == self.pi_a_0,
            cp.sum(self._w[a_indices[self._a_1]]) == self.pi_a_1,
            cp.sum(self._w[a_indices[self._a_p_0]]) == self.pi_ap_0,
            cp.sum(self._w[a_indices[self._a_p_1]]) == self.pi_ap_1,
            cp.sum(self._w) == 1, # don't EXACTLY sum to 1 sometimes
            0 <= self._w
        ]

        # Objective Function
        # NOTE: @ is dot product between w and the h prediction vector with all a or a_p zero'd out
        self._objective = cp.Maximize(
            (((1/self.pi_a_1) * (self._w @ self._h_xi_a_1)) - ((1/self.pi_ap_1) * (self._w @ self._h_xi_ap_1)))
            +
            (((1/self.pi_a_0) * (self._w @ self._h_xi_a_0)) - ((1/self.pi_ap_0) * (self._w @ self._h_xi_ap_0)))
        )
        self._prob = cp.Problem(self._objective, self._constraints) 

    def solve(self, pi):
        if(self._a == 'a0'):
            self.pi_a_0.value = pi[0] # fixed as constant from the Algorithm 2 loop
            self.pi_ap_0.value = pi[1] # fixed as constant from the Algorithm 2 loop
            self.pi_a_1.value = pi[2] # fixed as constant from the Algorithm 2 loop
            self.pi_ap_1.value = pi[3] # fixed as constant from the Algorithm 2 loop
        elif(self._a == 'a1'):
            self.pi_a_0.value = pi[1] # fixed as constant from the Algorithm 2 loop
            self.pi_ap_0.value = pi[0] # fixed as constant from the Algorithm 2 loop
            self.pi_a_1.value = pi[3] # fixed as constant from the Algorithm 2 loop
            self.pi_ap_1.value = pi[2] # fixed as constant from the Algorithm 2 loop
        else:
            raise ValueError("Incorrect value of a.")

        #print("OBJECTIVE : 1/{} (w * h({})) - 1/{} (w * h({}))".format(self.pi_0.value, self._a_y, self.pi_1.value, self._a_p_y))
        self._prob.solve(solver = self.solver, verbose=False, warm_start = True)
        return self._prob.value, self._w.value, (self._a, self._a_p, pi)

""" 
The Lambda Best Response step that solves an LP to give a single 3-tuple
Lambda back to the h player. This tuple represents the entry in the Lambda
vector that gets updated. The main function here is best_response().

Returns:
lambda_entry: (str, str, nparray). a 3-tuple (a, a_p, w), the argmax over the LP (and over all the pi's),
where a is either 'a0' or 'a1' (string), a_p (read: a prime) is 'a0' or 'a1' (the opposite
of a), and w is the discretized (based on N(gamma_1, W)) weight vector that maximizes the LP.
"""
class LambdaBestResponse:
    def __init__(self, y, h_pred, a_indices, gamma_1, gamma_1_buckets, gamma_2_buckets, epsilon, num_cores, solver, fair_constraint):
        self.y = y
        self.h_pred = np.asarray(h_pred)
        self.a_indices = a_indices
        self.gamma_1 = gamma_1
        self.gamma_1_buckets = gamma_1_buckets
        self.gamma_2_buckets = gamma_2_buckets
        self.epsilon = epsilon
        self.num_cores = num_cores
        self.solver = solver
        self.fair_constraint = fair_constraint

    def best_response(self):
        """
        Returns lambda_entry, the 3-tuple containing (a, a_p, w). a := 'a0' or 'a1'. 
        a_p := 'a1' or 'a0'. w:= discretized weight vector that was argmax of the best LP.

        :return: tuple.
        """
        w_dict = dict()
        val_dict = dict()
        N_gamma_2_A = self.gamma_2_buckets
        a_a_p = list(itertools.permutations(['a0', 'a1'])) 

        start = time.time()
        if(self.fair_constraint == 'dp'):
            solved_results = []
            for (a, a_p) in a_a_p: # either a = 'a0' and a_p = 'a1' or vice versa 
                problem = DpLinearProgram(len(self.h_pred), self.h_pred, self.a_indices, a, a_p, self.solver)
                pool = Pool(processes = self.num_cores)
                solved_results.extend(pool.map(problem.solve, N_gamma_2_A['dp'])) # multiprocessing maps each pi to new process
                pool.close()

        elif(self.fair_constraint == 'eo'):
            solved_results = []
            #for y_prop in ['y0', 'y1']:
            for (a, a_p) in a_a_p:
                problem = EoLinearProgram(len(self.h_pred), self.h_pred, self.a_indices, a, a_p, self.solver)
                pool = Pool(processes = self.num_cores)
                solved_results.extend(pool.map(problem.solve, N_gamma_2_A['eo']))
                pool.close()
        
        else:
            raise ValueError("Invalid fairness constraint. Choose dp or eo.")
        end = time.time()

        # max over the objective values
        max_lp = -1e5
        #argmax_lp = np.zeros
        for result in solved_results:
            if result[0] > max_lp: # result[0] is the objective value of the LP
                max_lp = result[0]
                argmax_lp = result[1] # result[1] is the weight vector from the LP
                optimal_tuple = result[2] # result[2] is a tuple of the form (a0, a1, (pi_0, pi_1)) or (a0, a1, (pi_00, pi_10, pi_01, pi_11))
        
        # Violation of fairness
        if(max_lp > self.epsilon - 4*self.gamma_1):
            optimal_w = argmax_lp
            optimal_w[optimal_w < 0] = 0
            #optimal_w = self._discretize_weights_bsearch(optimal_w) # let w_i be the upper end-point of bucket

            lambda_entry = (optimal_tuple[0], optimal_tuple[1], optimal_w) 
            # return of form ('a0', 'a1', weight_vector) or ('a1', 'a0', weight_vector)
        else:
            lambda_entry = (0, 0, 0)

        return lambda_entry

    '''
    ### NOT ACTUALLY NEEDED IN PRACTICE ###
    def _discretize_weights_bsearch(self, w):
        """
        Returns w, the discretized weight vector based on N(gamma_1, W)

        :return: nparray.
        """
        for i, w_i in enumerate(w):
            bucket = self._binary_search_buckets(w_i)
            w[i] = bucket[1]
        
        return w 
        
    def _binary_search_buckets(self, x):
        """
        Returns bucket, the tuple (w_lower, w_upper) in N(gamma_1, W) that 
        w is in. This is just a helper function for discretize_weights_bsearch().

        :return: tuple.
        """
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
    '''
