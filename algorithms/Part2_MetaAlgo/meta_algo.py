import cvxpy as cp 
import numpy as np
import math
import time 
import itertools
from bayesian_oracle import BayesianOracle
from voting_classifier import VotingClassifier

""" The Meta-Algorithm (Algorithm 1) that performs gradient descent on the weights and  
calls the Bayesian oracle at each time step t. The main function here is meta_algorithm().

:param T: the number of steps to run the Meta-Algo
:type int

:param T_inner: the number of steps to run the Bayesian Oracle
:type int:

:param card_A: the cardinality of A, the set of protected attributes
:type int:

:param M: the bound on the loss
:type float:

:param epsilon: the desired "fairness gap" between the protected attributes
:type float:

:param num_cores: the number of cores to use for multiprocessing in LPs
:type int:

:param solver: the LP solver designated
:type str:

:param B: the bound on each Lambda 
:type float:

:param eta: the learning/gradient descent rate for the weights in Meta-Algo
:type float:

:param gamma_1: a parameter that sets how many "buckets" to discretize the weight 
vectors in the Bayesian Oracle and Lambda Best Response step.
:type float:

:param gamma_2: a parameter that sets how many "buckets" to discretize the weight 
vectors pi in the Lambda Best Response step.
:type float:
"""

class MetaAlgorithm:
    def __init__(self, T, T_inner, card_A = 2, M = 1, epsilon = 0.05, num_cores = 2, solver = 'ECOS',
                B = 10, eta = 0.05, gamma_1 = 0.01, gamma_2 = 0.05, constraint_used='dp'):
        self.T = T
        self.T_inner = T_inner
        self.card_A = card_A
        self.B = B
        self.eta = eta
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.M = M
        self.epsilon = epsilon
        self.num_cores = num_cores
        self.solver = solver
        self.constraint_used = constraint_used

        if(self.epsilon - 4 * self.gamma_1 < 0):
            raise(ValueError("epsilon - 4 * gamma_1 must be positive for LPs."))
        if eta is None:
            self.eta = 1/np.sqrt(2*T)
        
        print("=== HYPERPARAMETERS ===")
        print("T=" + str(self.T))
        print("T_inner=" + str(self.T_inner))
        print("B=" + str(self.B))
        print("eta=" + str(self.eta))
        print("epsilon=" + str(self.epsilon))
        print("Cores in use=" + str(self.num_cores))
        print("Fairness Definition=" + str(self.constraint_used))

    def _gamma_1_buckets(self, X):
        """
        Returns the discretized buckets for each weight vector N(gamma_1, W).

        :return: list 'N_gamma_1_W' of 2-tuples for the range of each bucket.
        """
        delta_1 = (2 * len(X)) / self.gamma_1

        gamma_1_num_buckets = int(np.ceil(math.log(delta_1, 1 + self.gamma_1)))
        N_gamma_1_W = []
        N_gamma_1_W.append((0, 1/delta_1))
        for i in range(gamma_1_num_buckets):
            bucket_lower = ((1 + self.gamma_1) ** i) * (1/delta_1)
            bucket_upper = ((1 + self.gamma_1) ** (i + 1)) * (1/delta_1)
            N_gamma_1_W.append((bucket_lower, bucket_upper))
                
        return N_gamma_1_W

    def _gamma_2_buckets(self):
        """
        Returns the pi_a0 and pi_a1 for the LPs, N(gamma_2, A). Number of LPs
        depends on this.

        :return: list 'N_gamma_2_A' of 2-tuples for pairs of pi_a0 and pi_a1.
        """
        delta_2 = 0.05 

        gamma_2_num_buckets = np.ceil(math.log((1/delta_2), 1 + self.gamma_2)) 
        gamma_2_buckets = []
        for j in range(int(gamma_2_num_buckets)):
            bucket = (delta_2) * (1 + self.gamma_2)**j
            gamma_2_buckets.append(bucket)

        N_gamma_2_A  = []
        for pi_a in gamma_2_buckets:
            pi_ap = 1 - pi_a
            N_gamma_2_A.append((pi_a, pi_ap))
                        
        return N_gamma_2_A


    def _zero_one_loss_grad_w(self, pred, y):
        """
        Returns the zero one loss for each sample (gradient w.r.t. w)

        :return: nparray 'loss_vec' which is zero one loss for each training instance.
        """
        loss_vec = []
        for (i,y_true) in enumerate(y):
            if(y_true == pred[i]):
                loss_vec.append(0)
            else:
                loss_vec.append(1)
                
        return np.asarray(loss_vec)

    def _project_W(self, w):
        """
        Project w back onto the feasible set of weights

        :return: nparray 'x.value' which is the projected weight vector.
        """
        x = cp.Variable(len(w))
        objective = cp.Minimize(cp.sum_squares(w - x))
        constraints = [0 <= x, 
                        x <= 1, 
                        cp.sum(x) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.solver, verbose=False)

        if prob.status in ["infeasible", "unbounded"]:
            raise(cp.SolverError("project_W failed to find feasible solution."))

        return x.value

    def _set_a_indices(self, sensitive_features):
        """
        Creates a dictionary a_indices that contains the necessary information for which indices
        contain the sensitive/protected attributes.

        :return: dict 'a_indices' which contains a list of the a_0 indices, list of a_1 indices, and
        a list containing the a value of each sample.
        """
        sensitive_features = sensitive_features.replace('African-American', 0)
        sensitive_features = sensitive_features.replace('Caucasian', 1)
        sensitive_features = sensitive_features.replace('Female', 0)
        sensitive_features = sensitive_features.replace('Male', 1)

        a_indices = dict()
        a_indices['a0'] = sensitive_features.index[sensitive_features.eq(0)].tolist()
        a_indices['a1'] = sensitive_features.index[sensitive_features.eq(1)].tolist()
        a_indices['all'] = sensitive_features.tolist()
        return a_indices

    def meta_algorithm(self, X, y, sensitive_features, X_test, y_test, sensitive_features_test):
        """
        Runs the meta-algorithm, calling the bayesian_oracle at each time step (which itself calls
        the lambda_best_response_param_parallel). Meta-algorithm runs for T steps, and the Bayesian oracle
        runs for T_inner many steps. 
        
        NOTE: X, y, sensitive_features are given as Pandas dataframes here. Also note that
        X_test, y_test are only used to get some statistics on how well each Bayesian oracle 
        step is doing w.r.t. the fairness constraint.

        :return: 
        list 'hypotheses' the actual list of (T_inner * T) hypotheses
        VotingClassifier, an object that takes a majority vote over (T_inner * T) hypotheses
        """
         # dp, eo
        a_indices = self._set_a_indices(sensitive_features) # dictionary with a value information
        w = np.full((X.shape[0],), 1/X.shape[0]) # each weight starts as uniform 1/n
        gamma_1_buckets = self._gamma_1_buckets(X)
        gamma_2_buckets = self._gamma_2_buckets()

        # Start off with oracle prediction over uniform weights
        print("=== Initializing h_0... ===")
        oracle = BayesianOracle(X, y, X_test, y_test, w, sensitive_features, sensitive_features_test,
                                a_indices,
                                self.card_A, 
                                self.M, 
                                self.B, 
                                self.T_inner,
                                self.gamma_1,
                                gamma_1_buckets, 
                                gamma_2_buckets, 
                                self.epsilon,
                                self.eta,
                                self.num_cores,
                                self.solver,
                                self.constraint_used,
                                0)
        h_t, inner_hypotheses_t = oracle.execute_oracle()

        hypotheses = []
        start_outer = time.time()
        print("=== ALGORITHM 1 EXECUTION ===")
        for t in range(self.T):
            start_inner = time.time()

            # compute the loss of each of the T_inner classifiers (to avg. over)
            T_inner_sum_loss = np.zeros(len(X))
            for h in inner_hypotheses_t:
                T_inner_sum_loss += self._zero_one_loss_grad_w(h.predict(X), y)
            
            w += self.eta * (1/len(inner_hypotheses_t)) * T_inner_sum_loss # avg. over the T_inner classifiers
            w = self._project_W(w)
            oracle = BayesianOracle(X, y, X_test, y_test, w, sensitive_features, sensitive_features_test,
                                a_indices,
                                self.card_A, 
                                self.M, 
                                self.B, 
                                self.T_inner,
                                self.gamma_1,
                                gamma_1_buckets, 
                                gamma_2_buckets, 
                                self.epsilon,
                                self.eta,
                                self.num_cores,
                                self.solver,
                                self.constraint_used,
                                t + 1) # just to print which outer loop T we're on
            
            h_t, inner_hypotheses_t = oracle.execute_oracle()
            hypotheses.extend(inner_hypotheses_t) # concatenate all of the inner loop hypotheses 

            end_inner = time.time()
            print("ALGORITHM 1 (Meta Algorithm) Loop " + str(t  + 1) + " Completed!")
            print("ALGORITHM 1 (Meta Algorithm) Time/loop: " + str(end_inner - start_inner))
        
        end_outer = time.time()
        print("ALGORITHM 1 (Meta Algorithm) Total Execution Time: " + str(end_outer - start_outer))
        return hypotheses, VotingClassifier(hypotheses) 
