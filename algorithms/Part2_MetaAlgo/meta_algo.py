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
    def __init__(self, T, T_inner, eta, eta_inner, card_A = 2, M = 1, epsilon = 0.05, num_cores = 2, solver = 'ECOS', B = 10, 
                gamma_1 = 0.01, gamma_2 = 0.05, fair_constraint='dp', lbd_dp_wt=0.35, lbd_eo_wt=0.15, ubd_dp_wt=1.0, ubd_eo_wt=1.0):
        self.T = T
        self.T_inner = T_inner
        self.card_A = card_A
        self.B = B
        self.eta = eta
        self.eta_inner = eta_inner
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.M = M
        self.epsilon = epsilon
        self.num_cores = num_cores
        self.solver = solver
        self.fair_constraint = fair_constraint
        self.lbd_dp_wt = lbd_dp_wt
        self.lbd_eo_wt = lbd_eo_wt
        self.ubd_dp_wt = ubd_dp_wt
        self.ubd_eo_wt = ubd_eo_wt

        if(self.epsilon - 4 * self.gamma_1 < 0):
            raise(ValueError("epsilon - 4 * gamma_1 must be positive for LPs."))
        if(self.fair_constraint not in ['dp', 'eo']):
            raise(ValueError("Fairness constraint must be either dp or eo."))
        if eta is None:
            self.eta = 1/np.sqrt(2*self.T)
        
        print("=== HYPERPARAMETERS ===")
        print("T=" + str(self.T))
        print("T_inner=" + str(self.T_inner))
        print("B=" + str(self.B))
        print("eta=" + str(self.eta))
        print("eta_inner=" + str(self.eta_inner))
        print("epsilon=" + str(self.epsilon))
        print("lbd_dp_wt=" + str(self.lbd_dp_wt))
        print("lbd_eo_wt=" + str(self.lbd_eo_wt))
        print("ubd_dp_wt=" + str(self.ubd_dp_wt))
        print("ubd_eo_wt=" + str(self.ubd_eo_wt))
        print("Cores in use=" + str(self.num_cores))
        print("Fairness Definition=" + str(self.fair_constraint))

    def _gamma_1_buckets(self, X):
        """
        Returns the discretized buckets for each weight vector in N(gamma_1, W).

        :return: list 'gamma_1_buckets' of 2-tuples for the range of each bucket.
        """
        delta_1 = self.gamma_1 / (2 * len(X))

        gamma_1_num_buckets = int(np.ceil(math.log((1/delta_1), 1 + self.gamma_1)))
        gamma_1_buckets = []
        gamma_1_buckets.append((0, delta_1))
        for i in range(gamma_1_num_buckets):
            bucket_lower = ((1 + self.gamma_1) ** i) * (delta_1)
            bucket_upper = ((1 + self.gamma_1) ** (i + 1)) * (delta_1)
            gamma_1_buckets.append((bucket_lower, bucket_upper))
        
        print("First 5 entries of N_gamma_1:")
        print(gamma_1_buckets[:4])
                            
        return gamma_1_buckets

    def _gamma_2_buckets(self, y):
        """
        Returns the pi_a0 and pi_a1 for the LPs, N(gamma_2, A). Number of LPs
        depends on this.

        :return: dict 'N_gamma_2_A' of lists, indexed by 'dp,' 'eo_y0,' and 'eo_y1.' 
        each value in this dict is a list of pi_a0 and pi_a1 tuples.
        """
        N_gamma_2_A = {}
        delta_2 = 0.05 

        ### Demographic Parity (dp) buckets ###
        dp_gamma_2_num_buckets = np.ceil(math.log((1/delta_2), 1 + self.gamma_2)) 
        dp_gamma_2_buckets = []
        for j in range(int(dp_gamma_2_num_buckets)):
            bucket = (delta_2) * (1 + self.gamma_2)**j
            if bucket >= self.lbd_dp_wt and bucket <= 1.0 - self.lbd_dp_wt:
                dp_gamma_2_buckets.append(bucket)

        dp_N_gamma_2_A  = []
        for pi_a in dp_gamma_2_buckets:
            pi_ap = 1 - pi_a
            if(self.lbd_dp_wt <= pi_a and pi_a <= self.ubd_dp_wt 
            and self.lbd_dp_wt <= pi_ap and pi_ap <= self.ubd_dp_wt):
                dp_N_gamma_2_A.append((pi_a, pi_ap))

        N_gamma_2_A['dp'] = dp_N_gamma_2_A

        ### Compute proportion of y0 and y1 in the training data ###
        prop_y0 = (len(np.where(y == 0)[0]))/float(len(y))
        prop_y1 = (len(np.where(y == 1)[0]))/float(len(y))
        assert(prop_y0 + prop_y1 == 1)

        ### Equalized Odds Y0 (eo_y0) buckets ###
        eo_y0_gamma_2_num_buckets = np.ceil(math.log((prop_y0/delta_2), 1 + self.gamma_2))
        eo_y0_gamma_2_buckets = []
        for j in range(int(eo_y0_gamma_2_num_buckets)):
            bucket = (delta_2) * (1 + self.gamma_2)**j
            if bucket >= self.lbd_eo_wt and bucket <= prop_y0 - self.lbd_eo_wt:
                eo_y0_gamma_2_buckets.append(bucket)
        
        eo_y0_N_gamma_2_A = []
        for pi_a in eo_y0_gamma_2_buckets:
            pi_ap = prop_y0 - pi_a
            if(self.lbd_eo_wt <= pi_a and pi_a <= self.ubd_eo_wt 
            and self.lbd_eo_wt <= pi_ap and pi_ap <= self.ubd_eo_wt):
                eo_y0_N_gamma_2_A.append((pi_a, pi_ap))
        
        N_gamma_2_A['eo_y0'] = eo_y0_N_gamma_2_A

        ### Equalized Odds Y1 (eo_y1) buckets ###
        eo_y1_gamma_2_num_buckets = np.ceil(math.log((prop_y1/delta_2), 1 + self.gamma_2))
        eo_y1_gamma_2_buckets = []
        for j in range(int(eo_y1_gamma_2_num_buckets)):
            bucket = (delta_2) * (1 + self.gamma_2)**j
            if bucket >= self.lbd_eo_wt and bucket <= prop_y1 - self.lbd_eo_wt:
                eo_y1_gamma_2_buckets.append(bucket)
        
        eo_y1_N_gamma_2_A = []
        for pi_a in eo_y1_gamma_2_buckets:
            pi_ap = prop_y0 - pi_a
            if(self.lbd_eo_wt <= pi_a and pi_a <= self.ubd_eo_wt 
            and self.lbd_eo_wt <= pi_ap and pi_ap <= self.ubd_eo_wt):
                eo_y1_N_gamma_2_A.append((pi_a, pi_ap))
        
        N_gamma_2_A['eo_y1'] = eo_y1_N_gamma_2_A

        if(self.fair_constraint == 'dp'):
            print("N(gamma_2, A) constraints:")
            print(N_gamma_2_A['dp'])
        elif(self.fair_constraint == 'eo'):
            print("N(gamma_2, A) constraints for Y = 0:")
            print(N_gamma_2_A['eo_y0'])
            print("N(gamma_2, A) constraints for Y = 1:")
            print(N_gamma_2_A['eo_y1'])
                        
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

    def _project_W(self, w, a_indices):
        """
        Project w back onto the feasible set of weights

        :return: nparray 'x.value' which is the projected weight vector.
        """
        x = cp.Variable(len(w))
        objective = cp.Minimize(cp.sum_squares(w - x))
        constraints = [0 <= x, 
                        x <= 1, 
                        cp.sum(x) == 1]

        if(self.fair_constraint == 'dp'):
            constraints.append(cp.sum(x[a_indices['a0']]) >= self.lbd_dp_wt)
            constraints.append(cp.sum(x[a_indices['a1']]) >= self.lbd_dp_wt)
            constraints.append(cp.sum(x[a_indices['a0']]) <= self.ubd_dp_wt)
            constraints.append(cp.sum(x[a_indices['a1']]) <= self.ubd_dp_wt)
        elif(self.fair_constraint == 'eo'):
            constraints.append(cp.sum(x[a_indices['a0_y0']]) >= self.lbd_eo_wt)
            constraints.append(cp.sum(x[a_indices['a1_y0']]) >= self.lbd_eo_wt)
            constraints.append(cp.sum(x[a_indices['a0_y1']]) >= self.lbd_eo_wt)
            constraints.append(cp.sum(x[a_indices['a1_y1']]) >= self.lbd_eo_wt)
            constraints.append(cp.sum(x[a_indices['a0_y0']]) <= self.ubd_eo_wt)
            constraints.append(cp.sum(x[a_indices['a1_y0']]) <= self.ubd_eo_wt)
            constraints.append(cp.sum(x[a_indices['a0_y1']]) <= self.ubd_eo_wt)
            constraints.append(cp.sum(x[a_indices['a1_y1']]) <= self.ubd_eo_wt)
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.solver, verbose=False)

        if prob.status in ["infeasible", "unbounded"]:
            raise(cp.SolverError("project_W failed to find feasible solution."))
        
        return x.value
    
    def _update_w(self, X, y, a_indices, prev_h_t, w):
        # loss_vec = self._zero_one_loss_grad_w(prev_h_t.predict(X), y)
        w_t = w + self.eta * loss_vec 
        w_t = self._project_W(w_t, a_indices)
        return w_t

    def _set_a_indices(self, sensitive_features, y):
        """
        Creates a dictionary a_indices that contains the necessary information for which indices
        contain the sensitive/protected attributes.

        :return: dict 'a_indices' which contains a list of the a_0 indices, list of a_1 indices,
        list of a_0 indices where y = 0, list of a_0 indices where y = 1, list of a_1 indices
        where y = 0, list of a_1 indices where y = 1, and a list containing the a value of each sample.
        """
        a_indices = dict()
        a_indices['a0'] = sensitive_features.index[sensitive_features.eq(0)].tolist()
        a_indices['a1'] = sensitive_features.index[sensitive_features.eq(1)].tolist()
        a_indices['all'] = sensitive_features.tolist()

        y0 = set(np.where(y == 0)[0])
        y1 = set(np.where(y == 1)[0])
        a_indices['a0_y0'] = list(y0.intersection(set(a_indices['a0'])))
        a_indices['a0_y1'] = list(y1.intersection(set(a_indices['a0'])))
        a_indices['a1_y0'] = list(y0.intersection(set(a_indices['a1'])))
        a_indices['a1_y1'] = list(y1.intersection(set(a_indices['a1'])))

        assert(len(a_indices['a0']) + len(a_indices['a1']) == len(y))
        assert(len(a_indices['a0_y0']) + len(a_indices['a0_y1']) + len(a_indices['a1_y0']) + len(a_indices['a1_y1']) == len(y))
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
        a_indices = self._set_a_indices(sensitive_features, y) # dictionary with a value information
        w = np.full((X.shape[0],), 1/X.shape[0]) # each weight starts as uniform 1/n
        gamma_1_buckets = self._gamma_1_buckets(X)
        gamma_2_buckets = self._gamma_2_buckets(y)

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
                                self.eta_inner,
                                self.num_cores,
                                self.solver,
                                self.fair_constraint,
                                self.lbd_dp_wt,
                                self.lbd_eo_wt,
                                self.ubd_dp_wt,
                                self.ubd_eo_wt,
                                0)
        h_t, inner_hypotheses_t = oracle.execute_oracle() # t = 0

        hypotheses = []
        start_outer = time.time()
        print("=== ALGORITHM 1 EXECUTION ===")
        for t in range(self.T):
            start_inner = time.time()

            '''
            # compute the loss of each of the T_inner classifiers (to avg. over)
            T_inner_sum_loss = np.zeros(len(X))
            for h in inner_hypotheses_t:
                T_inner_sum_loss += self._zero_one_loss_grad_w(h.predict(X), y)
            print(T_inner_sum_loss[:100])
            T_inner_sum_loss = (1/len(inner_hypotheses_t)) * T_inner_sum_loss
            print(T_inner_sum_loss[:100])
            w = w + self.eta * T_inner_sum_loss # avg. over the T_inner classifiers
            w = self._project_W(w, a_indices)
            '''
            w = self._update_w(X, y, a_indices, h_t, w)
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
                                self.eta_inner,
                                self.num_cores,
                                self.solver,
                                self.fair_constraint,
                                self.lbd_dp_wt,
                                self.lbd_eo_wt,
                                self.ubd_dp_wt,
                                self.ubd_eo_wt,
                                t + 1) # just to print which outer loop T we're on
            
            h_t, inner_hypotheses_t = oracle.execute_oracle()
            hypotheses.extend(inner_hypotheses_t) # concatenate all of the inner loop hypotheses 

            end_inner = time.time()
            print("ALGORITHM 1 (Meta Algorithm) Loop " + str(t  + 1) + " Completed!")
            print("ALGORITHM 1 (Meta Algorithm) Time/loop: " + str(end_inner - start_inner))
        
        end_outer = time.time()
        print("ALGORITHM 1 (Meta Algorithm) Total Execution Time: " + str(end_outer - start_outer))
        return hypotheses, VotingClassifier(hypotheses) 
