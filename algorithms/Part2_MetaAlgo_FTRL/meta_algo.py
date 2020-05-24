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
    def __init__(self, T, T_inner, eta, eta_inner, card_A = 2, epsilon = 0.05, num_cores = 2, solver = 'ECOS',
                B = 1, gamma_1 = 0.001, gamma_2 = 0.05, fair_constraint='eo', gp_wt_bd=0.1):
        self.T = T
        self.T_inner = T_inner
        self.card_A = card_A
        self.B = B
        self.eta = eta
        self.eta_inner = eta_inner
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.epsilon = epsilon
        self.num_cores = num_cores
        self.solver = solver
        self.fair_constraint = fair_constraint
        self.gp_wt_bd = gp_wt_bd

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
        print("gp_wt_bd=" + str(self.gp_wt_bd))
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
        print("Number of gamma_1_buckets {}".format(len(gamma_1_buckets)))
                            
        return gamma_1_buckets

    def _gamma_2_buckets(self, y, proportions):
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
            #if self.lbd_dp_wt <= bucket and bucket <= 1.0 - self.lbd_dp_wt:
            dp_gamma_2_buckets.append(bucket)

        dp_N_gamma_2_A  = []
        for pi_a in dp_gamma_2_buckets:
            pi_ap = 1 - pi_a
            # if(self.lbd_dp_wt <= pi_a and pi_a <= 1.0 - self.lbd_dp_wt 
            # and self.lbd_dp_wt <= pi_ap and pi_ap <= 1.0 - self.lbd_dp_wt):
            if(proportions['a0'] - self.gp_wt_bd <= pi_a and pi_a <= proportions['a0'] + self.gp_wt_bd 
            and proportions['a1'] - self.gp_wt_bd <= pi_ap and pi_ap <= proportions['a1'] + self.gp_wt_bd):
                dp_N_gamma_2_A.append((pi_a, pi_ap))

        N_gamma_2_A['dp'] = dp_N_gamma_2_A

        ### Equalized Odds Y0 (eo_y0) buckets ###
        eo_y0_gamma_2_num_buckets = np.ceil(math.log((proportions['y0']/delta_2), 1 + self.gamma_2))
        eo_y0_gamma_2_buckets = []
        for j in range(int(eo_y0_gamma_2_num_buckets)):
            bucket = (delta_2) * (1 + self.gamma_2)**j
            #if proportions['a0_y0'] - self.gp_wt_bd <= bucket and bucket <= proportions['a0_y0'] + self.gp_wt_bd:
            eo_y0_gamma_2_buckets.append(bucket)
            
        eo_y0_N_gamma_2_A = []
        for pi_a in eo_y0_gamma_2_buckets:
            pi_ap = proportions['y0'] - pi_a
            #if(self.lbd_eo_wt <= pi_a and pi_a <= proportions['y0'] - self.lbd_eo_wt 
            #and self.lbd_eo_wt <= pi_ap and pi_ap <= proportions['y0'] - self.lbd_eo_wt):
            if(proportions['a0_y0'] - self.gp_wt_bd <= pi_a and pi_a <= proportions['a0_y0'] + self.gp_wt_bd 
            and proportions['a1_y0'] - self.gp_wt_bd <= pi_ap and pi_ap <= proportions['a1_y0'] + self.gp_wt_bd):
                eo_y0_N_gamma_2_A.append((pi_a, pi_ap))
        
        N_gamma_2_A['eo_y0'] = eo_y0_N_gamma_2_A

        ### Equalized Odds Y1 (eo_y1) buckets ###
        eo_y1_gamma_2_num_buckets = np.ceil(math.log((proportions['y1']/delta_2), 1 + self.gamma_2))
        eo_y1_gamma_2_buckets = []
        for j in range(int(eo_y1_gamma_2_num_buckets)):
            bucket = (delta_2) * (1 + self.gamma_2)**j
            #if proportions['a0_y1'] - self.gp_wt_bd <= bucket and bucket <= proportions['a0_y1'] + self.gp_wt_bd:
            eo_y1_gamma_2_buckets.append(bucket)
        
        eo_y1_N_gamma_2_A = []
        for pi_a in eo_y1_gamma_2_buckets:
            pi_ap = proportions['y1'] - pi_a
            #if(self.lbd_eo_wt <= pi_a and pi_a <= proportions['y1'] - self.lbd_eo_wt 
            #and self.lbd_eo_wt <= pi_ap and pi_ap <= proportions['y1'] - self.lbd_eo_wt):
            if(proportions['a0_y1'] - self.gp_wt_bd <= pi_a and pi_a <= proportions['a0_y1'] + self.gp_wt_bd 
            and proportions['a1_y1'] - self.gp_wt_bd <= pi_ap and pi_ap <= proportions['a1_y1'] + self.gp_wt_bd):
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

    def _project_W(self, w, a_indices, y, proportions):
        """
        Project w back onto the feasible set of weights

        :return: nparray 'x.value' which is the projected weight vector.
        """
        x = cp.Variable(len(w))
        objective = cp.Minimize(cp.sum_squares(w - x))
        constraints = [0 <= x, 
                        x <= 1, 
                        cp.sum(x) == 1]  # extra constraint for non-trivial distributions
        
        '''
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
            constraints.append(cp.sum(x[a_indices['a0_y0']]) <= prop_y0 - self.lbd_eo_wt) 
            constraints.append(cp.sum(x[a_indices['a1_y0']]) <= prop_y0 - self.lbd_eo_wt)
            constraints.append(cp.sum(x[a_indices['a0_y1']]) <= prop_y1 - self.lbd_eo_wt)
            constraints.append(cp.sum(x[a_indices['a1_y1']]) <= prop_y1 - self.lbd_eo_wt)
        '''

        if(self.fair_constraint == 'dp'):
            constraints.append(cp.sum(x[a_indices['a0']]) >= proportions['a0'] - self.gp_wt_bd)
            constraints.append(cp.sum(x[a_indices['a1']]) >= proportions['a1'] - self.gp_wt_bd)
            constraints.append(cp.sum(x[a_indices['a0']]) <= proportions['a0'] + self.gp_wt_bd)
            constraints.append(cp.sum(x[a_indices['a1']]) <= proportions['a1'] + self.gp_wt_bd)
        elif(self.fair_constraint == 'eo'):
            constraints.append(cp.sum(x[a_indices['a0_y0']]) >= proportions['a0_y0'] - self.gp_wt_bd)
            constraints.append(cp.sum(x[a_indices['a1_y0']]) >= proportions['a1_y0'] - self.gp_wt_bd)
            constraints.append(cp.sum(x[a_indices['a0_y1']]) >= proportions['a0_y1'] - self.gp_wt_bd)
            constraints.append(cp.sum(x[a_indices['a1_y1']]) >= proportions['a1_y1'] - self.gp_wt_bd)
            constraints.append(cp.sum(x[a_indices['a0_y0']]) <= proportions['a0_y0'] + self.gp_wt_bd) 
            constraints.append(cp.sum(x[a_indices['a1_y0']]) <= proportions['a1_y0'] + self.gp_wt_bd)
            constraints.append(cp.sum(x[a_indices['a0_y1']]) <= proportions['a0_y1'] + self.gp_wt_bd)
            constraints.append(cp.sum(x[a_indices['a1_y1']]) <= proportions['a1_y1'] + self.gp_wt_bd)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.solver, verbose=False)

        if prob.status in ["infeasible", "unbounded"]:
            raise(cp.SolverError("project_W failed to find feasible solution."))

        return x.value
    
    def _update_w(self, X, y, a_indices, prev_h_t, w, proportions):
        loss_vec = self._zero_one_loss_grad_w(prev_h_t.predict(X), y)
        w_t = w + self.eta * loss_vec 
        w_t = self._project_W(w_t, a_indices, y, proportions)
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

    def _set_proportions(self, a_indices, y):
        proportions = {}
        proportions['a0'] = len(a_indices['a0'])/float(len(y))
        proportions['a1'] = len(a_indices['a1'])/float(len(y))
        proportions['a0_y0'] = len(a_indices['a0_y0'])/float(len(y))
        proportions['a0_y1'] = len(a_indices['a0_y1'])/float(len(y))
        proportions['a1_y0'] = len(a_indices['a1_y0'])/float(len(y))
        proportions['a1_y1'] = len(a_indices['a1_y1'])/float(len(y))
        proportions['y0'] = (len(np.where(y == 0)[0]))/float(len(y))
        proportions['y1'] = (len(np.where(y == 1)[0]))/float(len(y))

        print('y0 proportion = {}'.format(proportions['y0']))
        print('y1 proportion = {}'.format(proportions['y1']))
        
        if(self.fair_constraint == 'dp'):
            print('a0 proportion = {}'.format(proportions['a0']))
            print('a1 proportion = {}'.format(proportions['a1']))
        elif(self.fair_constraint == 'eo'):
            print('a0 y0 proportion = {}'.format(proportions['a0_y0']))
            print('a1 y0 proportion = {}'.format(proportions['a1_y0']))
            print('a0 y1 proportion = {}'.format(proportions['a0_y1']))
            print('a1 y1 proportion = {}'.format(proportions['a1_y1']))

        assert(proportions['y0'] + proportions['y1'] == 1)
        assert(proportions['a0'] + proportions['a1'] == 1)
        assert(proportions['a0_y0'] + proportions['a0_y1'] + proportions['a1_y0'] + proportions['a1_y1'] == 1)

        return proportions

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
        print("Number of examples = {}".format(len(X)))
        a_indices = self._set_a_indices(sensitive_features, y) # dictionary with a value information

        # calculate proportions in the training data
        proportions = self._set_proportions(a_indices, y)

        w = np.full((X.shape[0],), 1/X.shape[0]) # each weight starts as uniform 1/n
        gamma_1_buckets = self._gamma_1_buckets(X)
        gamma_2_buckets = self._gamma_2_buckets(y, proportions)

        # Start off with oracle prediction over uniform weights
        print("=== Initializing h_0... ===")
        oracle = BayesianOracle(X, y, X_test, y_test, w, sensitive_features, sensitive_features_test,
                                a_indices,
                                self.card_A, 
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
                                0)
        h_t, inner_hypotheses_t = oracle.execute_oracle() # t = 0

        hypotheses = []
        hypotheses.extend(inner_hypotheses_t)
        start_outer = time.time()
        print("=== ALGORITHM 1 EXECUTION ===")
        for t in range(self.T):
            start_inner = time.time()

            '''
            # compute the loss of each of the T_inner classifiers (to avg. over)
            T_inner_sum_loss = np.zeros(len(X))
            for h in inner_hypotheses_t:
                T_inner_sum_loss += self._zero_one_loss_grad_w(h.predict(X), y)
            
            w += self.eta * (1/len(inner_hypotheses_t)) * T_inner_sum_loss # avg. over the T_inner classifiers
            w = self._project_W(w, a_indices, y)
            '''
            w = self._update_w(X, y, a_indices, h_t, w, proportions)
            oracle = BayesianOracle(X, y, X_test, y_test, w, sensitive_features, sensitive_features_test,
                                a_indices,
                                self.card_A, 
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
                                t + 1) # just to print which outer loop T we're on
            
            h_t, inner_hypotheses_t = oracle.execute_oracle()
            hypotheses.extend(inner_hypotheses_t) # concatenate all of the inner loop hypotheses 

            end_inner = time.time()
            print("ALGORITHM 1 (Meta Algorithm) Loop " + str(t  + 1) + " Completed!")
            print("ALGORITHM 1 (Meta Algorithm) Time/loop: " + str(end_inner - start_inner))
        
        end_outer = time.time()
        print("ALGORITHM 1 (Meta Algorithm) Total Execution Time: " + str(end_outer - start_outer))
        return hypotheses, VotingClassifier(hypotheses) 
