import cvxpy as cp 
import numpy as np
import math
import time 
import itertools
from bayesian_oracle import BayesianOracle
from voting_classifier import VotingClassifier

class MetaAlgorithm:
    """ The Meta-Algorithm (Algorithm 1) that performs gradient descent on the weights and  
    calls the Bayesian oracle at each time step t.
    
    :param T: the number of steps to run the Meta-Algo
    :type int

    :param T_inner: the number of steps to run the Bayesian Oracle
    :type int:

    :param card_A: the cardinality of A, the set of protected attributes
    :type int:

    :param nu: the desired accuracy to the Bayesian oracle guarantee (Theorem 4)
    :type float:

    :param M: the bound on the loss
    :type float:

    :param epsilon: the desired "fairness gap" between the protected attributes
    :type float:

    :param B: the bound on each Lambda 
    :type float:

    :param eta: the learning/gradient descent rate for the weights in Meta-Algo
    :type float:

    :param gamma_1: a parameter that sets how many "buckets" to discretize the weight 
    vectors in the Bayesian Oracle and Lambda Best Response step. Depends on nu.
    :type float:

    :param gamma_2: a parameter that sets how many "buckets" to discretize the weight 
    vectors pi in the Lambda Best Response step. Depends on nu.
    :type float:
    """

    def __init__(self, T, T_inner, card_A = 2, nu = 0.01, M = 1, epsilon = 0.1, num_cores = 2, solver = 'ECOS',
                B = None, eta = None, gamma_1 = None, gamma_2 = None):
        self.T = T
        self.T_inner = T_inner
        self.card_A = card_A
        self.nu = nu
        self.B = B
        self.eta = eta
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.M = M
        self.epsilon = epsilon
        self.num_cores = num_cores
        self.solver = solver
        
        if B is None:
            self.B = 10
        if eta is None:
            self.eta = 1/np.sqrt(2*T)
        if gamma_1 is None:
            self.gamma_1 = nu/self.B
        if gamma_2 is None:
            self.gamma_2 = nu/self.B
        
        print("=== HYPERPARAMETERS ===")
        print("T=" + str(self.T))
        print("T_inner=" + str(self.T_inner))
        print("B=" + str(self.B))
        print("eta=" + str(self.eta))
        print("Cores in use=" + str(self.num_cores))

    def _gamma_1_buckets(self, X):
        """
        Returns the discretized buckets for each weight vector.

        :return: list 'gamma_1_buckets' of 2-tuples describing the range of each bucket.
        """
        # Initialize N(gamma_1, W)
        delta_1 = (2 * len(X)) / self.gamma_1

        gamma_1_num_buckets = np.ceil(math.log(delta_1, 1 + self.gamma_1))
        gamma_1_buckets = []
        gamma_1_buckets.append((0, 1/delta_1))
        for i in range(int(gamma_1_num_buckets)):
            bucket_lower = ((1 + self.gamma_1) ** i) * (1/delta_1)
            bucket_upper = ((1 + self.gamma_1) ** (i + 1)) * (1/delta_1)
            gamma_1_buckets.append((bucket_lower, bucket_upper))
                
        return gamma_1_buckets

    def _gamma_2_buckets(self):
        delta_2 = 0.05 # FIXED

        gamma_2_num_buckets = np.ceil(math.log((1/delta_2), 1 + self.gamma_2)) 
        gamma_2_buckets = []
        for j in range(int(gamma_2_num_buckets)):
            bucket = (delta_2) * (1 + self.gamma_2)**j
            gamma_2_buckets.append(bucket)

        pi  = []
        for pi_a in gamma_2_buckets:
            pi_ap = 1 - pi_a
            pi.append((pi_a, pi_ap))
        
        '''
        N_gamma_2_A = []
        for i in range(len(pi)):
            if (((1 - (2 * self.gamma_2)) < pi[i][0] + pi[i][1] < (1 + (2 * self.gamma_2))) and
            pi[i][0] >= 0.1 and
            pi[i][1] >= 0.1):
                N_gamma_2_A.append(pi[i])
        '''
        N_gamma_2_A = pi
                
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
        Solves an LP to project w back onto the feasible set of weights

        :return: nparray 'x.value' which is the projected weight vector.
        """
        x = cp.Variable(len(w))
        objective = cp.Minimize(cp.sum_squares(w - x))
        constraints = [0 <= x, 
                        x <= 1, 
                        cp.sum(x) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.solver, verbose=False)
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

    def meta_algorithm(self, X, y, sensitive_features):

        """
        Runs the meta-algorithm, calling the Bayesian oracle at each time step (which itself calls
        the Lambda Best Response algorithm). Meta-algorithm runs for T steps, and the Bayesian oracle
        runs for T_inner many steps. 

        Returns a list of T hypotheses; a uniform distribution over the T hypotheses should be robust.

        :return: list 'hypotheses' which is a list of the T hypotheses.
        """
        constraint_used = 'dp' # dp, eo
        a_indices = self._set_a_indices(sensitive_features)
        w = np.full((X.shape[0],), 1/X.shape[0]) # each weight starts as 1/n
        gamma_1_buckets = self._gamma_1_buckets(X)
        gamma_2_buckets = self._gamma_2_buckets()

        # h_t_pred = self._fair_prediction(X, y, sensitive_features, constraint_used)
        # Start off with oracle prediction, uniform weights
        print("=== Initializing h_0... ===")
        oracle = BayesianOracle(X, y, w, sensitive_features, a_indices,
                                self.card_A, 
                                self.nu, 
                                self.M, 
                                self.B, 
                                self.T_inner,
                                self.gamma_1,
                                gamma_1_buckets, 
                                gamma_2_buckets, 
                                self.epsilon,
                                self.eta,
                                self.num_cores,
                                self.solver)

        h_t = oracle.execute_oracle()
        h_t_pred = h_t.predict(X)

        hypotheses = []
        start_outer = time.time()
        # hypotheses.append(h_t)
        print("=== ALGORITHM 1 EXECUTION ===")
        for t in range(self.T):
            start_inner = time.time()

            w += self.eta * self._zero_one_loss_grad_w(h_t_pred, y)
            w = self._project_W(w)
            oracle = BayesianOracle(X, y, w, sensitive_features, a_indices,
                                self.card_A, 
                                self.nu, 
                                self.M, 
                                self.B, 
                                self.T_inner,
                                self.gamma_1,
                                gamma_1_buckets, 
                                gamma_2_buckets, 
                                self.epsilon,
                                self.eta,
                                self.num_cores,
                                self.solver)
            
            h_t = oracle.execute_oracle()
            h_t_pred = h_t.predict(X)
            hypotheses.append(h_t)

            end_inner = time.time()
            print("ALGORITHM 1 (Meta Algorithm) Loop " + str(t  + 1) + " Completed!")
            print("ALGORITHM 1 (Meta Algorithm) Time/loop: " + str(end_inner - start_inner))
        
        end_outer = time.time()
        print("ALGORITHM 1 (Meta Algorithm) Total Execution Time: " + str(end_outer - start_outer))
        
        self._hypotheses = hypotheses
        return hypotheses, VotingClassifier(hypotheses) # (list of Oracle majority vote classifiers, majority vote classifier out of those)
    
    '''
    def _fair_prediction(self, X, y, sensitive_features, constraint_used):
        unconstrained_predictor = LogisticRegression(class_weight='balanced')
        unconstrained_predictor.fit(X, y)
        unconstrained_predictor_wrapper = LogisticRegressionAsRegression(unconstrained_predictor)

        if(constraint_used =='dp'):
            postprocessed_predictor = ThresholdOptimizer(
                unconstrained_predictor=unconstrained_predictor_wrapper,
                constraints="demographic_parity")
        elif(constraint_used == 'eo'):
            postprocessed_predictor = ThresholdOptimizer(
                unconstrained_predictor=unconstrained_predictor_wrapper,
                constraints="equalized_odds")
        
        postprocessed_predictor.fit(X, y, sensitive_features=sensitive_features)

        return postprocessed_predictor.predict(X, sensitive_features=sensitive_features)
    '''

    '''
    def _gamma_2_buckets(self):
        # Initialize N(gamma_1, W)
        # delta_2 = (2 * self.card_A) / self.gamma_2
        delta_2 = 2/self.gamma_2

        gamma_2_num_buckets = np.ceil(math.log(delta_2, 1 + self.gamma_2)) 
        gamma_2_buckets = []
        gamma_2_buckets.append(1e-6)
        for j in range(int(gamma_2_num_buckets)):
            bucket = (1/delta_2) * (1 + self.gamma_2)**j
            gamma_2_buckets.append(bucket)
        
        pi = list(itertools.product(gamma_2_buckets, gamma_2_buckets)) # kinda expensive

        N_gamma_2_A = []
        for i in range(len(pi)):
            if (((1 - (2 * self.gamma_2)) < pi[i][0] + pi[i][1] < (1 + (2 * self.gamma_2))) and
            pi[i][0] >= 0.1 and
            pi[i][1] >= 0.1):
                N_gamma_2_A.append(pi[i])
        
        print(str(len(N_gamma_2_A)))
        print(str(N_gamma_2_A))
        
        return N_gamma_2_A
    '''
