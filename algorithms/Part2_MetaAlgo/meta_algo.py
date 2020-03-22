import cvxpy as cp 
import numpy as np
import math
import time 
import itertools
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from bayesian_oracle_broadcasting import BayesianOracle, LogisticRegressionAsRegression
from tqdm import tqdm

class MetaAlgorithm:
    """ The Meta-Algorithm (Algorithm 1) that performs gradient descent on the weights and  
    calls the Bayesian oracle at each time step t.
    
    :param T: the number of steps to run the Meta-Algo
    :type int

    :param T_1: the number of steps to run the Bayesian Oracle
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

    def __init__(self, T, T_1, card_A = 2, nu = 0.01, M = 1, epsilon = 0.1, 
                B = None, eta = None, gamma_1 = None, gamma_2 = None):
        self.T = T
        self.T_1 = T_1
        self.card_A = card_A
        self.nu = nu
        self.B = B
        self.eta = eta
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.M = M
        self.epsilon = epsilon
        
        if B is None:
            self.B = M
        if eta is None:
            self.eta = 1/np.sqrt(2*T)
        if gamma_1 is None:
            self.gamma_1 = nu/self.B
        if gamma_2 is None:
            self.gamma_2 = nu/self.B

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
        # Initialize N(gamma_1, W)
        # delta_2 = (2 * self.card_A) / self.gamma_2
        delta_2 = 2/self.gamma_2
        print(delta_2)
        c = .1 # the constant in front of the log (EXPERIMENT WITH THIS)

        gamma_2_num_buckets = np.ceil(c * math.log(delta_2, 1 + self.gamma_2)) 
        gamma_2_buckets = []
        gamma_2_buckets.append(1e-6)
        for j in range(int(gamma_2_num_buckets)):
            bucket = ((1/delta_2) ** c) * (1 + self.gamma_2)**j
            gamma_2_buckets.append(bucket)
            
        pi = list(itertools.product(gamma_2_buckets, gamma_2_buckets))

        N_gamma_2_A = []
        for i in range(len(pi)):
            if ((1 - (2 * self.gamma_2)) < pi[i][0] + pi[i][1] < (1 + (2 * self.gamma_2))):
                N_gamma_2_A.append(pi[i])
        
        print(N_gamma_2_A)
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
        objective = cp.Minimize(0.5 * cp.sum_squares(w - x))
        constraints = [0 <= x, x <= 1, cp.sum(x) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver='GUROBI', verbose=False)
        
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
        runs for T_1 many steps. 

        Returns a list of T hypotheses; a uniform distribution over the T hypotheses should be robust.

        :return: list 'hypotheses' which is a list of the T hypotheses.
        """
        constraint_used = 'dp' # dp, eo
        a_indices = self._set_a_indices(sensitive_features)
        w = np.full((X.shape[0],), 1/X.shape[0]) # each weight starts as 1/n

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
        
        postprocessed_predictor.fit(X, y, 
            sensitive_features=sensitive_features, sample_weights=w)

        h_t_pred = postprocessed_predictor.predict(X, sensitive_features=sensitive_features)

        gamma_1_buckets = self._gamma_1_buckets(X)
        gamma_2_buckets = self._gamma_2_buckets()

        hypotheses = []
        for t in tqdm(range(self.T)):
            w += self.eta * self._zero_one_loss_grad_w(h_t_pred, y)
            w = self._project_W(w)
            oracle = BayesianOracle(X, y, w, sensitive_features, a_indices,
                                self.card_A, 
                                self.nu, 
                                self.M, 
                                self.B, 
                                self.T_1,
                                self.gamma_1,
                                gamma_1_buckets, 
                                gamma_2_buckets, 
                                self.epsilon,
                                self.eta)
            
            h_t = oracle.execute_oracle()
            h_t_pred = h_t.predict(X)
            hypotheses.append(h_t)
        
        self._hypotheses = hypotheses
        return hypotheses
    