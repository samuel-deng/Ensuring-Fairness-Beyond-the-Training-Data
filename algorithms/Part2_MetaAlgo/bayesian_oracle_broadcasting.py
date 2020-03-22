import cvxpy as cp 
import numpy as np
import math
import time
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression 
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score
from collections import defaultdict
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression
from lambda_best_response_param_parallel import LambdaBestResponse
import random as ran
from tqdm import tqdm

class LogisticRegressionAsRegression:
    def __init__(self, logistic_regression_estimator):
        self.logistic_regression_estimator = logistic_regression_estimator
    
    def fit(self, X, y):
        self.logistic_regression_estimator.fit(X, y)
    
    def predict(self, X):
        # use predict_proba to get real values instead of 0/1, select only prob for 1
        scores = self.logistic_regression_estimator.predict_proba(X)[:,1]
        return scores

class BayesianOracle:
    """ The Bayesian Oracle step (Algorithm 4) that learns the actual classifier
    via cost-sensitive classification, responding to the best Lambda response. 
    Returns a uniform distribution over T_1 classifiers. 
    
    :param X: the training set.
    :type nparray:

    :param y: the trainig labels.
    :type nparray:

    :param weights: the weights given by the Meta-Algorithm to construct the weighted
    classification problem.
    :type nparray:

    :param sensitive_features: a Series given to the ExpGrad algorithm to determine which
    instances are protected/non-protected
    :type Series:

    :param a_indices: a dictionary with the information on which indices have the sensitive features.
    :type dict:

    :param card_A: the cardinality of A, the set of protected attributes
    :type int:

    :param nu: the desired accuracy to the Bayesian oracle guarantee (Theorem 4)
    :type float:

    :param M: the bound on the loss
    :type float:

    :param B: the bound on each Lambda 
    :type float:

    :param T_1: the number of steps to run the Bayesian Oracle
    :type int:

    :param gamma_1: a parameter that sets how many "buckets" to discretize the weight 
    vectors in the Bayesian Oracle and Lambda Best Response step. Depends on nu.
    :type float:

    :param gamma_2: a parameter that sets how many "buckets" to discretize the weight 
    vectors pi in the Lambda Best Response step. Depends on nu.
    :type float:

    :param epsilon: the desired "fairness gap" between the protected attributes
    :type float:

    :param eta: the learning/gradient descent rate for the weights in Meta-Algo and the coefficient
    for the weighted classification problem.
    :type float:

    :param constraint_used: the constraint (DP or EO) used for fairness
    :type str:
    """

    def __init__(self, X, y, weights, sensitive_features, a_indices, card_A, nu, M, B, T_1, gamma_1, gamma_1_buckets, gamma_2_buckets, epsilon, eta, constraint_used = 'dp'):
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

        # speedup for delta_i function
        self._weights_a0_sum = self.weights[self.a_indices['a0']].sum()
        self._weights_a1_sum = self.weights[self.a_indices['a1']].sum()  

        # element-wise divide was acting funny, so did it this way
        delta_i_weights = self.weights.copy()
        for i in self.a_indices['a0']:
            delta_i_weights[i] = delta_i_weights[i]/self._weights_a0_sum
        for i in self.a_indices['a1']:
            delta_i_weights[i] = delta_i_weights[i]/self._weights_a1_sum
        self._delta_i_weights = delta_i_weights

        self.lambda_sum = np.zeros(len(self.weights)) # chosen to be lambda_0_1 - lambda_1_0 
        self.constraint_used = constraint_used
    
    def _L_i(self):
        """
        Returns L_i, which is a sample weight for some sample i, dependent on zero-one-loss, the
        weight vector from the Meta-Algo, and delta_i.

        :return: float. the value for L_i (for single sample)
        """
        return self._c_1_i() - self._c_0_i()

    def _c_1_i(self):
        """
        Returns c_1_i, which is the cost of classifying on a 1, dependent on the weights from
        the Meta-Algo and delta_i.

        :return: float. the value for c_1_i (for single sample)
        """
        # NOTE: the zero-one loss for the 1-vector and y is just 1-vector minus y
        loss = np.ones(len(self.y)) - self.y
        return np.multiply(loss, self.weights) + self._delta_i_fast()
        
    def _c_0_i(self):
        """
        Returns c_0_i, which is the cost of classifying on a 0, dependent on the weights from
        the Meta-Algo.

        :return: float. the value for c_0_i (for single sample)
        """
        # NOTE: the zero-one loss for the 0-vector and y is just y itself
        loss = self.y
        return np.multiply(loss, self.weights)

    def _delta_i_fast(self):
        delta_i_vec = np.multiply(self._delta_i_weights, self.lambda_sum) # elementwise multiply
        return delta_i_vec

    def _weighted_classification(self):
        # Learning becomes a weighted classification problem, dependent on L_i as weights
        final_weights = self.eta * self._L_i() + 0.5

        logreg = LogisticRegression(class_weight='balanced', solver='lbfgs')
        logreg.fit(self.X, self.y, sample_weight=final_weights)
        return logreg
    
    def _uniform_choice(self, hypotheses):
        return hypotheses[ran.randint(0, len(hypotheses) - 1)]

    def execute_oracle(self):
        hypotheses = []
        h = 0
        h_pred = [0 for i in range(len(self.X))]
        for t in tqdm(range(int(self.T_1))):
            start = time.time()
            lambda_best_response = LambdaBestResponse(h_pred, 
                                        self.X, 
                                        self.y, 
                                        self.weights, 
                                        self.sensitive_features, 
                                        self.a_indices, 
                                        self.card_A, 
                                        self.nu, 
                                        self.M, 
                                        self.B, 
                                        self.T_1,
                                        self.gamma_1,
                                        self.gamma_1_buckets,
                                        self.gamma_2_buckets, 
                                        self.epsilon, 
                                        self.eta)

            lambda_t = lambda_best_response.best_response()
            if(lambda_t != (0, 0, 0)):
                if(lambda_t[0] == 'a0'):
                    self.lambda_sum[self.a_indices['a0']] += self.B
                    self.lambda_sum[self.a_indices['a1']] -= self.B
                else:
                    self.lambda_sum[self.a_indices['a0']] -= self.B
                    self.lambda_sum[self.a_indices['a1']] += self.B
                    
            h_t = self._weighted_classification()
            h_pred = h_t.predict(self.X)
            hypotheses.append(h_t)

            end = time.time()
            print("oracle step TIME: " + str(end - start))
    
        return self._uniform_choice(hypotheses)