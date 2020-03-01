import cvxpy as cp 
import numpy as np
import math
import time
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression 
from fairlearn.postprocessing import ThresholdOptimizer
import sklearn.metrics
from collections import defaultdict
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression
from lambda_best_response import LambdaBestResponse
import random as ran

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

        self.lambda_dict = defaultdict(int)
        # self.gamma_1_buckets = self._gamma_1_buckets()
        self.discretized_weights = self._discretize_weights()
        self.constraint_used = constraint_used

    def _discretize_weights(self):
        """
        Returns the discretized weights depending on the gamma_1_buckets. Each weight becomes
        the upper endpoint of the bucket/interval it is in.

        :return: list 'discrete_weights' of discretized weights.
        """
        discrete_weights = []
        for i, w_i in enumerate(self.weights):
            for bucket in self.gamma_1_buckets:
                if(bucket[0] <= w_i <= bucket[1]):
                    discrete_weights.append(bucket[1])

        return discrete_weights
    
    def _zero_one_loss(self, y_pred, y_true):
        """
        Returns the 0-1 loss of a single point.

        :return: int. 0 if correct, 1 if not.
        """
        if(y_pred == y_true):
            return 0
        else:
            return 1
    
    def _L_i(self, i, y_i):
        """
        Returns L_i, which is a sample weight for some sample i, dependent on zero-one-loss, the
        weight vector from the Meta-Algo, and delta_i.

        :return: float. the value for L_i (for single sample)
        """
        return self._c_1_i(i, y_i) - self._c_0_i(i, y_i)

    def _c_1_i(self, i, y_i):
        """
        Returns c_1_i, which is the cost of classifying on a 1, dependent on the weights from
        the Meta-Algo and delta_i.

        :return: float. the value for c_1_i (for single sample)
        """
        return self._zero_one_loss(1, y_i)*self.weights[i] + self._delta_i(i)
        
    def _c_0_i(self, i, y_i):
        """
        Returns c_0_i, which is the cost of classifying on a 0, dependent on the weights from
        the Meta-Algo.

        :return: float. the value for c_0_i (for single sample)
        """
        return self._zero_one_loss(0, y_i)*self.weights[i]

    def _delta_i(self, i):
        start = time.time()
        """
        Returns delta_i, the quantity completing c_1_i which incorporates the Lambda best response
        vector.

        :return: float. the value for delta_i (for single sample)
        """
        # get a_i
        if(self.a_indices['all'][i] == 0):
            a_i = 'a0'
            a_p = 'a1'
        else:
            a_i = 'a1'
            a_p = 'a0'
            
        # weights quotient
        quotient = self.weights[i]/self.weights[self.a_indices[a_i]].sum()  
        
        # lambda difference. iterate over all keys of lambda_dict (the rest are 0)
        final_sum = 0
        for tup in self.lambda_dict:
            if(tup[0] == a_i):
                if (tup[1], tup[0], tup[2]) in self.lambda_dict:
                    diff = self.lambda_dict[tup] - self.lambda_dict[(tup[1], tup[0], tup[2])]
                else:
                    diff = self.lambda_dict[tup] # Else here because accessing defaultdict adds a value otherwise
                    
            elif(tup[0] == a_p):
                if (tup[1], tup[0], tup[2]) in self.lambda_dict:
                    diff = self.lambda_dict[(tup[1], tup[0], tup[2])] - self.lambda_dict[tup] 
                else:
                    diff = - self.lambda_dict[tup] # Else here because accessing defaultdict adds a value otherwise

            final_sum += diff*quotient
        
        end = time.time()
        print("delta_i TIME: " + str(end - start))
        return final_sum

    def _weighted_classification(self):
        # Learning becomes a weighted classification problem, dependent on L_i as weights
        ### TODO: OPTIMIZE THIS WITH BROADCASTING ### 
        start = time.time()
        final_weights = []
        for i in range(len(self.X)):
            y_i = self.y[i]
            final_weights.append(self.eta * self._L_i(i, y_i) + 0.5)
        end  = time.time()
        print("final weights TIME: " + str(end - start))

        start = time.time()
        unconstrained_predictor = LogisticRegression()
        unconstrained_predictor.fit(self.X, self.y)
        unconstrained_predictor_wrapper = LogisticRegressionAsRegression(unconstrained_predictor)

        if(self.constraint_used =='dp'):
            postprocessed_predictor = ThresholdOptimizer(
                unconstrained_predictor=unconstrained_predictor_wrapper,
                constraints="equalized_odds")
        elif(self.constraint_used == 'eo'):
            postprocessed_predictor = ThresholdOptimizer(
                unconstrained_predictor=unconstrained_predictor_wrapper,
                constraints="equalized_odds")
        
        postprocessed_predictor.fit(self.X, self.y, sensitive_features=self.sensitive_features, sample_weights=final_weights)
        end = time.time()
        print("fitting TIME:" + str(end - start))

        return postprocessed_predictor
    
    def _uniform_choice(self, hypotheses):
        return hypotheses[ran.randint(0, len(hypotheses) - 1)]

    def execute_oracle(self):
        hypotheses = []
        h = 0
        h_pred = [0 for i in range(len(self.X))]
        hypotheses.append(h_pred)
        for t in range(int(self.T_1)):
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
            end = time.time()
            print("init lambda TIME:" + str(end - start))
            lambda_t = lambda_best_response.best_response()
            if(lambda_t != (0, 0, 0)):
                self.lambda_dict[lambda_t] += self.B
            
            start = time.time()
            h_t = self._weighted_classification()
            end = time.time()
            print("weighted class TIME:" + str(end - start))
            h_pred = h_t.predict(self.X, sensitive_features=self.sensitive_features)
            hypotheses.append(h_pred)
    
        return self._uniform_choice(hypotheses)

    """
    Below is just for testing (generates a random lambda):

    def best_response_lambda(self, h):
        w = [ran.random() for i in range(len(self.X))]
        s = sum(w)
        w = [ i/s for i in w ]

        for i, w_i in enumerate(w):
            for b in self.gamma_1_buckets:
                if(b[0] <= w_i <= b[1]):
                    w[i] = b[1]

        if(ran.randint(0, 1) == 0):
            return ('a0', 'a1', tuple(w))
        else:
            return ('a1', 'a0', tuple(w))
    """