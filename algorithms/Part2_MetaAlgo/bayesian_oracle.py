import cvxpy as cp 
import numpy as np
import math
import time
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from lambda_best_response_param_parallel import LambdaBestResponse
from voting_classifier import VotingClassifier

""" 
The Bayesian Oracle step (Algorithm 4) that learns the actual classifier
via cost-sensitive classification, calling lambda_best_response_param_parallel
to solve LPs. The main function here is execute_oracle().

Returns:
(1) VotingClassifier: A majority-vote classifier to get the first step of the outer loop (initializing
new weights) off the ground
(2) list: List of T_inner many hypotheses to append to the big list in the outer loop

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
"""

class BayesianOracle:
    def __init__(self, X, y, X_test, y_test, weights, sensitive_features, sensitive_features_test, 
                a_indices, card_A, M, B, T_inner, gamma_1, gamma_1_buckets, gamma_2_buckets, 
                epsilon, eta, num_cores, solver, constraint_used, current_t):
        self.X = X
        self.y = y 
        self.X_test = X_test
        self.y_test = y_test
        self.weights = weights
        self.sensitive_features = sensitive_features
        self.sensitive_features_test = sensitive_features_test
        self.a_indices = a_indices
        self.card_A = card_A
        self.M = M
        self.B = B
        self.T_inner = T_inner
        self.gamma_1 = gamma_1
        self.gamma_1_buckets = gamma_1_buckets
        self.gamma_2_buckets = gamma_2_buckets
        self.epsilon = epsilon
        self.eta = eta
        self.num_cores = num_cores
        self.solver = solver
        self.constraint_used = constraint_used
        self.current_t = current_t

        # the denominator in the Delta_i term (pg. 8)
        self._weights_a0_sum = self.weights[self.a_indices['a0']].sum()
        self._weights_a1_sum = self.weights[self.a_indices['a1']].sum()

        # element-wise divide was acting funny, so did it this way
        # entire rational term (w_i/sum w_j) in Delta_i term (pg. 8), as a vector for each example
        delta_i_weights = self.weights.copy()
        for i in self.a_indices['a0']:
            delta_i_weights[i] = delta_i_weights[i]/self._weights_a0_sum
        for i in self.a_indices['a1']:
            delta_i_weights[i] = delta_i_weights[i]/self._weights_a1_sum
        self._delta_i_weights = delta_i_weights

        # left term in the Delta_i term, the double-sum
        # we choose to use the ordering: lambda_0_1 - lambda_1_0 
        self.lambda_sum = np.zeros(len(self.weights)) 
    
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
        """
        Returns delta_i_vec, a vector that includes delta_i for each sample i in [n]

        :return: nparray. delta_i_vec vector.
        """
        delta_i_vec = np.multiply(self._delta_i_weights, self.lambda_sum) # elementwise multiply
        return delta_i_vec

    def _weighted_classification(self):
        """
        Returns LogisticRegression classifier that uses the weights L_i(lambda).

        :return: LogisticRegression. sklearn logistic regression object.
        """
        # Learning becomes a weighted classification problem, dependent on L_i as weights
        final_weights = self.eta * self._L_i() + 0.5

        logreg = LogisticRegression(class_weight='balanced', solver='lbfgs')
        logreg.fit(self.X, self.y, sample_weight=final_weights)
        return logreg

    def _evaluate_fairness(self, y_pred, sensitive_features):
        """
        Evaluates fairness of the final majority vote classifier over T_inner hypotheses
        on the test set.
        #TODO: add equalized odds option
        #NOTE: defined in the meta_algo file, but we chose:
        a0 := African-American (COMPAS), Female (Adult)
        a1 := Caucasian (COMPAS), Male (Adult)

        :return: list. subgroups in sensitive_features.
        :return: dict. recidivism_pct for each group.
        """
        groups = np.unique(sensitive_features.values)
        indices = {}
        recidivism_count = {}
        recidivism_pct = {}
        for index, group in enumerate(groups):
            indices[group] = sensitive_features.index[sensitive_features == group]
            recidivism_count[group] = sum(y_pred[indices[group]])
            recidivism_pct[group] = recidivism_count[group]/len(indices[group])
        
        gap = abs(recidivism_pct[groups[0]] - recidivism_pct[groups[1]])
        return groups, recidivism_pct, gap
    
    def execute_oracle(self):
        """
        Runs the Algorithm 4. Most of the time here is used in solving the LPs in the 
        LambdaBestResponse object, which is called via lambda_best_response.best_response().
        This entire function returns:
        
        (1) a VotingClassifier object for majority vote over T_inner-many hypotheses, 
        which is just used to check for fairness violation and get the loss vector for the outer loop.
        (2) a list of T_inner-many hypotheses that is appended to the outer loop hypotheses.

        :return: VotingClassifier. majority vote classifier object (details in voting_classifier.py)
        :return: list. a list of hypotheses to append to our big list in the outer loop.
        """
        hypotheses = []
        h = 0
        h_pred = [0 for i in range(len(self.X))]
        start_outer = time.time()

        print("Executing ALGORITHM 4 (Learning Algorithm)...")
        print("ALGORITHM 2 (Best Response) will solve: " + str(2 * len(self.gamma_2_buckets)) + " LPs...") # twice because a, a_p
        for t in range(int(self.T_inner)):
            start_inner = time.time()

            lambda_best_response = LambdaBestResponse(h_pred, 
                                        self.X, 
                                        self.y, 
                                        self.weights, 
                                        self.sensitive_features, 
                                        self.a_indices, 
                                        self.card_A, 
                                        self.M, 
                                        self.B, 
                                        self.T_inner,
                                        self.gamma_1,
                                        self.gamma_1_buckets,
                                        self.gamma_2_buckets, 
                                        self.epsilon, 
                                        self.eta,
                                        self.num_cores,
                                        self.solver)

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

            end_inner = time.time()
            if(t % 50 == 0):
                print("ALGORITHM 4 (Meta Algorithm) Loop " + str(t + 1) + " Completed!")
                print("ALGORITHM 4 (Learning Algorithm) Time/loop: " + str(end_inner - start_inner))

        end_outer = time.time()

        T_inner_ensemble = VotingClassifier(hypotheses)
        print("ALGORITHM 4 (Learning Algorithm) Total Execution Time: " + str(end_outer - start_outer))
        print("=== ALGORITHM 4 (Learning Algorithm) T={} Statistics ===".format(self.current_t))
        y_pred = T_inner_ensemble.predict(self.X_test)
        groups, recidivism_pct, gap = self._evaluate_fairness(y_pred, self.sensitive_features_test)
        print("Accuracy = {}".format(accuracy_score(self.y_test, y_pred)))
        for group in groups:
            print("P[h(X) = 1 | {}] = {}".format(group, recidivism_pct[group]))
        print("Delta_DP = {}".format(gap))

        return T_inner_ensemble, hypotheses