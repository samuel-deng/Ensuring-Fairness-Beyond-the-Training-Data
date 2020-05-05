import cvxpy as cp 
import numpy as np
import math
import time
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from lambda_best_response_param_parallel import LambdaBestResponse
from voting_classifier import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
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
    def __init__(self, X, y, X_test, y_test, weights_org, sensitive_features, sensitive_features_test, 
                a_indices, card_A, M, B, T_inner, gamma_1, gamma_1_buckets, gamma_2_buckets, 
                epsilon, eta, num_cores, solver, constraint_used, current_t):
        self.X = X
        self.y = y 
        self.X_test = X_test
        self.y_test = y_test
        self.weights_org = weights_org
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

        # preset for the delta_i computation
        self.delta_i = np.zeros(len(self.weights_org))

        # weights that don't change over time
        self.const_i = np.zeros(len(self.weights_org))
        #self.const_i = np.multiply(self.weights_org, 1 - 2*self.y)
        # the B vector is the LHS of the Delta_i term (Lambda_w^{a_i, a'} - Lambda_w^{a', a_i})
        # this can be either B or -B, depending on the subgroup of example i
        self.B_vec_a0a1 = np.zeros(len(self.weights_org))
        self.B_vec_a0a1[self.a_indices['a0']] += self.B
        self.B_vec_a0a1[self.a_indices['a1']] -= self.B
        self.B_vec_a1a0 = np.zeros(len(self.weights_org))
        self.B_vec_a1a0[self.a_indices['a1']] += self.B
        self.B_vec_a1a0[self.a_indices['a0']] -= self.B

    def _update_delta_i(self, lambda_tuple):
        """
        Updates our current Delta_i vector based on Lambda Best Response

        :return: none.
        """
        new_delta_i = self._get_new_delta_i(lambda_tuple)

        self.delta_i = self.delta_i + new_delta_i

    def _get_new_delta_i(self, lambda_tuple):

        weights = lambda_tuple[2]

        # compute the denominator (sum_{j:a_j = a_i} w_j)
        a0_denominator = weights[self.a_indices['a0']].sum()
        a1_denominator = weights[self.a_indices['a1']].sum()

        # divide element-wise by the denominator (right term of Delta_i)
        for i in self.a_indices['a0']:
            weights[i] = weights[i]/a0_denominator
        for i in self.a_indices['a1']:
            weights[i] = weights[i]/a1_denominator

        # then, multiply by the B term (left term of Delta_i)
        if(lambda_tuple[0] == 'a0'): # a = a0, a' = a1
            new_delta_i = np.multiply(self.B_vec_a0a1, weights)
        else:                        # a = a1, a' = a0
            new_delta_i = np.multiply(self.B_vec_a1a0, weights)
        return new_delta_i


    def _randomized_classification(self, lambda_tuple):
        
        w = self.eta * (self.const_i + self.delta_i) 
        if lambda_tuple != (0,0,0):
            w = w + self.eta * ( np.multiply(self.weights_org, 1 - 2*self.y) +  self._get_new_delta_i(lambda_tuple) )
        else:
            w = w + self.eta * ( np.multiply(self.weights_org, 1 - 2*self.y) )
        #print(np.max(self.const_i))
        #print(np.max(self.delta_i))
        #print('------')
        nvar = len(w)
        #print('inside randomized classification')
        #print(nvar)
        #solve a convex optimization problem
        Q = cp.Variable(nvar)
        objective = cp.Minimize( Q.T @ w + (0.05) * cp.sum_squares(Q))
        constraints = [0.001 <= Q, Q <= 0.999]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        #print(Q.value)
        #print(np.max(Q.value))
        return Q.value


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
        #h_pred = np.random.binomial(n=1,p=0.5,size=len(self.X))
        #print('printing h_pred')
        #print(h_pred)
        start_outer = time.time()

        print("Executing ALGORITHM 4 (Learning Algorithm)...")
        print("ALGORITHM 2 (Best Response) will solve: " + str(2 * len(self.gamma_2_buckets)) + " LPs...") # twice because a, a_p
        for t in range(int(self.T_inner)):
            # if self.eta*0.99 < 0.01:
            #     self.eta = 0.01
            # else:
            #     self.eta = self.eta * 0.99

            start_inner = time.time()

            lambda_best_response = LambdaBestResponse(h_pred, 
                                        self.a_indices, 
                                        self.gamma_1,
                                        self.gamma_1_buckets,
                                        self.gamma_2_buckets, 
                                        self.epsilon, 
                                        self.num_cores,
                                        self.solver)

            lambda_t = lambda_best_response.best_response()
            #print('printing lambda t')
            #print(lambda_t[2].size)
            #print(lambda_t)
            if(lambda_t != (0, 0, 0)):
                #print('non-zero case')
                self._update_delta_i(lambda_t)

            self.const_i += np.multiply(self.weights_org, 1 - 2*self.y)
            #print(self.const_i)

            Q_t = self._randomized_classification(lambda_t)
            h_pred = [np.random.binomial(n=1,p=Q_t[i]) for i in range(len(Q_t))]
            h_pred = [int(round(h_pred[v])) for v in range(len(Q_t))]
            #get classifier that matches the prediction h_pred
            #print(np.sum(h_pred))
            #logreg = LogisticRegression(penalty='none', solver='lbfgs')
            #logreg.fit(self.X, np.asarray(h_pred) )
            forest = RandomForestClassifier(n_estimators=100,max_depth=50)
            forest.fit(self.X, h_pred)
            #print(h_pred)
            hypotheses.append(forest)
            new_h_pred = forest.predict(self.X)
            #print(accuracy_score(np.asarray(self.y), np.asarray(new_h_pred) ) )
            #print(self.const_i)
            end_inner = time.time()
            if(t % 50 == 0):
                print("ALGORITHM 4 (Learning Algorithm) Loop " + str(t + 1) + " Completed!")
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

        #performance on training dataset
        # y_pred = T_inner_ensemble.predict(self.X)
        # groups, recidivism_pct, gap = self._evaluate_fairness(y_pred, self.sensitive_features)
        # print("Accuracy (train) = {}".format(accuracy_score(np.asarray(self.y), np.asarray(y_pred) )))
        # for group in groups:
        #     print("P[h(X) = 1 | {}] = {}".format(group, recidivism_pct[group]))
        # print("Delta_DP = {}".format(gap))

        return T_inner_ensemble, hypotheses
