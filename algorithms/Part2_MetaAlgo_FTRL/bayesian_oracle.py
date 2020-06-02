import cvxpy as cp 
import numpy as np
import math
import time
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from lambda_best_response_param_parallel import LambdaBestResponse
from voting_classifier import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

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
                a_indices, card_A, B, T_inner, gamma_1, gamma_1_buckets, gamma_2_buckets, epsilon, eta, 
                num_cores, solver, fair_constraint, current_t, verbose):
        self.X = X
        self.y = y 
        self.X_test = X_test
        self.y_test = y_test
        self.weights_org = weights_org
        self.sensitive_features = sensitive_features
        self.sensitive_features_test = sensitive_features_test
        self.a_indices = a_indices
        self.card_A = card_A
        self.B = B
        self.T_inner = T_inner
        self.gamma_1 = gamma_1
        self.gamma_1_buckets = gamma_1_buckets
        self.gamma_2_buckets = gamma_2_buckets
        self.epsilon = epsilon
        self.eta = eta
        self.num_cores = num_cores
        self.solver = solver
        self.fair_constraint = fair_constraint
        self.current_t = current_t
        self.verbose = verbose

        # preset for the delta_i computation
        self.delta_i = np.zeros(len(self.weights_org))
        
        # the B vector is the LHS of the Delta_i term (Lambda_w^{a_i, a'} - Lambda_w^{a', a_i})
        # this can be either B or -B, depending on the subgroup of example i
        ### Demographic Parity B vectors ###
        self.B_vec_a0a1 = np.zeros(len(self.weights_org))
        self.B_vec_a0a1[self.a_indices['a0']] += self.B
        self.B_vec_a0a1[self.a_indices['a1']] -= self.B
        self.B_vec_a1a0 = np.zeros(len(self.weights_org))
        self.B_vec_a1a0[self.a_indices['a1']] += self.B
        self.B_vec_a1a0[self.a_indices['a0']] -= self.B

        ### Equalized Odds B vectors ###
        self.B_vec_a0a1y0 = np.zeros(len(self.weights_org))
        self.B_vec_a0a1y0[self.a_indices['a0_y0']] += self.B
        self.B_vec_a0a1y0[self.a_indices['a1_y0']] -= self.B
        self.B_vec_a1a0y0 = np.zeros(len(self.weights_org))
        self.B_vec_a1a0y0[self.a_indices['a1_y0']] += self.B
        self.B_vec_a1a0y0[self.a_indices['a0_y0']] -= self.B
        self.B_vec_a0a1y1 = np.zeros(len(self.weights_org))
        self.B_vec_a0a1y1[self.a_indices['a0_y1']] += self.B
        self.B_vec_a0a1y1[self.a_indices['a1_y1']] -= self.B
        self.B_vec_a1a0y1 = np.zeros(len(self.weights_org))
        self.B_vec_a1a0y1[self.a_indices['a1_y1']] += self.B
        self.B_vec_a1a0y1[self.a_indices['a0_y1']] -= self.B

        # c_1_i (without the Delta_i) and c_0_i can be set only knowing the weights vector and y
        loss_1 = np.ones(len(self.y)) - self.y
        self.c_1_i = np.multiply(loss_1, self.weights_org)
        loss_0 = self.y
        self.c_0_i = np.multiply(loss_0, self.weights_org)

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
        a0_y0_denominator = weights[self.a_indices['a0_y0']].sum()
        a0_y1_denominator = weights[self.a_indices['a0_y1']].sum()
        a1_y0_denominator = weights[self.a_indices['a1_y0']].sum()
        a1_y1_denominator = weights[self.a_indices['a1_y1']].sum()

        # divide element-wise by the denominator (right term of Delta_i)
        if self.fair_constraint == 'dp':
            for i in self.a_indices['a0']:
                weights[i] = weights[i]/a0_denominator
            for i in self.a_indices['a1']:
                weights[i] = weights[i]/a1_denominator
        elif self.fair_constraint == 'eo':
            for i in self.a_indices['a0_y0']:
                weights[i] = weights[i]/a0_y0_denominator
            for i in self.a_indices['a0_y1']:
                weights[i] = weights[i]/a0_y1_denominator
            for i in self.a_indices['a1_y0']:
                weights[i] = weights[i]/a1_y0_denominator
            for i in self.a_indices['a1_y1']:
                weights[i] = weights[i]/a1_y1_denominator
        else:
            raise ValueError("Invalid fairness constraint. Use dp or eo.")

        # then, multiply by the B term (left term of Delta_i)
        if(lambda_tuple[0] == 'a0' and lambda_tuple[1] == 'a1'): # a = a0, a' = a1
            new_delta_i = np.multiply(self.B_vec_a0a1, weights)
        elif(lambda_tuple[0] == 'a1' and lambda_tuple[1] =='a0'): # a = a1, a' = a0
            new_delta_i = np.multiply(self.B_vec_a1a0, weights)
        elif(lambda_tuple[0] == 'a0_y0' and lambda_tuple[1] == 'a1_y0'): # a = a0, a' = a1, y = 0
            new_delta_i = np.multiply(self.B_vec_a0a1y0, weights)
        elif(lambda_tuple[0] == 'a1_y0' and lambda_tuple[1] == 'a0_y0'): # a = a1, a' = a0, y = 0
            new_delta_i = np.multiply(self.B_vec_a1a0y0, weights)
        elif(lambda_tuple[0] == 'a0_y1' and lambda_tuple[1] == 'a1_y1'):
            new_delta_i = np.multiply(self.B_vec_a0a1y1, weights)
        elif(lambda_tuple[0] == 'a1_y1' and lambda_tuple[1] == 'a0_y1'):
            new_delta_i = np.multiply(self.B_vec_a1a0y1, weights)
        else:
            raise ValueError("lambda_tuple in wrong format.")
        return new_delta_i


    def _weighted_classification(self, t):
        """
        Returns LogisticRegression classifier that uses the weights L_i(lambda).

        :return: LogisticRegression. sklearn logistic regression object.
        """
        # Learning becomes a weighted classification problem, dependent on L_i as weights
        # NOTE: c_1_i does NOT have the Delta_i here (just the loss * weight term). just named it that for convenience.
        w = self.eta * t * (self.c_1_i - self.c_0_i) + self.eta * self.delta_i + 0.05
        pos_indices = np.where(w >= 0)
        neg_indices = np.where(w < 0)

        redY = 1 * (w < 0)
        redY = np.asarray(redY)
        redW = w.abs()
        redW = np.asarray(redW)

        if(neg_indices[0].size):
            logreg = LogisticRegression(solver='lbfgs')
            logreg.fit(self.X, redY, sample_weight=redW)
        else:
            logreg = DummyClassifier(strategy="constant", constant=0)
            logreg.fit(self.X, redY, sample_weight=redW)

        return logreg, redW

    def _evaluate_fairness(self, y_true, y_pred, sensitive_features):
        """
        Evaluates fairness of the final majority vote classifier over T_inner hypotheses
        on the test set.
        #NOTE: defined in the meta_algo file, but we chose:
        a0 := African-American (COMPAS), Female (Adult)
        a1 := Caucasian (COMPAS), Male (Adult)

        :return: list. subgroups in sensitive_features.
        :return: dict. recidivism_pct for each group.
        """
        groups = np.unique(sensitive_features.values)
        pos_count = {}
        dp_pct = {}
        eo_y0_pct = {}
        eo_y1_pct = {}
        
        for index, group in enumerate(groups):
            # Demographic Parity
            indices = {}
            indices[group] = sensitive_features.index[sensitive_features == group]
            dp_pct[group] = sum(y_pred[indices[group]])/len(indices[group])

            # Equalized Odds
            y1_indices = {}
            y0_indices = {}
            y1_indices[group] = sensitive_features.index[(sensitive_features == group) & (y_true == 1)]
            y0_indices[group] = sensitive_features.index[(sensitive_features == group) & (y_true == 0)]
            eo_y0_pct[group] = sum(y_pred[y0_indices[group]])/len(y0_indices[group])   
            eo_y1_pct[group] = sum(y_pred[y1_indices[group]])/len(y1_indices[group])
        
        gaps = {}
        group_metrics = {} # a dictionary of dictionaries

        gaps['dp'] = abs(dp_pct[groups[0]] - dp_pct[groups[1]])
        gaps['eo_y0'] = abs(eo_y0_pct[groups[0]] - eo_y0_pct[groups[1]])
        gaps['eo_y1'] = abs(eo_y1_pct[groups[0]] - eo_y1_pct[groups[1]])
        group_metrics['dp'] = dp_pct
        group_metrics['eo_y0'] = eo_y0_pct
        group_metrics['eo_y1'] = eo_y1_pct
        
        return groups, group_metrics, gaps
    
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
        if(self.fair_constraint == 'dp'):
            print("ALGORITHM 2 (Best Response) will solve: " + str(2 * len(self.gamma_2_buckets['dp'])) + " LPs...") # twice because a, a_p
        else:
            print("ALGORITHM 2 (Best Response) will solve: " + str(2 * len(self.gamma_2_buckets['eo_y0']) + 2 * len(self.gamma_2_buckets['eo_y1'])) + " LPs...")

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
                                        self.solver,
                                        self.fair_constraint)

            lambda_t = lambda_best_response.best_response()
            if(lambda_t != (0, 0, 0)):
                self._update_delta_i(lambda_t)

            h_t, final_weights = self._weighted_classification(t)
            h_pred = h_t.predict(self.X)
            hypotheses.append(h_t)

            end_inner = time.time()
            if(self.verbose):
                if((t + 1) % 10 == 0):
                    train_pred = h_t.predict(self.X)
                    test_pred = h_t.predict(self.X_test)
                    train_acc = accuracy_score(train_pred, self.y)
                    test_acc = accuracy_score(test_pred, self.y_test)
                    groups, group_metrics, gaps = self._evaluate_fairness(self.y_test, test_pred, self.sensitive_features_test)
                    print("Train accuracy of classifier {}: {}".format(t + 1, train_acc))
                    print("Test accuracy of classifier {}: {}".format(t + 1, test_acc))
                    if(self.fair_constraint == 'dp'):
                        for group in groups:
                            print("P[h(X) = 1 | {}] = {}".format(group, group_metrics['dp'][group]))
                        print("Delta_dp = {}".format(gaps['dp']))
                    elif(self.fair_constraint == 'eo'):
                        for group in groups:
                            print("P[h(X) = 1 | {}, Y = 0] = {}".format(group, group_metrics['eo_y0'][group]))
                            print("P[h(X) = 1 | {}, Y = 1] = {}".format(group, group_metrics['eo_y1'][group]))
                        print("Delta_eo0 = {}".format(gaps['eo_y0']))
                        print("Delta_eo1 = {}".format(gaps['eo_y1']))
                    else:
                        raise ValueError("Invalid fairness constraint. Choose dp or eo.")
                if(t % 50 == 0):
                    print("ALGORITHM 4 (Learning Algorithm) Loop " + str(t + 1) + " Completed!")
                    print("ALGORITHM 4 (Learning Algorithm) Time/loop: " + str(end_inner - start_inner))

        end_outer = time.time()

        T_inner_ensemble = VotingClassifier(hypotheses)
        print("ALGORITHM 4 (Learning Algorithm) Total Execution Time: " + str(end_outer - start_outer))
        print("=== ALGORITHM 4 (Learning Algorithm) T={} Statistics ===".format(self.current_t))
        y_train_pred = T_inner_ensemble.predict(self.X)
        y_pred = T_inner_ensemble.predict(self.X_test)
        print("Unweighted Train Accuracy = {}".format(accuracy_score(self.y, y_train_pred)))
        print("Weighted Train Accuracy = {}".format(accuracy_score(self.y, y_train_pred, sample_weight=self.weights_org)))
        print("Test Accuracy = {}".format(accuracy_score(self.y_test, y_pred)))
        groups, group_metrics, gaps = self._evaluate_fairness(self.y_test, y_pred, self.sensitive_features_test)
        if(self.fair_constraint == 'dp'):
            for group in groups:
                print("P[h(X) = 1 | {}] = {}".format(group, group_metrics['dp'][group]))
            print("Delta_dp = {}".format(gaps['dp']))
        elif(self.fair_constraint == 'eo'):
            for group in groups:
                print("P[h(X) = 1 | {}, Y = 0] = {}".format(group, group_metrics['eo_y0'][group]))
                print("P[h(X) = 1 | {}, Y = 1] = {}".format(group, group_metrics['eo_y1'][group]))
            print("Delta_eo0 = {}".format(gaps['eo_y0']))
            print("Delta_eo1 = {}".format(gaps['eo_y1']))
        else:
            raise ValueError("Invalid fairness constraint. Choose dp or eo.")

        return T_inner_ensemble, hypotheses
