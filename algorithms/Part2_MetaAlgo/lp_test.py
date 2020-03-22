import cvxpy as cp 
import numpy as np

class LinearProgram():
    def __init__(self, n, h_pred, a_indices, a, a_p):
        self._w = cp.Variable(n)
        self._a = a
        self._a_p = a_p

        # Problem constants
        self._h_xi_a = h_pred.copy()
        self._h_xi_a[a_indices[a_p]] = 0
        self._h_xi_ap = h_pred.copy()
        self._h_xi_ap[a_indices[a]] = 0
        self.pi_0 = cp.Parameter(nonneg=True)
        self.pi_1 = cp.Parameter(nonneg=True)

        # Constraints
        self._constraints = [
            cp.sum(self._w[a_indices[a]]) == self.pi_0,
            cp.sum(self._w[a_indices[a_p]]) == self.pi_1,
            cp.sum(self._w) == self.pi_0 + self.pi_1,
            0 <= self._w
        ]

        # Objective Function
        self._objective = cp.Maximize((1/self.pi_0 * (self._w @ self._h_xi_a)) - (1/self.pi_1 * (self._w @ self._h_xi_ap)))
        self._prob = cp.Problem(self._objective, self._constraints) 

    def solve(self, pi):
        if(self._a == 'a0'):
            self.pi_0.value = pi[0]
            self.pi_1.value = pi[1]
        else:
            self.pi_0.value = pi[1]
            self.pi_1.value = pi[0]
        self._prob.solve(solver='GUROBI', verbose=False, warm_start = True)
        return self._prob.value, self._w.value, (self._a, self._a_p, (pi[0], pi[1]))

n = 5
h_pred = np.array([1, 1, 0, 1, 0])
a_indices = dict()
a_indices['a0'] = [0, 1, 2]
a_indices['a1'] = [3, 4]
a = 'a0'
a_p = 'a1'
lp = LinearProgram(n, h_pred, a_indices, a, a_p)
print(lp.solve((0.7, 0.3)))