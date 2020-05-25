import cvxpy as cp 
import numpy as np
import time
from lambda_best_response_param_parallel import DpLinearProgram

n = 10
h_pred = np.asarray([0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
a_indices = dict()
a_indices['a0'] = np.arange(6)
a_indices['a1'] = np.arange(6, 10)
a = 'a0'
a_p = 'a1'
y = 'y0'
lp = DpLinearProgram(n, h_pred, a_indices, a, a_p, 'ECOS')
value, w, tup = lp.solve((0.65, 0.35))
print(value)
print(h_pred)
w[w < 0] = 0
print(w)
print(tup)
