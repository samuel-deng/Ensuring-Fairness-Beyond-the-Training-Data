import cvxpy as cp 
import numpy as np
import time
from lambda_best_response_param_parallel import EoLinearProgram

n = 10
h_pred = np.asarray([0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
a_indices = dict()
a_indices['a0_y0'] = np.arange(3)
a_indices['a1_y0'] = np.arange(3, 5)
a_indices['a0_y1'] = np.arange(5, 8)
a_indices['a1_y1'] = np.arange(8, 10)
a = 'a0'
a_p = 'a1'
y = 'y0'
lp = EoLinearProgram(n, h_pred, a_indices, a, a_p, y, 'ECOS')
value, w, tup = lp.solve((0.35, 0.15))
print(value)
print(h_pred)
w[w < 0] = 0
print(w)
print(tup)
