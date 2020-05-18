import cvxpy as cp 
import numpy as np
import time
from lambda_best_response_param_parallel import EoLinearProgram

n = 100
h_pred = np.random.randint(2, size=n)
a_indices = dict()
a_indices['a0_y0'] = np.arange(25)
a_indices['a1_y0'] = np.arange(25, 50)
a_indices['a0_y1'] = np.arange(50, 75)
a_indices['a1_y1'] = np.arange(75, 100)
a = 'a0'
a_p = 'a1'
y = 'y0'
lp = EoLinearProgram(n, h_pred, a_indices, a, a_p, y, 'ECOS', 0.15)
start = time.time()
value, w, tup = lp.solve((0.3, 0.2))
print(value)
print(h_pred[:25])
w[w < 0] = 0
print(w)
print(tup)
end = time.time()
print('TIME TO SOLVE: ' + str(end - start))