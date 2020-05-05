import cvxpy as cp 
import numpy as np
import time
from lambda_best_response_param_parallel import LinearProgram

n = 4222
h_pred = np.random.randint(2, size=n)
a_indices = dict()
a_indices['a0'] = np.arange(2111)
a_indices['a1'] = np.arange(2111, 4222)
a = 'a0'
a_p = 'a1'
lp = LinearProgram(n, h_pred, a_indices, a, a_p, solver = 'ECOS')
start = time.time()
value, w, tup = lp.solve((0.6, 0.4))
print(value)
print(h_pred[:25])
print(w[:25])
w[w < 0] = 0
print(w[:25])
end = time.time()
print('TIME TO SOLVE: ' + str(end - start))