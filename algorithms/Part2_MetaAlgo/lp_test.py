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
lp = LinearProgram(n, h_pred, a_indices, a, a_p)
start = time.time()
for i in range(114):
    value, w, tup = lp.solve((0.7, 0.3))
end = time.time()
print('TIME TO SOLVE: ' + str(end - start))