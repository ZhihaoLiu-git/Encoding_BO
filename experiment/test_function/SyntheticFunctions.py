import numpy as np
from numpy import pi, sin
import pandas as pd


def get_obj_func_params(obj_func=''):
    if obj_func == 'Ackley5C':
        f = Ackley5C
        C = [17, 17, 17, 17, 17]
        bounds = [
            {'name': 'h1', 'type': 'categorical', 'domain': tuple(range(0, 17))},
            {'name': 'h2', 'type': 'categorical', 'domain': tuple(range(0, 17))},
            {'name': 'h3', 'type': 'categorical', 'domain': tuple(range(0, 17))},
            {'name': 'h4', 'type': 'categorical', 'domain': tuple(range(0, 17))},
            {'name': 'h5', 'type': 'categorical', 'domain': tuple(range(0, 17))},
            {'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)}]

    else:
        raise NotImplementedError

    bounds_h = list([d['domain'] for d in bounds if d['type'] == 'categorical'])
    bound_x = np.tile(np.array([-1, 1]), (len(bounds) - len(C), 1))
    af_bounds = np.array([[d['domain'][0], d['domain'][-1]] for d in bounds])
    af_bounds[len(C):] = bound_x
    lb = list([d[0] for d in bounds_h])
    ub = list([d[-1] for d in bounds_h])

    return f, C, bounds, af_bounds, lb, ub


def Ackley5C(h_list, d):
    h_arr = np.array(h_list)
    x_cate = -1 + 1 / 8 * h_arr
    x = np.hstack((x_cate, d))
    a = 20
    b = 0.2
    c = 2 * np.pi
    dim = x.shape[0]
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(np.square(x)) / dim))
    cos_term = -1 * np.exp(np.sum(np.cos(c * np.copy(x)) / dim))
    result = sum_sq_term + cos_term + a + np.exp(1)
    return -(result + 1e-6 * np.random.rand())
