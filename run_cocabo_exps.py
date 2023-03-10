# -*- coding: utf-8 -*-
# ==========================================
# Title:  run_cocabo_exps.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
# ==========================================


import argparse
import os
import time
from experiment.run_experiments.CoCaBO.methods.CoCaBO import CoCaBO
from experiment.test_function import SyntheticFunctions


start = time.perf_counter()
parser = argparse.ArgumentParser(description="Run BayesOpt Experiments")
parser.add_argument('-f', '--func', help='Objective function', default='svm_mse', type=str)
parser.add_argument('-mix', '--kernel_mix', help='Mixture weight, Default = 0.0', default=0.5, type=float)
parser.add_argument('-n', '--max_itr', help='Max Optimisation iterations. Default = 200', default=1, type=int)
parser.add_argument('-tl', '--trials', help='Number of random trials. Default = 20', default=1, type=int)
args = parser.parse_args()
obj_func = args.func
kernel_mix = args.kernel_mix
budget = args.max_itr
n_trials = args.trials
initN = 24
f, C, bounds, af_bounds, lb, ub = SyntheticFunctions.get_obj_func_params(obj_func)
cwd_data = os.path.abspath(os.path.join(os.getcwd(), "experiment")) + "/data/"
saving_path = os.path.join(cwd_data + f'init_data/{obj_func}', '')
mabbo = CoCaBO(objfn=f, initN=initN, bounds=bounds, acq_type='LCB', C=C, kernel_mix=kernel_mix)
mabbo.runTrials(n_trials, budget, saving_path)

end = time.perf_counter()
duration = end - start
print("-----------duration=", round(duration))

