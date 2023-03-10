# -*- coding: utf-8 -*-
# ==========================================
# Title:  BaseBO.py
# Author: Binxin Ru and Ahsan Alvi
# Date:   20 August 2019
# Link:   https://arxiv.org/abs/1906.08878
# ==========================================

import os
import pickle
import random

import numpy as np
import pandas as pd
from openpyxl import load_workbook


class BaseBO():
    """
    Base class with common operations for BO with continuous and categorical
    inputs
    """

    def __init__(self, objfn, initN, bounds, C, rand_seed=108, debug=False,
                 batch_size=1, **kwargs):
        self.f = objfn  # function to optimise
        self.bounds = bounds  # function bounds
        self.batch_size = batch_size
        self.C = C  # no of categories = [3, 5]
        self.initN = initN  # no: of initial points
        self.nDim = len(self.bounds)  # dimension
        self.rand_seed = rand_seed
        self.debug = debug
        self.saving_path = None
        self.kwargs = kwargs
        self.x_bounds = np.vstack([d['domain'] for d in self.bounds
                                   if d['type'] == 'continuous'])

    def initialise(self, seed):
        data = []
        result = []
        np.random.seed(seed)
        init_fname = self.saving_path + 'init_data_' + str(seed)
        if os.path.exists(init_fname):
            print(f"Using existing init data for seed {seed}")
            with open(init_fname, 'rb') as init_data_filefile2:
                init_data = pickle.load(init_data_filefile2)

            Zinit = init_data['Z_init']
            yinit = init_data['y_init']
        # else:
            # print(f"Creating init data for seed {seed}")
            # Xinit = self.generateInitialPoints(self.initN, self.bounds[len(self.C):])
            # hinit = np.zeros(shape=(self.initN, len(self.C)))
            #Make sure each sub categorical variable is drawn at least once
            # for c, i in zip(self.C, range(len(self.C))):
            #     fea_once = np.array(range(0, c))
            #     remain_fea = np.random.randint(1, c+1, self.initN - c)
            #     fea = np.hstack([fea_once, remain_fea])
            #     np.random.shuffle(fea)
            #     print("fea column:", fea)
            #     hinit[:, i] = fea

        else:
            print(f"Creating init data for seed {seed}")
            Xinit = self.generateInitialPoints(self.initN, self.bounds[len(self.C):])
            # all the elements are the same
            hinit = np.hstack(
                [np.random.randint(0, c, self.initN)[:, None] for c in self.C])
            # print("sampling at least once")
            # h_at_leat_once = self.sub_at_least_once()
            # h_remain = np.hstack(
            #     # all the elements are the same
            #     [np.random.randint(0, C, self.initN-self.C[0])[:, None] for C in self.C])
            # hinit = np.vstack([h_at_leat_once, h_remain])


            Zinit = np.hstack((hinit, Xinit))
            yinit = np.zeros([Zinit.shape[0], 1])

            for j in range(self.initN):
                ht_list = list(hinit[j])
                yinit[j] = self.f(ht_list, Xinit[j])
                # print(ht_list, Xinit[j], yinit[j])

            init_data = {}
            init_data['Z_init'] = Zinit
            init_data['y_init'] = yinit
            with open(init_fname, 'wb') as init_data_file:
                pickle.dump(init_data, init_data_file)

        data.append(Zinit)
        result.append(yinit)
        return data, result

    
#     def initialise(self, seed):
#         """Get NxN intial points"""
#         data = []
#         result = []

#         np.random.seed(seed)
#         random.seed(seed)

#         init_fname = self.saving_path + 'init_data_' + str(seed)

#         if os.path.exists(init_fname):
#             print(f"Using existing init data for seed {seed}")
#             with open(init_fname, 'rb') as init_data_filefile2:
#                 init_data = pickle.load(init_data_filefile2)
#             Zinit = init_data['Z_init']
#             yinit = init_data['y_init']
#         else:
#             print(f"Creating init data for seed {seed}")
#             Xinit = self.generateInitialPoints(self.initN,
#                                                self.bounds[len(self.C):])
#             num_once = max(self.C)
#             if self.f.__name__.startswith('Ackley'):
#                 #                 h_at_least_once = np.hstack([np.random.randint(1, C+1, num_once)[:,None] for C in self.C])
#                 h_at_least_once = np.zeros(shape=(num_once, len(self.C)))
#                 for c, i in zip(self.C, range(len(self.C))):
#                     fea = np.linspace(1, c, num_once)
#                     np.random.shuffle(fea)
#                     h_at_least_once[:, i] = fea
#                 h_remain = np.hstack([np.random.randint(1, C + 1, self.initN - num_once)[:, None] for C in self.C])
#             else:
#                 h_at_least_once = np.hstack([np.random.randint(0, C, num_once)[:, None] for C in self.C])
#                 h_remain = np.hstack([np.random.randint(0, C, self.initN - num_once)[:, None] for C in self.C])


#             hinit = np.vstack([h_at_least_once, h_remain])
#             # hinit = np.hstack(
#             #     [np.random.randint(0, C, self.initN)[:, None] for C in self.C])
#             Zinit = np.hstack((hinit, Xinit))
#             yinit = np.zeros([Zinit.shape[0], 1])

#             for j in range(self.initN):
#                 ht_list = list(hinit[j])
#                 yinit[j] = self.f(ht_list, Xinit[j])
#                 # print(ht_list, Xinit[j], yinit[j])

#             init_data = {}
#             init_data['Z_init'] = Zinit
#             init_data['y_init'] = yinit

#             with open(init_fname, 'wb') as init_data_file:
#                 pickle.dump(init_data, init_data_file)

#         data.append(Zinit)
#         result.append(yinit)
#         return data, result

    def generateInitialPoints(self, initN, bounds):
        nDim = len(bounds)
        Xinit = np.zeros((initN, len(bounds)))
        for i in range(initN):
            Xinit[i, :] = np.array(
                [np.random.uniform(bounds[b]['domain'][0],
                                   bounds[b]['domain'][1], 1)[0]
                 for b in range(nDim)])
        return Xinit

    def my_func(self, Z):
        Z = np.atleast_2d(Z)
        if len(Z) == 1:
            X = Z[0, len(self.C):]
            ht_list = list(Z[0, :len(self.C)])
            return self.f(ht_list, X)
        else:
            f_vals = np.zeros(len(Z))
            for ii in range(len(Z)):
                X = Z[ii, len(self.C):]
                ht_list = list(Z[ii, :len(self.C)].astype(int))
                f_vals[ii] = self.f(ht_list, X)
            return f_vals

    def save_progress_to_disk(self, *args):
        raise NotImplementedError

    def runTrials(self, trials, budget, saving_path):
        raise NotImplementedError

