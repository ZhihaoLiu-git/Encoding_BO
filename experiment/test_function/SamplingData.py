import os
import pickle
import numpy as np
import random
import torch
from scipy.stats import qmc

class SamplingData:
    def __init__(self, objfn, num_sampling, bounds, C, saving_path='.//', data_category=''):
        self.f = objfn
        self.bounds = bounds
        self.C = C
        self.num_sampling = num_sampling
        self.saving_path = saving_path
        self.data_category = data_category  # data_category: init_data or design_data
        print("self.saving_path:", self.saving_path)

    def initialise(self, seed):
        # data_category: init_data or design_data
        data = []
        result = []
        # The seeds are fixed to ensure that the same data is used
        np.random.seed(seed)
        random.seed(seed)
        file_path_name = self.saving_path + self.data_category + "_" + str(seed)
        print("file_path_name: ", os.path.exists(file_path_name))
        print("os.path", os.path)
        # Verify that the saving folder exists, if not, then create it
        if not os.path.exists(self.saving_path):
            os.mkdir(self.saving_path)

        # Verify that the data exists
        if os.path.exists(file_path_name):
            print(f"Using existing {self.data_category} data for seed {seed}")
            with open(file_path_name, 'rb') as save_file:
                init_data = pickle.load(save_file)
            Zinit = init_data['Z_init']
            yinit = init_data['y_init']
        else:
            print(f"Creating init data for {self.data_category} seed {seed}")
            Xinit = self.generateInitialPoints(self.num_sampling, self.bounds[len(self.C):])
            hinit = np.hstack(
                [np.random.randint(0, c, self.num_sampling)[:, None] for c in self.C])

            Zinit = np.hstack((hinit, Xinit))
            yinit = np.zeros([Zinit.shape[0], 1])

            for j in range(self.num_sampling):
                ht_list = list(hinit[j])
                if self.f.__name__ == 'pest':
                    yinit[j] = self.f(ht_list)
                else:
                    yinit[j] = self.f(ht_list, Xinit[j])

            init_data = {'Z_init': Zinit, 'y_init': yinit}
            with open(file_path_name, 'wb') as init_data_file:
                pickle.dump(init_data, init_data_file)

        data.append(Zinit)
        result.append(yinit)

        return data[0], torch.from_numpy(result[0])

    def generateInitialPoints(self, num_sampling, bounds):
        nDim = len(bounds)
        Xinit = np.zeros((num_sampling, len(bounds)))
        for i in range(num_sampling):
            Xinit[i, :] = np.array(
                [np.random.uniform(bounds[b]['domain'][0],
                                   bounds[b]['domain'][1], 1)[0]
                 for b in range(nDim)])
        return Xinit

    def sub_at_least_once(self):
        s = np.zeros(shape=[self.C[0], 1])  # ackley's choices are the same
        for C in self.C:
            # fea = np.linspace(1, C, num=C) #[1,17]
            fea = np.linspace(0, C - 1, num=C)  # [0, 16]
            np.random.shuffle(fea)
            s = np.hstack([s, fea[:, None]])
        s = s[:, 1:]
        return s

    def samplingDesignData(self, seed):
        file_path_name = self.saving_path + self.data_category + '_' + str(seed)
        # Verify that the saving folder exists
        if not os.path.exists(self.saving_path):
            # print('the folder of ' + self.data_category + ' already exists')
            print('creating the folder:' + self.data_category)
            os.mkdir(self.saving_path)
        # Verify that the saving folder exists
        if os.path.exists(file_path_name):
            print(f"Using existing {self.data_category} for seed {seed}")
            with open(file_path_name, 'rb') as init_data_file:
                init_data = pickle.load(init_data_file)
            design_data = init_data
        else:
            print(f"Creating {self.data_category} for seed {seed}")
            # sampling from continuous dimensions
            design_x = self.generateInitialPoints(self.num_sampling, self.bounds[len(self.C):])
            np.random.seed(seed)
            design_h = np.hstack(
                [np.random.randint(0, C, self.num_sampling)[:, None] for C in self.C])
            # concatenate h and x
            design_data = np.hstack((design_h, design_x))
            with open(file_path_name, 'wb') as design_data_file:
                pickle.dump(design_data, design_data_file)

        return design_data
