import os
import pickle
import numpy as np
import random
import torch

class initBO():
    def __init__(self, objfn, initN, bounds, C, rand_seed=108, saving_path='.//'):
        self.f = objfn
        self.bounds = bounds
        self.C = C
        self.initN = initN
        self.rand_seed = rand_seed
        # self.saving_path = './/'
        self.saving_path = saving_path
        print("self.saving_path:", self.saving_path)
        
    # sampling the input fo design domain
    def sampling_at_least_once_z(self, seed):
        np.random.seed(seed)
        init_fname = self.saving_path + 'init_data_' + str(seed)
        if os.path.exists(init_fname):
            print(f"Using existing init data for seed {seed}")
            with open(init_fname, 'rb') as init_data_filefile2:
                init_data = pickle.load(init_data_filefile2)
            Zinit = init_data['Z_init']
        else:
            print(f"Creating init data for seed {seed}")
            Xinit = self.generateInitialPoints(self.initN, self.bounds[len(self.C):])
            hinit = np.zeros(shape=(self.initN, len(self.C)))
            for c, i in zip(self.C, range(len(self.C))):
                fea_once = np.array(range(0, c))
                remain_fea = np.random.randint(0, c, self.initN - c)
                fea = np.hstack([fea_once, remain_fea])
                np.random.shuffle(fea)
#                 print("fea:", fea)
                hinit[:, i] = fea
            print("hinit", hinit.dtype, hinit)
            Zinit = np.hstack((hinit, Xinit))
            init_data = {}
            init_data['Z_init'] = Zinit
            with open(init_fname, 'wb') as init_data_file:
                pickle.dump(init_data, init_data_file)

        return Zinit
    
    def initialise(self, seed):
        """Get NxN intial points"""
        data = []
        result = []

        np.random.seed(seed)
        random.seed(seed)
        init_fname = self.saving_path + 'init_data_' + str(seed)
        # init_fname = self.saving_path + 'experiment_data/'+'init_data_' + str(seed)
        print("init_fname: ", os.path.exists(init_fname))
        print("os.path", os.path)
        if os.path.exists(init_fname):
            print(f"Using existing init data for seed {seed}")
            with open(init_fname, 'rb') as init_data_filefile2:
                init_data = pickle.load(init_data_filefile2)
            Zinit = init_data['Z_init']
            yinit = init_data['y_init']
        else:
            print(f"Creating init data for seed {seed}")
            Xinit = self.generateInitialPoints(self.initN,
                                               self.bounds[len(self.C):])
            # start: add for ackley  as h start from 1
            if 'Ackley' in self.f.__name__ :
                print("sampling at least once")
                h_at_leat_once = self.sub_at_least_once()
                h_remain = np.hstack(
                    # all the elements are the same
                    [np.random.randint(0, C, self.initN-self.C[0])[:, None] for C in self.C])
                hinit = np.vstack([h_at_leat_once, h_remain])
            # end
            else:
                hinit = np.hstack(
                    [np.random.randint(0, C, self.initN)[:, None] for C in self.C])
            Zinit = np.hstack((hinit, Xinit))
            yinit = np.zeros([Zinit.shape[0], 1])

            for j in range(self.initN):
                ht_list = list(hinit[j])
#               # print(ht_list, Xinit[j], yinit[j])
                yinit[j] = self.f(ht_list, Xinit[j])
                # print(ht_list, Xinit[j], yinit[j])

            init_data = {}
            init_data['Z_init'] = Zinit
            init_data['y_init'] = yinit
            print("./")
            with open(init_fname, 'wb') as init_data_file:
                pickle.dump(init_data, init_data_file)

        data.append(Zinit)
        result.append(yinit)
        # write to csv
        # dataframe = pd.DataFrame({'data': data, 'result': result})
        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        # dataframe.to_csv("data_result.csv", index=False, sep=',', mode='a', header=True)

        return data[0], torch.from_numpy(result[0])

    def generateInitialPoints(self, initN, bounds):
        nDim = len(bounds)
        Xinit = np.zeros((initN, len(bounds)))
        for i in range(initN):
            Xinit[i, :] = np.array(
                [np.random.uniform(bounds[b]['domain'][0],
                                   bounds[b]['domain'][1], 1)[0]
                 for b in range(nDim)])
        return Xinit

      
    def sub_at_least_once(self):
        s = np.zeros(shape=[self.C[0], 1])
        for C in self.C:
            fea = np.linspace(1, C, num=C)
            np.random.shuffle(fea)
            s = np.hstack([s, fea[:, None]])
        s = s[:, 1:]
        return s