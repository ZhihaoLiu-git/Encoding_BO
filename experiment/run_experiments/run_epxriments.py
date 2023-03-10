import argparse
import os
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import GPy
import GPyOpt
import torch


from experiment.run_experiments.CoCaBO.methods.CoCaBO import CoCaBO
from experiment.test_function import SyntheticFunctions
from experiment.test_function.SamplingData import SamplingData
from method.encoders import PermutationEncoder
from method.gp import GP
from method.acquisition import UCB

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from category_encoders import TargetEncoder, OneHotEncoder
from hyperopt import fmin, tpe, hp
from hyperopt.fmin import generate_trials_to_calculate


def get_acq_values(acq_object, design_array):
    num_chunks = 10
    x_chunks = np.split(design_array, num_chunks)
    f_samples_list = []
    for x_chunk in x_chunks:
        f_samples_list.append(acq_object.evaluate(x_chunk))

    return np.array(f_samples_list).ravel()


def update_weight(mse_arr, gp_weights):
    max_mse = max(mse_arr)
    min_mse = min(mse_arr)
    # mse is scaled to [0,1]
    mse_arr = (mse_arr - min_mse) / (max_mse - min_mse)
    neg_weights_exp = np.exp(-1 * mse_arr)
    gp_weights = neg_weights_exp / np.sum(neg_weights_exp)

    print("mse_list:", mse_arr, "gp_weights = ", gp_weights)
    return gp_weights

def get_beta(iteration, dim):
    delt = 0.1
    a = 0.5
    b = 0.5
    r = 1
    beta = 2.0 * np.log((iteration + 1) * (iteration + 1) * math.pi ** 2 / (3 * delt)) + 2 * dim * np.log(
        (iteration + 1) * (iteration + 1) * dim * b * r * np.sqrt(np.log(4 * dim * a / delt)))

    return beta

# methods = ["Ordinal", "Aggregate", "RandomOrder", "TargetMean", "Onehot", "CoCaBO", "SMAC", "TPE"]
select_method = 'CoCaBO'
obj_func = 'Ackley5C'
save_result = False
parser = argparse.ArgumentParser(description="Run Experiments")
parser.add_argument('--encoder', help='the type of encoder', default=f'{select_method}', type=str)
parser.add_argument('--num_sampling', help='number of sampling ', default='24', type=int)
parser.add_argument('--budget', help='the number of iterations in each experiment', default=200, type=int)
parser.add_argument('--max_trial', help='Number of random experiments', default=8, type=int)
parser.add_argument('--obj_func', help='Objective function', default=obj_func, type=str)
# add for CoCaBO
parser.add_argument('-mix', '--kernel_mix', help='Mixture weight, Default = 0.0', default=0.5, type=float)
args = parser.parse_args()
print(f"Got arguments: \n{args}")
cwd_data = os.path.abspath(os.getcwd()) + "/experiment/data/"
init_data_path = os.path.join(cwd_data + f'init_data/{obj_func}', '')
design_data_path = os.path.join(cwd_data + f'design_data/{obj_func}', '')
result_path = os.path.join(cwd_data + f'result_data/{obj_func}', '')
encoder = args.encoder
budget = args.budget
max_trials = args.max_trial
num_sampling = args.num_sampling
f, C, bounds, af_bounds, lb, ub = SyntheticFunctions.get_obj_func_params(obj_func)
sca_z = preprocessing.MinMaxScaler(feature_range=(-1, 1))
sca_y = preprocessing.MinMaxScaler(feature_range=(-1, 1))
df_list = []
num_perms = 6
update_w_iter = 50  # start to update weights in AggrateBO
num_design = 10000
default_lengthscale = 0.2 * np.ones(shape=len(bounds))
hp_lengthscale_bounds = np.repeat([[1.e-04, 3.e+00]], repeats=len(bounds), axis=0)
lik_var_bound = np.array([1.e-06, 1.e+00])
hp_bounds = np.vstack((hp_lengthscale_bounds, lik_var_bound))
gp_opt_params = {'method': 'multigrad', 'num_restarts': 5,
                 'restart_bounds': hp_bounds, 'hp_bounds': hp_bounds, 'verbose': False}

start = time.perf_counter()
if select_method == 'Aggregate':
    for trial in range(max_trials):
        print("----------num_trial: ", trial)
        data, y = SamplingData(f, num_sampling, bounds, C, saving_path=init_data_path, data_category='init_data') \
            .initialise(seed=trial)
        ys = sca_y.fit_transform(y)
        gp_w = np.ones(shape=[num_perms, 1]) * 1 / num_perms
        # gp_w = np.array([[1], [0], [0], [0], [0], [0]])
        enc = PermutationEncoder(obj_func, C=C).fit(data, num_perms)
        z_list = enc.transform(data)
        data_list = [data for n in range(num_perms)]
        y_list = [y for n in range(num_perms)]
        sub_y_max_list = [[y.max()] for n in range(num_perms)]
        max_index_list = []
        cand_y_list = []
        max_y_list = [y.max()]
        cand_sub_model_list = [[] for i in range(num_perms + 1)]  # +1 is for agg
        for iteration in range(budget):
            print("iteration: ", iteration)
            beta = get_beta(iteration, len(bounds))

            design_data = SamplingData(f, num_design, bounds, C, saving_path=design_data_path,
                                       data_category='design_data').samplingDesignData(seed=num_design + iteration)
            design_z_list = enc.transform(design_data)
            gp_list = []
            gp_acq_list = []
            # multi-gp encoding
            gps_acq = np.ones(shape=[num_design, num_perms])
            mse_list = []
            for i, z, design_z in zip(range(num_perms), z_list, design_z_list):
                z.values[:, len(C):] = sca_z.fit_transform(z.values[:, len(C):])
                design_z.values[:, len(C):] = sca_z.transform(design_z.values[:, len(C):])
                # k_m52 = GPy.kern.Matern52(input_dim=len(bounds), lengthscale=default_lengthscale,
                #                           active_dims=list(range(len(bounds))), ARD=False)
                k_m52 = GPy.kern.Matern52(input_dim=z.shape[1], lengthscale=default_lengthscale, ARD=True)
                k_m52.unlink_parameter(k_m52.variance)  # optimize 2 parameters: lengthscale, lik_variance
                gp = GP(z.values, ys, k_m52, y_norm='meanstd', opt_params=gp_opt_params)
                if iteration % 10 == 0:
                    gp.optimize()
                acq = UCB(gp, beta)
                sub_acq_value = get_acq_values(acq_object=acq, design_array=design_z.values)
                gps_acq[:, i] = sub_acq_value
                # set weights
                if iteration >= update_w_iter:
                    z_train, z_test, y_train, y_test = train_test_split(z.values, ys,
                            test_size=0.3, random_state=iteration)
                    from sklearn.svm import SVR
                    regr = SVR().fit(z_train, y_train.ravel())
                    y_pred = regr.predict(z_test)
                    mse_list.append(mean_squared_error(y_pred, y_test))

            if iteration >= update_w_iter:
                gp_w = update_weight(np.array(mse_list), gp_w)
                print("mse_list=", mse_list, "gp_w=", gp_w)
            next_index = np.argmax(np.matmul(gps_acq, gp_w))
            max_index_list.append(next_index)
            cand_data = design_data[next_index]
            # cand_sub_model_list[-1].append(cand_data)
            cand_h_list = design_data[next_index, :len(C)]
            cand_x = design_data[next_index, len(C):]
            candidate_y = f(cand_h_list, cand_x)
            data = np.vstack((data, cand_data))
            z_list = enc.transform(data)
            y = np.vstack((y, candidate_y))
            ys = sca_y.transform(y)
            cand_y_list.append(candidate_y)
            max_y_list.append(max(y)[0])

        df_list.append(max_y_list)
elif select_method == 'RandomOrder':
    for trial in range(max_trials):
        print("----------num_trial: ", trial)
        data, y = SamplingData(f, num_sampling, bounds, C, saving_path=init_data_path, data_category='init_data') \
            .initialise(seed=trial)
        ys = sca_y.fit_transform(y)
        enc = PermutationEncoder(obj_func, C=C).fit(data, num_perms)
        z = enc.transform(data)[0]
        cand_y_list = []
        max_y_list = [y.max()]

        for iteration in range(budget):
            print("iteration: ", iteration)
            z.values[:, len(C):] = sca_z.fit_transform(z.values[:, len(C):])
            design_data = SamplingData(f, num_design, bounds, C, saving_path=design_data_path,
                                       data_category='design_data').samplingDesignData(seed=num_design + iteration)
            design_z = enc.transform(design_data)[0]
            design_z.values[:, len(C):] = sca_z.transform(design_z.values[:, len(C):])
            k_m52 = GPy.kern.Matern52(input_dim=z.shape[1], lengthscale=default_lengthscale, ARD=True)
            k_m52.unlink_parameter(k_m52.variance)
            gp = GP(z.values, ys, k_m52, y_norm='meanstd', opt_params=gp_opt_params)
            if iteration % 10 == 0:
                gp.optimize()
            acq = UCB(gp, 2)
            acq_value = get_acq_values(acq_object=acq, design_array=design_z.values)
            max_index = np.argmax(acq_value)
            cand_data = design_data[max_index]
            cand_h_list = design_data[max_index, :len(C)]
            cand_x = design_data[max_index, len(C):]
            cand_y = f(cand_h_list, cand_x)
            data = np.vstack((data, cand_data))
            y = np.vstack((y, cand_y))
            ys = sca_y.transform(y)
            cand_y_list.append(cand_y)
            max_y_list.append(max(y)[0])

            enc = PermutationEncoder(obj_func, C=C).fit(data, num_perms)
            z = enc.transform(data)[0]

        df_list.append(max_y_list)
elif select_method == 'Onehot':
    Encoder = OneHotEncoder
    dim_continuous = len(bounds) - len(C)
    for trial in range(max_trials):
        print("----------num_trial: ", trial)
        data, y = SamplingData(f, num_sampling, bounds, C, saving_path=init_data_path, data_category='init_data') \
            .initialise(seed=trial)
        ys = sca_y.fit_transform(y)
        enc = Encoder(cols=list(range(len(C)))).fit(X=data, y=ys)
        z = enc.transform(data)
        cand_y_list = []
        max_y_list = [y.numpy().max()]

        for iteration in range(budget):
            print("iteration: ", iteration)
            # the number of dimension increased after onehot, the shape of default_lengthscale should be adjusted
            ard_onehot_default_ls = 0.2 * np.ones(shape=z.shape[1])
            ard_onehot_lengthscale_bounds = np.vstack(
                (np.repeat([[1.e-04, 3.e+00]], repeats=z.shape[1], axis=0), lik_var_bound))
            one_hot_opt_params = {'method': 'multigrad', 'num_restarts': 5,
                                  'restart_bounds': ard_onehot_lengthscale_bounds,
                                  'hp_bounds': ard_onehot_lengthscale_bounds, 'verbose': False}

            beta = get_beta(iteration, len(bounds))
            design_data = SamplingData(f, num_design, bounds, C, saving_path=design_data_path,
                                       data_category='design_data').samplingDesignData(seed=num_design + iteration)
            design_z = enc.transform(design_data)
            km52 = GPy.kern.Matern52(input_dim=z.shape[1], lengthscale=ard_onehot_default_ls,
                                     active_dims=list(range(z.shape[1])), ARD=True)
            km52.unlink_parameter(km52.variance)  # optimize 2 parameters: lengthscale, lik_variance
            gp = GP(z.values, ys, km52, y_norm='meanstd', opt_params=one_hot_opt_params)
            if iteration % 10 == 0:
                gp.optimize()
            acq = UCB(gp, beta)
            acq_value = get_acq_values(acq_object=acq, design_array=design_z.values)
            max_index = np.argmax(acq_value)
            # max_index = np.argmax(acq.evaluate(design_z.values))
            cand_data = design_data[max_index]
            cand_h_list = design_data[max_index, :len(C)]
            cand_x = design_data[max_index, len(C):]
            cand_y = f(cand_h_list, cand_x)
            data = np.vstack((data, cand_data))
            y = np.vstack((y, cand_y))
            max_y_list.append(y.max())
            ys = sca_y.transform(y)

            enc = Encoder(cols=list(range(len(C)))).fit(X=data, y=ys)
            z = enc.transform(data)

        df_list.append(max_y_list)
elif select_method == 'TargetMean':
    Encoder = TargetEncoder
    for trial in range(max_trials):
        print("----------num_trial: ", trial)
        data, y = SamplingData(f, num_sampling, bounds, C, saving_path=init_data_path, data_category='init_data') \
            .initialise(seed=trial)
        ys = sca_y.fit_transform(y)
        enc = Encoder(cols=list(range(len(C)))).fit(X=data, y=ys)
        z = enc.transform(data)
        # only scale the continuous variables
        z.values[:, len(C):] = sca_z.fit_transform(z.values[:, len(C):])
        cand_y_list = []
        max_y_list = [y.numpy().max()]
        default_lengthscale = 0.2 * np.ones(shape=z.shape[1])
        for iteration in range(budget):
            print("iteration: ", iteration)
            beta = get_beta(iteration, len(bounds))
            design_data = SamplingData(f, num_design, bounds, C, saving_path=design_data_path,
                                       data_category='design_data').samplingDesignData(seed=num_design + iteration)
            design_z = enc.transform(design_data)
            design_z.values[:, len(C):] = sca_z.transform(design_z.values[:, len(C):])
            km52 = GPy.kern.Matern52(input_dim=z.shape[1], lengthscale=default_lengthscale, ARD=True)
            km52.unlink_parameter(km52.variance)
            gp = GP(z.values, ys, km52, y_norm='meanstd', opt_params=gp_opt_params)
            if iteration % 10 == 0:
                gp.optimize()
            acq = UCB(gp, beta)
            acq_value = get_acq_values(acq_object=acq, design_array=design_z.values)
            max_index = np.argmax(acq_value)
            cand_data = design_data[max_index]
            cand_h_list = design_data[max_index, :len(C)]
            cand_x = design_data[max_index, len(C):]
            cand_y = f(cand_h_list, cand_x)
            data = np.vstack((data, cand_data))
            y = np.vstack((y, cand_y))
            max_y_list.append(y.max())
            ys = sca_y.transform(y)

            enc = Encoder(cols=list(range(len(C)))).fit(X=data, y=ys)
            z = enc.transform(data)
            z.values[:, len(C):] = sca_z.fit_transform(z.values[:, len(C):])

        df_list.append(max_y_list)

elif select_method == 'CoCaBO':
    obj_func = args.obj_func
    kernel_mix = args.kernel_mix
    n_trials = args.max_trial
    budget = args.budget
    initN = args.num_sampling
    f, C, bounds, af_bounds, lb, ub = SyntheticFunctions.get_obj_func_params(obj_func)

    mabbo = CoCaBO(objfn=f, initN=initN, bounds=bounds, acq_type='LCB', C=C, kernel_mix=kernel_mix)
    mabbo.runTrials(n_trials, budget, cwd_data)

elif select_method == 'SMAC':
    for trial in range(max_trials):
        print("----------num_trial: ", trial)
        data, y = SamplingData(f, num_sampling, bounds, C, saving_path=init_data_path, data_category='init_data') \
            .initialise(seed=trial)
        y = -1 * y
        y_s = sca_y.fit_transform(y)
        cand_y_list = []
        min_y_list = [y.min()]

        for i in range(budget):
            beta = get_beta(iteration, len(bounds))
            smac = GPyOpt.methods.BayesianOptimization(f=None, domain=bounds, X=data, Y=y_s, normalize_Y=False,
                                                       maximize=True, acquisition_type='LCB', acquisition_weight=beta,
                                                       model_type="RF")
            candidate = smac.suggest_next_locations()
            candidate_y = -1 * torch.tensor(f(candidate[0, :len(C)], candidate[0, len(C):])).unsqueeze(0)
            min_y_list.append(y.min().item())
            cand_y_list.append(candidate_y)
            data = np.vstack((data, candidate))
            y = torch.vstack((y, candidate_y))
            y_s = sca_y.fit_transform(y)

        max_y_list = list(-1 * np.array(min_y_list))
        df_list.append(max_y_list)

elif select_method == 'TPE':
    names = []
    name_cat = [d['name'] for d in bounds if d['type'] == 'categorical']
    name_con = [d['name'] for d in bounds if d['type'] == 'continuous']
    bound_cat = [range(d['domain'][0], d['domain'][-1] + 1) for d in bounds if d['type'] == 'categorical']
    bound_con = [(d['domain'][0], d['domain'][-1]) for d in bounds if d['type'] == 'continuous']
    search_space = {name: hp.choice(name, range(_b[0], _b[-1]+1)) for name, _b in zip(name_cat, bound_cat)}
    dict_con = {name: hp.uniform(name, _b[0], _b[-1]) for name, _b in zip(name_con, bound_con)}
    search_space.update(dict_con)
    names.extend(name_cat)
    names.extend(name_con)


    def objective_function(params):
        h_arr = np.array([params[cat_] for cat_ in name_cat])
        x_con_arr = np.array([params[con_] for con_ in name_con])
        return f(h_arr, x_con_arr)

    def get_initial_points(input_name):
        init_data = []
        for j in range(num_sampling):
            init_data.append(dict(zip(input_name, data_init[j])))
        return init_data

    for trial in range(max_trials):
        print("----------num_trial: ", trial)
        data_init, y = SamplingData(f, num_sampling, bounds, C, saving_path=init_data_path, data_category='init_data') \
            .initialise(seed=trial)
        print("min_y = ", y.min())
        max_y_list = [y.numpy().max()]
        cand_y_list = []
        points_init = get_initial_points(names)
        # get the existing initial data
        trials = generate_trials_to_calculate(points_init)
        best = fmin(
            fn=objective_function,  # Objective Function to optimize
            space=search_space,  # Hyperparameter's Search Space
            algo=tpe.suggest,  # Optimization algorithm
            max_evals=budget,  # Number of optimization attempts
            trials=trials)

        cand_y_list = trials.losses()[num_sampling:]
        curr_max = max(max_y_list)
        for cand in cand_y_list:
            if cand > curr_max:
                curr_max = cand
            max_y_list.append(curr_max)

        df_list.append(max_y_list)

end = time.perf_counter()
duration = round(end - start)
print("-----------duration=", round(duration))

if save_result and encoder != 'CoCaBO':
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    file_name = f"{result_path}{encoder}_trials_{max_trials}_budget_{budget}_duration{duration}"
    output = open(file_name, 'wb')
    pickle.dump(df_list, output)
    output.close()
    plt.plot(max_y_list)
    plt.show()