# Bayesian Optimization over Mixed Type Inputs with Encoding Method 
- Target mean encoding BO (TmBO) transfers each value of a categorical input based on the outputs corresponding to this value.  
- Aggregate encoding BO AggBO encodes multiple choices of a categorical input through several distinct ranks.
- Different from the prominent one-hot encoding, both approaches transfer each categorical input into exactly one numerical input and thus avoid severely increasing the dimension of the input space.
- For more details on the method, please read our paper Bayesian Optimization over Mixed Type Inputs with Encoding Method. 
- 
## Target encoding & Ordinal encoding

![](https://github.com/honghaow/Triple-Q/blob/master/env/grid_world.png)

![avatar](https://github.com/WholeG/Bayesian-Optimisation-over-Categorical-Inputs-with-Target-Encoding-Methods/blob/main/pics/distribution1.jpg)
![avatar](https://github.com/ZhihaoLiu-git/Encoding_BO/blob/master/encoding_example.png)


## Dependencies
- category_encoders
- GPy
- GPyOpt 
- hyperopt
- pytorch
- sklearn

## Usage

1. Run Encoding-BO experiments: python run_epxriments.py followed by the following flags: 
- save_result: True/False 
- encoder: In ["Ordinal", "Aggregate", "RandomOrder", "TargetMean", "Onehot"]
- num_sampling: The number of inital data
- budget: Max Optimisation iterations
- max_trial: Max Optimisation trials (different initial data)
- obj_func: Objective function

2. Run CoCaBO/TPE/SMAC experiments in this repository:
- set select_method = 'CoCaBO'/TPE/SMAC 

3. Data storage:
- Encoding_BO\experiment\data\init_data **# initial data**
- Encoding_BO\experiment\data\result_data **# result data**
- Encoding_BO\experiment\data\design_data **# design data, Discretization evaluation for acquisition function**




