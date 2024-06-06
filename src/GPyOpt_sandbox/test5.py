import GPyOpt
import matplotlib.pyplot as plt
from numpy.random import seed
import sys

# --- Objective function
objective_true  = GPyOpt.objective_examples.experiments2d.branin()                 # true function
objective_noisy = GPyOpt.objective_examples.experiments2d.branin(sd = 0.1)         # noisy version
bounds = objective_noisy.bounds

domain = [{'name': 'var_1', 'type': 'continuous', 'domain': bounds[0]}, ## use default bounds
          {'name': 'var_2', 'type': 'continuous', 'domain': bounds[1]}]

objective_true.plot()
plt.savefig('out.pdf')
plt.close()

batch_size = 4
num_cores = 4
seed(123)

BO_demo_parallel = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,  
                                            domain = domain,                  
                                            acquisition_type = 'EI',              
                                            normalize_Y = True,
                                            initial_design_numdata = 10,
                                            evaluator_type = 'local_penalization',
                                            batch_size = batch_size,
                                            num_cores = num_cores,
                                            acquisition_jitter = 0)

# BO_demo_parallel.plot_acquisition()
# plt.savefig('prior.pdf')
# plt.close()

# --- Run the optimization for 10 iterations
max_iter = 10                                        
BO_demo_parallel.run_optimization(max_iter)

BO_demo_parallel.plot_acquisition()
plt.savefig('posterior.pdf')
plt.close()
