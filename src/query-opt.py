import GPy, sys, random,os
from RFM_RBF import RFM_RBF
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

def f(X):
    return -(6*X-2)**2 * np.sin(12*X-4)

def g(X):
    dim = X.shape[1]
    a = 20
    b = 0.2
    c = 2 * np.pi
    return -(-a * np.exp(-b * np.sqrt(np.sum(X**2,axis = 1)/dim))-np.exp(np.sum(np.cos(c*X),axis =1)/dim) + a+ np.exp(1))



def expected_improvement(X, X_train, y_train, model, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        y_train: Evaluated function f's value (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, var = model.predict_noiseless(X)
    sigma = np.sqrt(var)
    #sigma = sigma.reshape(-1, 1)
    
    y_max = np.max(y_train)

    with np.errstate(divide='warn'):
        imp = mu - y_max - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

# def mes(X,):

def propose_location(acquisition, X_sample, Y_sample, model, bounds, n_restarts=25):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, model)
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')       
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x           
            
    return np.atleast_2d(min_x)

def experiment(seed, train_num, n_iter,fig_ok):
    input_dim = 1
    random.seed(seed)
    np.random.seed(seed)
    noise_var = 1.0e-4
    
    init_num = 3
    grid_num = init_num ** input_dim
    if input_dim == 1:
        bounds = np.array([[-0.0, 1.0]])
        X = np.c_[np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)]
        X_train = np.c_[np.random.choice(X.ravel(), train_num,replace=False)]
        y_true_max = f(0.75724876)

    if input_dim == 2:
        bounds = np.array([[-32.768, 32.768]])
        X = np.c_[np.linspace(-32.768, 32.768, init_num)]
        xx , yy = np.meshgrid(X,X)
        aa = xx.reshape(grid_num,1)
        bb = yy.reshape(grid_num,1)
        X = np.hstack((bb, aa))

    # X = np.c_[np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)]
    train_index = np.random.randint(grid_num,size=train_num)
    X_train = X[train_index]
    y = f(X)
    y_train = f(X_train)

    y_max = y_train.max()
    simple_regret = y_true_max - y_max
    print('iter:0 ',simple_regret)
    kernel = GPy.kern.RBF(input_dim = X.shape[1],lengthscale = 1)
    model = GPy.models.GPRegression(X_train, y_train, kernel=kernel,normalizer = True)
    model['.*Gaussian_noise.variance'].constrain_fixed(noise_var)
    model.optimize_restarts(num_restarts=10, verbose=0)
    pred_mean, pred_var = model.predict_noiseless(X)

    # PLOT
    if fig_ok:
        plt.plot(X,y,'r',label = 'true')
        plt.plot(X,pred_mean,'b',label = 'pred_mean')
        plt.plot(X_train,y_train,'ro',label = 'observed')
        plt.fill_between(X.ravel(), (pred_mean + 1.96 * np.sqrt(pred_var)).ravel(), (pred_mean - 1.96 * np.sqrt(pred_var)).ravel(), alpha=0.3, color="blue", label="credible interval")
        plt.legend(loc="lower left")
        plt.savefig('../out/fig.pdf')
        plt.close()
    
    #BO-LOOP
    for i in range(n_iter):
        # Obtain next sampling point from the acquisition function (expected_improvement)
        X_next = propose_location(expected_improvement, X_train, y_train, model, bounds)
        # Obtain next noisy sample from the objective function
        Y_next = f(X_next)
        if Y_next > y_max:
            y_max = Y_next
        
        pred_mean, pred_var = model.predict_noiseless(X)
        # Plot samples, surrogate function and next sampling location
        if fig_ok:
            plt.plot(X,y,'r',label = 'true')
            plt.plot(X,pred_mean,'b',label = 'pred_mean')
            plt.plot(X_train,y_train,'ro',label = 'observed')
            plt.fill_between(X.ravel(), (pred_mean + 1.96 * np.sqrt(pred_var)).ravel(), (pred_mean - 1.96 * np.sqrt(pred_var)).ravel(), alpha=0.3, color="blue", label="credible interval")
            plt.legend(loc="lower left")
            plt.savefig('../out/fig_'+str(i)+'.pdf')
            plt.close()

        # Caluculate Regret
        simple_regret = y_true_max - y_max
        print('iter:'+str(i+1)+' ',simple_regret)
        # print('point: ',X_next)

        # Add sample to previous samples
        X_train = np.vstack((X_train, X_next))
        y_train = np.vstack((y_train, Y_next))

        # Update Gaussian process with existing samples
        model.set_XY(X_train, y_train)
        model.optimize_restarts(num_restarts=10, verbose=0)

def main():
    seed = 0
    train_num = 2
    n_iter = 10
    fig_ok = True

    experiment(seed,train_num,n_iter,fig_ok)

if __name__ == "__main__":
    main()