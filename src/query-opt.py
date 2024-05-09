import GPy, sys, random, os
from RFM_RBF import RFM_RBF
from sklearn.kernel_approximation import RBFSampler

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from ThetaGenerator import ThetaGenerator
from sklearn.utils import check_array, check_random_state, as_float_array


def f(X):
    """FORRESTER FUNCTION

    Args:
        X : input

    Returns:
        function's value of input X
    """  #
    return -((6 * X - 2) ** 2) * np.sin(12 * X - 4)


def g(X):
    """ACKLEY FUNCTION

    Args:
        X : input

    Returns:
        function's value of input X
    """
    dim = X.shape[1]
    a = 20
    b = 0.2
    c = 2 * np.pi
    return -(
        -a * np.exp(-b * np.sqrt(np.sum(X**2, axis=1) / dim))
        - np.exp(np.sum(np.cos(c * X), axis=1) / dim)
        + a
        + np.exp(1)
    )


def expected_improvement(X, X_train, y_train, model, xi=0.01):
    """
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.

    Args:
        X: Points at which EI shall be computed (m x d).
        y_train: Evaluated function f's value (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.

    Returns:
        Expected improvements at points X.
    """
    mu, var = model.predict_noiseless(X)
    sigma = np.sqrt(var)
    # sigma = sigma.reshape(-1, 1)
    y_max = np.max(y_train)

    with np.errstate(divide="warn"):
        imp = mu - y_max - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei


def upper_confidence_bound(X, X_train, y_train, model):
    mu, var = model.predict_noiseless(X)
    t = X_train.shape[0]
    ucb = mu + np.sqrt(2 * np.log(t**2 + 1)) * np.sqrt(var)
    return ucb


# def max_value_entropy_search(X,X_train,y_train,model):
#     mu, var = model.predict_noiseless(X)


def propose_location(acquisition, X_sample, Y_sample, model, bounds, n_restarts=25):
    """
    Proposes the next sampling point by optimizing the acquisition function.

    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    """
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, model)

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method="L-BFGS-B")
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x

    return np.atleast_2d(min_x)


def experiment(seed, train_num, n_iter, fig_ok):
    random.seed(seed)
    np.random.seed(seed)
    input_dim = 1
    noise_var = 1.0e-4
    if input_dim == 1:
        init_num = 100
        grid_num = init_num**input_dim
        bounds = np.array([[-0.0, 1.0]])
        # train_index = np.random.randint(np.random.randint(0, init_num, train_num))
        index_list = range(grid_num)
        train_index = random.sample(index_list, train_num)
        X = np.c_[np.linspace(0, 1, init_num)]
        X_train = X[train_index]
        y_train = f(X_train)
        y = f(X)
        y_true_max = f(0.75724876)
        acquisition = f

    if input_dim == 2:
        init_num = 20
        grid_num = init_num**input_dim
        bounds = np.array([[-32.768, 32.768]])
        X = np.c_[np.linspace(-32.768, 32.768, init_num)]
        xx, yy = np.meshgrid(X, X)
        aa = xx.reshape(grid_num, 1)
        bb = yy.reshape(grid_num, 1)
        X = np.hstack((bb, aa))
        index_list = range(grid_num)
        train_index = random.sample(index_list, train_num)
        X_train = X[train_index]
        y_true_max = g(np.atleast_2d([0, 0]))
        y = np.c_[g(X)]
        y_train = np.c_[g(X_train)]
        func = g

    y_max = y_train.max()
    simple_regret = y_true_max - y_max
    kernel = GPy.kern.RBF(input_dim=X_train.shape[1], ARD=True)
    model = GPy.models.GPRegression(
        X_train, y_train, kernel=kernel
    )  # , normalizer=True)
    model[".*Gaussian_noise.variance"].constrain_fixed(noise_var)
    # model["rbf.variance"].constrain_fixed(1.0)
    # variance = 1.0
    model.optimize_restarts(num_restarts=10, verbose=0)
    pred_mean, pred_var = model.predict_noiseless(X)

    # PLOT
    if fig_ok:
        plt.plot(X, y, "r", label="true")
        plt.plot(X, pred_mean, "b", label="pred_mean")
        plt.plot(X_train, y_train, "ro", label="observed")
        plt.fill_between(
            X.ravel(),
            (pred_mean + 1.96 * np.sqrt(pred_var)).ravel(),
            (pred_mean - 1.96 * np.sqrt(pred_var)).ravel(),
            alpha=0.3,
            color="blue",
            label="credible interval",
        )
        plt.legend(loc="lower left")
        plt.savefig("../out/fig.pdf")
        plt.close()

    # BO-LOOP
    for i in range(n_iter):
        # Obtain next sampling point from the acquisition function (expected_improvement)
        X_next = propose_location(expected_improvement, X_train, y_train, model, bounds)
        print("X_next: ", X_next)
        # Obtain next noisy sample from the objective function
        if input_dim == 1:
            Y_next = f(X_next)
        if input_dim == 2:
            Y_next = g(X_next)

        if Y_next > y_max:
            y_max = Y_next

        pred_mean, pred_var = model.predict_noiseless(X)

        print(model.rbf.lengthscale)

        # basis_dim = 1000
        # sample_size = 100
        # # RFM
        # rfm_feature = RFM_RBF(
        #     model.rbf.lengthscale[0],
        #     X.shape[1],
        #     variance=model.rbf.variance[0],
        #     basis_dim=basis_dim,
        # )
        # features = rfm_feature.transform(X_train)
        # Theta = ThetaGenerator(basis_dim, noise_var)
        # Theta.calc(features, y_train)
        # theta = Theta.getTheta(sample_size)
        # feature = rfm_feature.transform(X)
        # f_hat = np.dot(feature, theta.T)
        # f_hat = f_hat.T

        # RFM ライブラリー使用版 (未検証)
        # rbf_feature = RBFSampler(
        #     gamma=1 / (2 * model.rbf.lengthscale[0] ** 2),
        #     n_components=basis_dim,
        #     random_state=1,
        # )
        # features = rbf_feature.fit_transform(X)
        # Theta = ThetaGenerator(basis_dim, noise_var)
        # Theta.calc_init(features)
        # theta = Theta.getTheta(sample_size)
        # f_hat = np.dot(theta, features.T)

        # 離散型(正解確認用)
        # me, gpy_pred_cov = model.predict(X, full_cov=True)
        # z = np.random.randn(len(X), sample_size)
        # A = np.linalg.cholesky(gpy_pred_cov)
        # f_hat = me + np.dot(A, z)
        # f_hat = f_hat.T

        # Plot samples, surrogate function and next sampling location
        if fig_ok:
            fig, axes = plt.subplots(
                2,
                1,
                gridspec_kw=dict(height_ratios=[2, 1]),
                tight_layout=True,
            )
            axes[0].plot(X, y, "r", label="true")
            axes[0].plot(X, pred_mean, "b", label="pred_mean")
            axes[0].plot(X_train, y_train, "kx", label="observed")
            axes[0].axvline(X_next, color="red", linestyle="dashed", linewidth=1)
            # RFMの表示
            # for j in range(f_hat.shape[0]):
            #     axes[0].plot(X, f_hat[j], "g")
            axes[0].fill_between(
                X.ravel(),
                (pred_mean + 1.96 * np.sqrt(pred_var)).ravel(),
                (pred_mean - 1.96 * np.sqrt(pred_var)).ravel(),
                alpha=0.3,
                color="blue",
                label="credible interval",
            )
            axes[0].legend(loc="lower left")
            axes[0].set_ylim(-25, 18)
            # 獲得関数値の計算
            ei = expected_improvement(X.reshape(-1, 1), X_train, y_train, model)
            axes[1].plot(X, ei, "g", label="ei")

            # ucb = upper_confidence_bound(X.reshape(-1, 1), X_train, y_train, model)
            # axes[1].plot(X, ucb, "b", label="ucb")
            axes[1].axvline(X_next, color="red", linestyle="dashed", linewidth=1)
            plt.savefig("../out/fig_" + str(i) + ".pdf")
            plt.close()

        # Caluculate Regret
        simple_regret = y_true_max - y_max
        print("iter:" + str(i + 1) + " ", simple_regret)
        # inference_regret = y_true - max - func()
        # print('point: ',X_next)

        # Add sample to previous samples
        X_train = np.vstack((X_train, X_next))
        y_train = np.vstack((y_train, Y_next))

        # Update Gaussian process with existing samples
        model.set_XY(X_train, y_train)
        if i % 5 == 0:
            model.optimize_restarts(num_restarts=10, verbose=0)


def main():
    seed = 1
    train_num = 5
    n_iter = 5
    fig_ok = True

    experiment(seed, train_num, n_iter, fig_ok)


if __name__ == "__main__":
    main()
