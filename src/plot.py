import matplotlib.pyplot as plt
import sys, pickle
import numpy as np
import os
from pathlib import Path


def main():
    # 各seedごと、各獲得関数ごとにSR値を算出
    input_dim = 2
    experiment_name = "Branin"
    dir_path = (
        f"{Path(__file__).parents[0]}/../out/Test/dim_"
        + str(input_dim)
        + "/"
        + experiment_name
        + "/"
    )
    seed = 10
    acq_list = os.listdir(dir_path)
    color_list = ['b','r']
    for acq, color in zip(acq_list,color_list):
        for i in range(seed):
            seed_path = acq+"/seed_" + str(i) + "/simple_regret.pickle"
            with open(dir_path + seed_path, mode="br") as f:
                simple_regret = pickle.load(f)
                if i == 0:
                    simple_regret_list = simple_regret
                else:
                    simple_regret_list = np.vstack((simple_regret_list,simple_regret))
        mean = simple_regret_list.sum(axis=0)/(seed+1)
        SE = np.std(simple_regret_list,axis=0,ddof=1)/np.sqrt(seed+1)
        iter = np.arange(len(mean))
        plt.plot(iter,mean,color=color)
        plt.fill_between(iter,mean+1.96*SE,mean-1.96*SE,alpha = 0.3,color=color)
    plt.savefig(dir_path+'/simple_regret.pdf')
    plt.close()

if __name__ == "__main__":
    main()
