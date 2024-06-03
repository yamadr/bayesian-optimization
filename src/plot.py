import matplotlib.pyplot as plt
import sys, pickle
import os
from pathlib import Path


def main():
    # 各seedごと、各獲得関数ごとにSR値を算出
    input_dim = 2
    experiment_name = "Branin"
    acq_name = "EI"
    dir_path = (
        f"{Path(__file__).parents[0]}/../out/Test/dim_"
        + str(input_dim)
        + "/"
        + experiment_name
        + "/"
        + acq_name
    )
    seed = 10
    acq_list = os.listdir(dir_path)
    simple_regret_list = []
    for i in range(seed):
        seed_path = "/seed_" + str(i) + "/simple_regret.pickle"
        with open(dir_path + seed_path, mode="br") as f:
            simple_regret = pickle.load(f)
        simple_regret_list.append(simple_regret)
    print(simple_regret_list)


if __name__ == "__main__":
    main()
