import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization

def objective_function(x):
    return (x - 2) ** 2

domain = [{'name': 'x', 'type': 'continuous', 'domain': (0, 10)}]

# バッチサイズを設定（例：3）
batch_size = 3

# 初期点を設定（例：ランダムに3点）
initial_points = 5

# 最適化の実行
optimizer = BayesianOptimization(
    f=objective_function,            # 目的関数
    domain=domain,                   # 探索空間
    model_type='GP',                 # ガウス過程を使用
    acquisition_type='EI',           # 獲得関数の選択（期待改善）
    acquisition_weight=2,            # 獲得関数の重み
    num_cores=batch_size,            # 並列実行のためのコア数
    initial_design_numdata=initial_points, # 初期点の設定
    exact_feval=True,                # 目的関数がノイズなし
)
optimizer.plot_convergence('./out.pdf')
optimizer.run_optimization(max_iter=50)

print("最適な値: x = %.4f, f(x) = %.4f" % (optimizer.X[-1], optimizer.Y[-1]))
print(optimizer.X)
