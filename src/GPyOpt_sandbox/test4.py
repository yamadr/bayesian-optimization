import GPy
import GPyOpt
import numpy as np
import matplotlib.pyplot as plt
from GPyOpt.methods import BayesianOptimization

# 目的関数の定義
def objective_function(x):
    return (x - 2) ** 2

# 真の最適値（単純な二次関数なので簡単にわかる）
true_minimum = 2.0
true_minimum_value = objective_function(np.array([[true_minimum]]))

# 探索空間の定義
domain = [{'name': 'x', 'type': 'continuous', 'domain': (0, 10)}]

# バッチサイズの設定
batch_size = 3

# 初期点の設定
initial_points = 5

# ベイズ最適化の設定
optimizer = BayesianOptimization(
    f=objective_function,            # 目的関数
    domain=domain,                   # 探索空間
    model_type='GP',                 # ガウス過程を使用
    acquisition_type='EI',           # 獲得関数の選択（期待改善）
    acquisition_weight=2,            # 獲得関数の重み
    num_cores=1,                     # 並列実行のためのコア数（1に設定）
    initial_design_numdata=initial_points, # 初期点の設定
    exact_feval=True,                # 目的関数がノイズなし
)

# 最大イテレーション数
max_iter = 50

# Simple Regretを保存するリスト
simple_regrets = []

# 最適化の実行とSimple Regretの計算
for i in range(max_iter):
    # バッチサイズ分の次の観測点を提案
    next_points = optimizer.suggest_next_locations()
    
    print(f"Iteration {i+1}: Next points to evaluate x = {next_points}")
    
    # 提案された点を評価し、最適化器に追加
    Y_next = np.array([objective_function(x) for x in next_points])
    optimizer.X = np.vstack((optimizer.X, next_points))
    optimizer.Y = np.vstack((optimizer.Y, Y_next))
    
    # モデルの更新
    optimizer.updateModel(optimizer.X,optimizer.Y)
    
    # Simple Regretの計算
    current_minimum_value = optimizer.Y.min()
    simple_regret = current_minimum_value - true_minimum_value
    simple_regrets.append(simple_regret)

# Simple Regretのプロット
#plt.figure(figsize=(10, 5))
#plt.plot(simple_regrets, marker='o')
#plt.xlabel('Iteration')
#plt.ylabel('Simple Regret')
#plt.title('Simple Regret over Iterations')
#plt.grid(True)
#plt.show()

# 最終的な結果の表示
print("最適な値: x = %.4f, f(x) = %.4f" % (optimizer.X[np.argmin(optimizer.Y)], optimizer.Y.min()))

