import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier # 导入随机森林分类器
# 如果是回归任务，请使用: from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score # 导入交叉验证工具
from sklearn.datasets import make_classification # 用于生成示例数据集，方便演示

df=pd.read_excel('FinalProcessData.xlsx')
X=df.iloc[:,3:24]
Y=df.iloc[:,24]

model = RandomForestClassifier(
    n_estimators=300,       # 森林中决策树的数量。通常越多越好，但会增加计算时间。
    max_features='sqrt',    # 寻找最佳分裂时考虑的特征数量。'sqrt' 表示 sqrt(总特征数)。
                            # 'log2' 表示 log2(总特征数)。也可以是整数或浮点数。
    max_depth=None,         # 树的最大深度。None 表示节点会一直分裂，直到叶子节点纯净或达到min_samples_split。
    min_samples_split=2,    # 内部节点再划分所需最小样本数。
    min_samples_leaf=1,     # 叶节点所需的最小样本数。
    random_state=42,        # 随机种子，确保结果可复现。
    n_jobs=-1,              # 并行运行的作业数。-1 表示使用所有可用的处理器。
    oob_score=False         # 是否使用袋外样本 (Out-of-Bag samples) 来估计泛化准确率。
                            # 如果为 True，训练后可以通过 model.oob_score_ 查看。
)


# --- 3. 定义 K 折交叉验证分割器 ---
k = 5  # 设置折数 (K值)，例如 5 折或 10 折
shuffle_data = True # 是否在分割数据前打乱数据顺序（推荐）
random_seed = 66    # 为打乱数据设置随机种子，确保可复现性
# 对于分类任务，强烈推荐使用 StratifiedKFold。
# 它会确保在每一折中，各个类别的样本比例与原始数据集中大致相同，
# 这对于类别不平衡的数据集尤其重要。
cv_splitter = StratifiedKFold(n_splits=k, shuffle=shuffle_data, random_state=random_seed)

# --- 4. 执行 K 折交叉验证并获取分数 ---
# scoring 参数指定了评估模型性能的指标。
# 对于分类任务，常用的有: 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr' 等。
# 对于回归任务，常用的有: 'neg_mean_squared_error', 'r2', 'neg_mean_absolute_error' 等。
# (注意回归指标中的 'neg_' 前缀，因为 cross_val_score 试图最大化分数，所以误差指标取负值)
scoring_metric = 'accuracy' # 以准确率为例
# cross_val_score 函数会自动处理数据的分割、模型的训练和评估过程。
# 它可以直接接受 Pandas DataFrame 和 Series 作为输入 X 和 y。
scores = cross_val_score(
    estimator=model,        # 要评估的模型实例
    X=X,                    # 特征 DataFrame
    y=Y,                    # 目标 Series
    cv=cv_splitter,         # 交叉验证的分割策略 (我们定义的 StratifiedKFold 对象)
                            # 也可以直接传入整数 k，例如 cv=5。此时，如果 model 是分类器，
                            # cross_val_score 通常会自动使用 StratifiedKFold。
    scoring=scoring_metric, # 评估指标
    n_jobs=-1               # (可选) 并行计算的核数，-1 表示使用所有可用的核
)

# --- 5. 打印交叉验证的结果 ---
print(f"交叉验证评估指标: {scoring_metric}")
print(f"每一折的分数: {scores}") # scores 是一个包含 K 个分数的 NumPy 数组
print(f"平均分数: {np.mean(scores):.4f}")
print(f"分数的标准差: {np.std(scores):.4f}")
print(f"\n总结：随机森林模型使用 {k}-折交叉验证，得到的平均 {scoring_metric} 为: {np.mean(scores):.4f} (标准差: {np.std(scores):.4f})")