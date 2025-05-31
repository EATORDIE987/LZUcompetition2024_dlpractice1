import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
)  # 导入 cross_val_predict
from sklearn.metrics import roc_curve, auc  # Import roc_curve and auc
import matplotlib.pyplot as plt

df = pd.read_excel("FinalProcessData.xlsx")
X = df.iloc[:, 3:24]
Y = df.iloc[:, 24]

model = RandomForestClassifier(
    n_estimators=300,  # 森林中决策树的数量。通常越多越好，但会增加计算时间。
    max_features="sqrt",  # 寻找最佳分裂时考虑的特征数量。'sqrt' 表示 sqrt(总特征数)。
    # 'log2' 表示 log2(总特征数)。也可以是整数或浮点数。
    max_depth=None,  # 树的最大深度。None 表示节点会一直分裂，直到叶子节点纯净或达到min_samples_split。
    min_samples_split=2,  # 内部节点再划分所需最小样本数。
    min_samples_leaf=1,  # 叶节点所需的最小样本数。
    random_state=42,  # 随机种子，确保结果可复现。
    n_jobs=-1,  # 并行运行的作业数。-1 表示使用所有可用的处理器。
    oob_score=False,  # 是否使用袋外样本 (Out-of-Bag samples) 来估计泛化准确率。
    # 如果为 True，训练后可以通过 model.oob_score_ 查看。
)


# --- 3. 定义 K 折交叉验证分割器 ---
k = 5  # 设置折数 (K值)，例如 5 折或 10 折
shuffle_data = True  # 是否在分割数据前打乱数据顺序（推荐）
random_seed = 66  # 为打乱数据设置随机种子，确保可复现性
# 对于分类任务，强烈推荐使用 StratifiedKFold。
# 它会确保在每一折中，各个类别的样本比例与原始数据集中大致相同，
# 这对于类别不平衡的数据集尤其重要。
cv_splitter = StratifiedKFold(
    n_splits=k, shuffle=shuffle_data, random_state=random_seed
)

# --- 4. 执行 K 折交叉验证并获取分数 ---
# scoring 参数指定了评估模型性能的指标。
# 对于分类任务，常用的有: 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr' 等。
# 对于回归任务，常用的有: 'neg_mean_squared_error', 'r2', 'neg_mean_absolute_error' 等。
# (注意回归指标中的 'neg_' 前缀，因为 cross_val_score 试图最大化分数，所以误差指标取负值)
scoring_metric = "accuracy"  # 以准确率为例
# cross_val_score 函数会自动处理数据的分割、模型的训练和评估过程。
# 它可以直接接受 Pandas DataFrame 和 Series 作为输入 X 和 y。
scores = cross_val_score(
    estimator=model,  # 要评估的模型实例
    X=X,  # 特征 DataFrame
    y=Y,  # 目标 Series
    cv=cv_splitter,  # 交叉验证的分割策略 (我们定义的 StratifiedKFold 对象)
    # 也可以直接传入整数 k，例如 cv=5。此时，如果 model 是分类器，
    # cross_val_score 通常会自动使用 StratifiedKFold。
    scoring=scoring_metric,  # 评估指标
    n_jobs=-1,  # (可选) 并行计算的核数，-1 表示使用所有可用的核
)

# --- 5. 打印交叉验证的结果 ---
print(f"交叉验证评估指标: {scoring_metric}")
print(f"每一折的分数: {scores}")  # scores 是一个包含 K 个分数的 NumPy 数组
print(f"平均分数: {np.mean(scores):.4f}")
print(f"分数的标准差: {np.std(scores):.4f}")
print(
    f"\n总结：随机森林模型使用 {k}-折交叉验证，得到的平均 {scoring_metric} 为: {np.mean(scores):.4f} (标准差: {np.std(scores):.4f})"
)

oof_predictions = cross_val_predict(
    estimator=model,  # 与 cross_val_score 中使用相同的模型实例
    X=X,  # 特征 DataFrame
    y=Y,  # 目标 Series
    cv=cv_splitter,  # 与 cross_val_score 中使用相同的交叉验证分割策略
    method="predict_proba",  # 指定获取预测的类别标签。对于概率，使用 'predict_proba'
    n_jobs=-1,  # (可选) 并行计算的核数
)

print(f"已为全部 {len(oof_predictions)} 个样本生成折外预测。")
PREresult_df = pd.DataFrame(oof_predictions)
PREresult = PREresult_df.iloc[:, 1]

# 修正 PREresult 的处理，使其能够被正确地重命名和使用
temp_df_for_concat = pd.DataFrame({"颅内感染风险": PREresult.values}, index=df.index)
result1 = pd.concat([df, temp_df_for_concat], axis=1)
# 此时 result1 中就有了名为 "颅内感染风险" 的OOF预测概率列，后续代码可以按预期工作

result2 = result1[["住院号", "颅内感染风险"]].copy()

# 可以选平均数，改成.mean()
median_table = result2.groupby("住院号", as_index=False, sort=False)[
    "颅内感染风险"
].median()

# 3. (可选) 重命名包含中位数值的列，使其更清晰
#    当 as_index=False 时，聚合结果列名通常保持不变 (这里是 '感染概率')

median_table.to_excel("FinalResult.xlsx")

df2 = pd.read_excel("ProcessedData.xlsx")
df2 = df2[["住院号", "是否患有颅内感染"]]
compare = pd.concat([median_table, df2], axis=1)
compare.to_excel("compare.xlsx")
loss = abs(
    compare["颅内感染风险"].round().astype("int")
    - compare["是否患有颅内感染"].astype("int")
)
print("取中位数后预测准确率：", 1 - loss.sum() / len(loss))

# ---绘制ROC曲线---
# 注意：以下ROC曲线是基于模型对原始数据集中每个独立样本的
# 袋外（OOF）预测概率（PREresult）和原始真实标签（Y）生成的。
# 这直接反映了模型在进行任何后续按“住院号”聚合之前，在交叉验证中的原始预测性能。

# Y 是原始的真实标签 Series
# PREresult 是对应样本的类别为1的预测概率 Series
y_true_for_roc = Y
y_scores_for_roc = PREresult  # PREresult 已是阳性类别的概率 Series

# 计算ROC曲线的各个点和AUC值
fpr, tpr, thresholds = roc_curve(y_true_for_roc, y_scores_for_roc)
roc_auc_value = auc(fpr, tpr)

# 开始绘图
plt.figure(figsize=(9, 7))  # 设置图像大小
plt.plot(
    fpr,
    tpr,
    color="dodgerblue",
    lw=2,
    label=f"样本级别 OOF ROC (AUC = {roc_auc_value:.2f})",
)
plt.plot([0, 1], [0, 1], color="dimgray", lw=2, linestyle="--")  # 对角线（随机猜测）

# 设置图像属性
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])  # y轴上限略大于1，确保曲线完全可见
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC(OOF)")
plt.legend(loc="lower right")  # 图例位置
plt.grid(True, linestyle=":", alpha=0.6)  # 添加网格
plt.savefig("roc_curve.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()  # 显示图像
