import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve  # Ensure this is imported
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    accuracy_score,
)  # 导入 accuracy_score

# 假设 X_df 是你的 680*21 特征 DataFrame
# 假设 y_series 是你的 680*1 目标 Series (值为 0 或 1)

# 为了演示，我们创建一些模拟数据
# 在实际使用中，请替换成你自己的 X_df 和 y_series
data = pd.read_excel("FinalProcessData.xlsx")

X_df = data.iloc[:, 3:24].copy()
y_series = data["是否患有颅内感染"].copy()

# 将DataFrame和Series转换为Numpy数组
X = X_df.values
y = y_series.values

# 初始化用于存储袋外预测概率的数组 (类别为1的概率)
oof_preds_proba = np.zeros(X.shape[0])
# 初始化用于存储袋外预测类别的数组 (0 或 1)
oof_preds_class = np.zeros(X.shape[0])

# 初始化用于存储每折验证分数的列表
fold_val_losses = []
fold_val_auc = []
fold_val_accuracies = []  # 新增: 存储每折的准确率

# 设置交叉验证的折数
n_folds = 5
# 使用StratifiedKFold确保每折中类别比例与整体数据集相似
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# XGBoost 参数说明 (使用 sklearn 接口的 XGBClassifier)
# 这些参数与直接使用xgboost库时的参数名称和含义基本一致
params = {
    "objective": "binary:logistic",  # 目标函数: 'binary:logistic' 用于二分类问题，输出概率。
    "eval_metric": "logloss",  # 评估指标: 早停时主要关注的指标。也可以是 'auc', 'error'。
    # 当使用 XGBClassifier 的 fit 方法时，eval_metric 可以在 fit 中指定，
    # 或者在模型初始化时指定。若在fit中指定了 eval_set，则此处的 eval_metric 将用于早停。
    "learning_rate": 0.07,  # 学习率 (同eta)。
    "max_depth": 5,  # 每棵树的最大深度。
    "subsample": 0.8,  # 训练每棵树时样本的采样比例。
    "colsample_bytree": 0.8,  # 构建每棵树时特征的采样比例。
    "min_child_weight": 1,  # 叶子节点最小权重和。
    "gamma": 0,  # 节点分裂所需的最小损失降低。
    "reg_lambda": 1,  # L2 正则化项的权重。
    "reg_alpha": 0,  # L1 正则化项的权重。
    "random_state": 50,  # 随机种子 (sklearn接口中通常用random_state)。
    # 'n_estimators': 1000,          # 树的数量。通过早停确定。
    # 'use_label_encoder': False     # 从XGBoost 1.3.0开始，如果y是整数标签，建议设置为False避免警告。
    # XGBClassifier在较新版本中默认会处理好标签编码。
    # 对于XGBoost >= 1.6.0，use_label_encoder 参数已被弃用，不再需要设置。
}

# 开始交叉验证
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n----- 第 {fold+1} 折 -----")

    # 划分训练集和验证集
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # 初始化并训练XGBoost模型 (使用XGBClassifier)
    model = xgb.XGBClassifier(
        **params,
        n_estimators=2000,  # 设置一个较大的初始值，由早停来决定实际的树数量
        early_stopping_rounds=100,  # 如果验证集性能在50轮内没有提升，则停止训练
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],  # 评估集，用于早停
        verbose=False,  # 可以设置为True或数字(如100)来打印训练过程中的评估结果
    )

    # 在验证集上进行预测
    # 1. 预测概率 (用于LogLoss, AUC, 和OOF概率存储)
    val_preds_proba = model.predict_proba(X_val)[:, 1]  # 取类别为1的概率

    # 2. 预测类别 (用于Accuracy)
    # predict 方法直接输出类别 (0或1)，基于默认的0.5阈值
    val_preds_class = model.predict(X_val)

    # 将当前折的验证集预测结果存储到袋外预测数组中
    oof_preds_proba[val_idx] = val_preds_proba
    oof_preds_class[val_idx] = val_preds_class  # 存储预测的类别

    # 计算并打印当前折的评估指标
    loss = log_loss(y_val, val_preds_proba)
    auc = roc_auc_score(y_val, val_preds_proba)
    accuracy = accuracy_score(y_val, val_preds_class)  # 计算准确率

    fold_val_losses.append(loss)
    fold_val_auc.append(auc)
    fold_val_accuracies.append(accuracy)  # 存储准确率

    print(f"第 {fold+1} 折 - 验证集 LogLoss: {loss:.4f}")
    print(f"第 {fold+1} 折 - 验证集 AUC: {auc:.4f}")
    print(f"第 {fold+1} 折 - 验证集 准确率: {accuracy:.4f}")  # 打印准确率


# 交叉验证完成，计算并打印整体的袋外预测性能
mean_loss = np.mean(fold_val_losses)
mean_auc = np.mean(fold_val_auc)
mean_accuracy = np.mean(fold_val_accuracies)  # 计算平均准确率

# 计算整体OOF指标
# 对于LogLoss和AUC，使用存储的OOF概率
oof_overall_loss = log_loss(y, oof_preds_proba)
oof_overall_auc = roc_auc_score(y, oof_preds_proba)
# 对于Accuracy，可以使用存储的OOF类别，或者基于OOF概率重新确定类别
# 这里我们直接用存储的OOF类别 oof_preds_class
oof_overall_accuracy = accuracy_score(y, oof_preds_class)
# 或者，如果你想基于oof_preds_proba以0.5为阈值计算准确率：
# oof_overall_accuracy_from_proba = accuracy_score(y, (oof_preds_proba > 0.5).astype(int))


print("\n----- 交叉验证总结 -----")
print(f"平均验证集 LogLoss: {mean_loss:.4f}")
print(f"平均验证集 AUC: {mean_auc:.4f}")
print(f"平均验证集 准确率: {mean_accuracy:.4f}")  # 打印平均准确率

print(f"\n整体袋外 (OOF) LogLoss: {oof_overall_loss:.4f}")
print(f"整体袋外 (OOF) AUC: {oof_overall_auc:.4f}")
print(f"整体袋外 (OOF) 准确率: {oof_overall_accuracy:.4f}")  # 打印整体OOF准确率
# print(f"整体袋外 (OOF) 准确率 (基于概率0.5阈值): {oof_overall_accuracy_from_proba:.4f}")


# 参数选择标准说明:
# (与之前版本类似，这里不再重复，主要集中在模型参数和训练过程)
# 关键点:
# - `XGBClassifier` 是 XGBoost 的 scikit-learn 包装器，它使得 XGBoost 可以像其他 sklearn 模型一样使用。
# - `objective='binary:logistic'` 用于二分类并输出概率。
# - `eval_metric` 用于早停时的评估。
# - `early_stopping_rounds` 在 `fit` 方法中与 `eval_set` 配合使用，是防止过拟合和自动确定 `n_estimators` 的好方法。
# - 准确率 (`accuracy_score`) 是一个直观的分类指标，表示正确分类的样本比例。
#   但对于不平衡数据集，准确率可能会有误导性，此时 AUC 或 F1-score 可能更合适。
# - LogLoss (对数损失) 衡量的是预测概率与真实标签之间的差异，对概率的准确性更敏感。


print("\n----- 交叉验证总结 -----")
print(f"平均验证集 LogLoss: {mean_loss:.4f}")
print(f"平均验证集 AUC: {mean_auc:.4f}")
print(f"平均验证集 准确率: {mean_accuracy:.4f}")

print(f"\n整体袋外 (OOF) LogLoss: {oof_overall_loss:.4f}")
print(f"整体袋外 (OOF) AUC: {oof_overall_auc:.4f}")
print(f"整体袋外 (OOF) 准确率: {oof_overall_accuracy:.4f}")

temp_df_for_concat = pd.DataFrame({"颅内感染风险": oof_preds_proba}, index=data.index)
result1 = pd.concat([data, temp_df_for_concat], axis=1)
# 此时 result1 中就有了名为 "颅内感染风险" 的OOF预测概率列，后续代码可以按预期工作

result2 = result1[["住院号", "颅内感染风险"]].copy()

# 可以选平均数，改成.mean()
median_table = result2.groupby("住院号", as_index=False, sort=False)[
    "颅内感染风险"
].median()

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


# --- 新增绘制ROC曲线的代码 ---
print("\n----- 正在绘制ROC曲线 -----")

# 计算 ROC 曲线的点
# roc_curve 返回 fpr, tpr, 和 thresholds
# fpr: False Positive Rate (假阳性率)
# tpr: True Positive Rate (真阳性率) / Recall (召回率)
fpr, tpr, thresholds = roc_curve(y, oof_preds_proba)

# 开始绘图
plt.figure(figsize=(8, 6))  # 设置图像大小

# 绘制 ROC 曲线
# 使用 OOF AUC 作为图例中的 AUC 值
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=2,
    label=f"OOF ROC curve (area = {oof_overall_auc:.2f})",
)

# 绘制随机猜测的对角线
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

# 设置坐标轴范围和标签
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])  # y轴略微超出1.0，以确保曲线顶部可见
plt.xlabel("False Positive Rate (1 - Specificity)")  # 横轴：假阳性率
plt.ylabel("True Positive Rate (Sensitivity)")  # 纵轴：真阳性率
plt.title("Receiver Operating Characteristic (ROC) - OOF Predictions")  # 标题
plt.legend(loc="lower right")  # 图例位置
plt.grid(True)  # 显示网格
plt.savefig("roc_curve.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()  # 显示图像
