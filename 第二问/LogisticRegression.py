import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt # 新增
from sklearn.metrics import roc_curve, auc # 新增

# * ---------------------------------- 获取ROC曲线所需数据 ----------------------------------
def get_roc_data(net, data_iter):
    # ... (如上定义) ...
    net.eval()
    all_true_labels = []
    all_pred_scores = []
    with torch.no_grad():
        for X_batch, y_batch in data_iter:
            outputs = net(X_batch)
            scores = outputs[:, 1] # 假设类别1是正类别
            all_true_labels.extend(y_batch.cpu().numpy())
            all_pred_scores.extend(scores.cpu().numpy())
    return np.array(all_true_labels), np.array(all_pred_scores)

# * ---------------------------------- 定义累加器的类 ----------------------------------
# ! 很重要的累加器，可用于别的程序
class Accumulator:  # @save
    def __init__(self, n):
        """
        # ? 构造函数 (初始化方法)。
        # ? 当创建一个 Accumulator 对象时，这个方法会被调用。
        参数:
        # ? n (int): 需要累加的变量的数量。例如，如果 n=2，则这个累加器可以同时追踪两个数值的累加。
        """
        # ? self.data 是一个实例变量，它被初始化为一个列表。
        # ? 这个列表包含 n 个元素，每个元素都被初始化为浮点数 0.0。
        # ? 这个列表将用来存储各个变量的累加值。
        # ? 例如，如果 Accumulator(3) 被调用，则 self.data 将是 [0.0, 0.0, 0.0]。
        self.data = [0.0] * n

    def add(self, *args):
        # ? zip(self.data, args) 会将 self.data 列表中的元素和传入的 args 元组中的元素
        # ? 一一配对。例如，如果 self.data = [a1, a2] 而 args = (b1, b2)，
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        # todo 将所有累加的变量重置为 0.0
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        """
        # ! 使得 Accumulator 对象可以使用方括号索引来获取特定变量的累加值。
        这是 Python 中的一个“魔术方法”(magic method) 或“双下划线方法”(dunder method)。
        参数:
        # ? idx (int): 要获取的累加变量的索引 (从 0 开始)。
        返回:
        # ? float: 索引 idx 对应的变量的当前累加值。
        """
        # ? 直接返回 self.data 列表中索引为 idx 的元素。
        # ? 例如，如果 acc 是一个 Accumulator 对象，acc[0] 就会调用这个方法并返回 self.data[0]。
        return self.data[idx]

# * ---------------------------------- 定义精确率函数 ----------------------------------
# todo 计算精确率
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# * ---------------------------------- 计算在指定数据集上模型的精度 ----------------------------------
# todo 计算在指定数据集上模型的精确率
def evaluate_accuracy(net, data_iter):
    """
    参数:
    # ? net (torch.nn.Module): 需要评估的 PyTorch 模型。
    # ? data_iter (torch.utils.data.DataLoader): 包含数据的数据加载器，它会迭代产生一批批的特征 (X) 和标签 (y)。
    返回:
    # ? float: 模型在该数据集上的整体准确率。
    """

    # ? 1. 检查模型是否为 PyTorch 的 nn.Module 类型
    # ?     这是为了确保我们传入的是一个合法的 PyTorch 模型
    if isinstance(net, torch.nn.Module):
        # ? 2. 将模型设置为评估模式 (evaluation mode)
        # ?     这非常重要，因为它会关闭一些在训练时启用但在评估时应禁用的层，
        # ?     例如 Dropout 层和 BatchNorm 层（在评估时会使用其学到的全局统计量而不是当前批次的统计量）。
        # ?     如果不设置 net.eval()，评估结果可能会不准确或不稳定。
        net.eval()
        # ? 3. 初始化一个累加器 (Accumulator) 对象
        # ?     这个 Accumulator 类（通常由 d2l 库提供或用户自定义）用于累积两个值：
        # ?     metric[0]: 累积正确预测的样本数量
        # ?     metric[1]: 累积总的预测样本数量
        # ?     这里的 Accumulator(2) 表示它内部维护一个长度为 2 的列表或数组来存储这两个累加值。
    metric = Accumulator(2)
    # ? 4. 禁用梯度计算
    # ?     在模型评估阶段，我们不需要计算梯度，因为我们不会进行参数更新（反向传播）。
    # ?     torch.no_grad() 上下文管理器可以临时关闭所有涉及的张量的 requires_grad 属性，
    # ?     从而减少内存消耗并加速计算。
    with torch.no_grad():
        for X, y in data_iter:
            # ? 6. 进行预测并累加结果
            # ?   a. net(X): 将当前批次的特征 X 输入到模型 net 中，得到模型的预测输出。对于分类问题，这通常是每个类别的原始分数 (logits) 或概率。
            # ?   b. y.numel():计算真实标签张量 y 中元素的总数量。在一个批次中，这通常等于该批次的样本数量。
            # ?   c. metric.add(num_correct_in_batch, num_samples_in_batch):
            # ?      将当前批次的正确预测数和样本总数添加到累加器 metric 中。
            # ?      metric[0] 会加上 num_correct_in_batch
            # ?      metric[1] 会加上 num_samples_in_batch
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

dataset_dataframe=pd.read_excel('dataset_fillna.xlsx')
feature_columns = ['葡萄糖[Glu]', '微量蛋白[MTP]', '蛋白定性', '透明度', '白细胞计数', '多个核细胞百分比', '氯[CL]']
lables_columns=['是否患有颅内感染']
X_df = dataset_dataframe[feature_columns]
Y_series = dataset_dataframe[lables_columns]

X = torch.tensor(X_df.values).to(torch.float32)
Y = torch.tensor(Y_series.values.squeeze())
dataset=data.TensorDataset(X,Y)
total_size = len(dataset)
train_ratio = 0.8  # 80% 作为训练集
test_ratio = 0.2   # 20% 作为测试集
split_ratios = [train_ratio, test_ratio]

torch.manual_seed(888) # 设置随机种子
train_dataset, test_dataset = data.random_split(dataset, split_ratios)

batch_size=64
train_iter=data.DataLoader(train_dataset,batch_size,shuffle=True,num_workers=4)
test_iter=data.DataLoader(test_dataset,batch_size,shuffle=False,num_workers=4)

net=nn.Sequential(nn.Linear(7,2))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
loss=nn.CrossEntropyLoss(reduction='mean')
lr = 0.007

trainer=torch.optim.SGD(net.parameters(),lr)
scheduler=torch.optim.lr_scheduler.StepLR(trainer,step_size=10,gamma=0.02)

num_epochs=14
for epoch in range(num_epochs):
    net.train()
    for x,y in train_iter:
        l=loss(net(x),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    scheduler.step()
    print(f"epoch{epoch+1}:Accuracy:{evaluate_accuracy(net, test_iter)}", "\n")
# --- 在这里添加ROC曲线的绘制 ---
print("\n训练完成，开始生成和绘制ROC曲线...")
y_true, y_scores = get_roc_data(net, test_iter)
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("roc_curve.png")
print(f"ROC曲线已保存为 roc_curve.png, AUC = {roc_auc:.4f}")
plt.show()