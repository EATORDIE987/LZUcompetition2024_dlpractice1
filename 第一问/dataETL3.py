import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import sys

# todo 本程序负责处理手术时间和是否感染的关系


# ! 该函数用于把dataframe数值表格转化为R矩阵
def Dataframe_To_Rmatrix(data):
    # ! data为dataframe格式
    pandas2ri.activate()
    np_array = data.values

    # * 使用 R 的 'matrix' 函数来创建 R 矩阵
    # * R 的 matrix 函数是按列填充的，所以需要将 numpy 数组扁平化 (flatten)
    # * nrow 参数指定行数，与原始 DataFrame 的行数一致
    r_matrix = ro.r["matrix"](np_array.flatten(), nrow=data.shape[0], byrow=True)
    # ! 输出转换后的R矩阵检验转换是否正确,这里注释掉了
    # // print("--- 转换后的R矩阵 ---")
    # // print(r_matrix)
    return r_matrix


# ! 该函数用于调用R的广义fisher检验函数
def Generalized_FisherExactTest(data):
    # ! data是dataframe格式的列联表
    # * 转换为R矩阵
    r_matrix = Dataframe_To_Rmatrix(data)

    stats = importr("stats")
    # 调用 fisher.test 函数
    # 对于大于 2x2 的列联表，R 的 fisher.test 会自动进行广义 Fisher 精确检验。
    # ! 推荐使用 simulate_p_value=True 来进行模拟计算 p 值，尤其当数据量较大时，可以避免耗时过长。
    # ! B 参数指定蒙特卡洛模拟的次数，通常设置为 10000 或更高。
    fisher_result = stats.fisher_test(r_matrix, simulate_p_value=True, B=10000)
    # ! 从结果中提取 p 值,默认直接输出报告，但你可以只输出p值，调用p值用fisher_result.rx2("p.value")[0]即可,这里注释掉了
    # // p_value = fisher_result.rx2("p.value")[0]  # 或者 fisher_result[0][0]
    # // return p_value
    return fisher_result


Record = pd.read_excel("治疗过程记录.xlsx", header=1)
# todo 加入列‘是否患有颅内感染’
Record["是否患有颅内感染"] = Record.iloc[:, 4:15].eq("颅内感染").any(axis=1)

OperationTime_data = Record.drop_duplicates(
    subset=["住院号", "手术持续时间（h）", "手术持续时间4"], keep="first"
)[["住院号", "手术持续时间（h）", "手术持续时间4", "是否患有颅内感染"]]

df_part2 = OperationTime_data[["手术持续时间4", "是否患有颅内感染"]].copy()
# todo 重命名手术时间列为与上面相同的统一名称
df_part2.rename(columns={"手术持续时间4": "手术持续时间（h）"}, inplace=True)

OperationTime_data = pd.concat(
    [
        OperationTime_data[["手术持续时间（h）", "是否患有颅内感染"]],
        df_part2,
    ],
    axis=0,
)
OperationTime_data = OperationTime_data.dropna(subset=["手术持续时间（h）"])
OperationTime_data = OperationTime_data[OperationTime_data["手术持续时间（h）"]!='-']
OperationTime_data.to_csv("OperationTime_data.csv")
