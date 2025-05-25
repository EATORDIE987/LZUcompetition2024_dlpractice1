import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


def Generalized_FisherExactTest(data):
    pandas2ri.activate()
    np_array = data.values
    # 使用 R 的 'matrix' 函数来创建 R 矩阵
    # R 的 matrix 函数是按列填充的，所以需要将 numpy 数组扁平化 (flatten)
    # nrow 参数指定行数，与原始 DataFrame 的行数一致
    r_matrix = ro.r["matrix"](np_array.flatten(), nrow=data.shape[0], byrow=True)
    # 导入 R 的 'stats' 包，其中包含 fisher.test 函数
    stats = importr("stats")
    # 调用 fisher.test 函数
    # 对于大于 2x2 的列联表，R 的 fisher.test 会自动进行广义 Fisher 精确检验。
    # 推荐使用 simulate_p_value=True 来进行模拟计算 p 值，尤其当数据量较大时，可以避免耗时过长。
    # B 参数指定模拟的次数，通常设置为 10000 或更高。
    fisher_result = stats.fisher_test(r_matrix, simulate_p_value=True, B=10000)
    # 从结果中提取 p 值
    p_value = fisher_result.rx2("p.value")[0]  # 或者 fisher_result[0][0]
    return r_matrix,p_value

def CrossTable(df, col_i, col_j):
    # 要处理的数据
    Data1= df.iloc[:, col_i].dropna()
    Data2 = df.iloc[:, col_j].dropna()
    Data2 = Data2.rename(Data1.name)
    Data1 = Data1.to_frame()
    Data2 = Data2.to_frame()
    Data1["是否颅内感染"] = False
    Data2["是否颅内感染"] = True
    DataIntergrated = pd.concat([Data1, Data2], axis=0, ignore_index=True)
    Data = pd.crosstab(
        DataIntergrated.iloc[:, 1], DataIntergrated.iloc[:, 0], dropna=False
    )
    return Data

if __name__ == "__main__":
    df = pd.read_excel("IngratedTableForTest2.xlsx")
    data = CrossTable(df, 7, 16)
    print("--- 原始Pandas DataFrame ---")
    print(data)
    #进行广义Fisher精确检验
    '''r_matrix,p_value=Generalized_FisherExactTest(data)
    print('--- 转换后的R矩阵 ---')
    print(r_matrix)'''

    chi2, p, dof, expected = chi2_contingency(data)

    '''alpha = 0.05  # 显著性水平
    if p_value < alpha:
        print(
            f"P 值 ({p_value:.4f}) 小于显著性水平 ({alpha})，因此两组多分类数据的分布存在显著差异。"
        )
    else:
        print(
            f"P 值 ({p_value:.4f}) 大于显著性水平 ({alpha})，因此没有证据表明两组多分类数据的分布存在显著差异。"
        )'''
    alpha = 0.05
    if p < alpha:
        print(
            f"P 值 ({p:.4f}) 小于显著性水平 ({alpha})，因此两组多分类数据的分布存在显著差异。"
        )
    else:
        print(
            f"P 值 ({p:.4f}) 大于显著性水平 ({alpha})，因此没有证据表明两组多分类数据的分布存在显著差异。"
        )
