import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import chi2_contingency
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


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


# ! 该函数负责对列联表进行Cochran_ArmitageTrendTest趋势检验
# ! 该检验用于检验两组有序多分类数据，各组中的发生率和对应的排序之间是否存在线性关系，p值小于置信水平说明线性关系显著
# ! 可以用于逻辑斯蒂回归之前
def Cochran_ArmitageTrendTest(data):
    # ! 列名顺序需已代表有序性
    # ! data是dataframe格式列联表
    # ! 调用R语言的函数prop.trend.test

    # 激活 pandas 到 R 的数据转换功能。
    # 这允许 rpy2 自动将 Pandas 对象（如 DataFrame）转换为对应的 R 对象。
    pandas2ri.activate()

    # * 导入 R 的 'coin' 包
    # * 注意：如果 'coin' 包未安装，importr 会失败
    try:
        coin = importr("coin")
    except Exception as e:
        print("Error: R package 'coin' not found.")
        print("Please install it in R: install.packages('coin')")
        raise e

    # * --- 数据准备 ---
    # * 根据 prop.trend.test 函数的输入要求，我们需要准备以下数据：
    # ! 1. 'x' 参数：一个向量，表示每个有序分类中“成功”（或第一组）的计数。
    # *     这里我们假定 DataFrame 的第一行包含了这些计数。
    # ?     这里data.ioc[0]代表dataframe列联表的第一行，.tolist()将该行转化为一个python列表
    # ?     ro.IntYector()作用是把这个列表转化为R的整数向量，这里相当于把列联表第一行转化为R格式的整数向量
    successes = ro.IntVector(data.iloc[0].tolist())

    # ! 2. 'n' 参数：一个向量，表示每个有序分类中“总观察次数”的计数。
    # *     这通过对 DataFrame 的每一列求和得到（即该分类下所有组的总计数）。
    # ?     列和转化为R形式的整数向量
    totals = ro.IntVector(data.sum(axis=0).tolist())

    # ! 3. 'score' 参数：一个向量，表示每个有序分类的“分数”或权重。
    # *     这些分数反映了分类的有序性，并且在趋势检验中是关键。
    # *     这里我们根据列的顺序（从左到右）简单地分配 1, 2, 3... 的等距分数。
    # *     例如，如果列是 ['低', '中', '高']，则分数分别为 1, 2, 3。
    # !     请务必确保 DataFrame 的列顺序与你希望的有序性相符。
    # !     如果有序性和原列联表的列顺序不同，也可以改这块代码自定义scores变量
    scores = ro.FloatVector(range(1, data.shape[1] + 1))

    # * --- 执行 R 检验 ---
    # * 导入 R 的 'stats' 包，它是 R 的基础包，无需单独安装。
    # * prop.trend.test 函数就包含在这个包中
    stats = importr("stats")

    # * 尝试调用 R 的 prop.trend.test 函数。
    # 如果在执行过程中发生任何 R 端的错误（例如数据格式不符合要求），会捕获异常。
    # ! 这里直接输出了结果，p值可以用trend_test_result.rx2("p.value")[0]调用，这里注释掉了
    # ! df是自由度，x-squared是卡方统计量的值
    try:
        trend_test_result = stats.prop_trend_test(x=successes, n=totals, score=scores)
        # // p_value = trend_test_result.rx2("p.value")[0]
    except Exception as e:
        print("Error running R's prop.trend.test. Check data format or scores.")
        raise e
    # // return p_value
    return trend_test_result


# ! 该函数用于调用scipy库对列联表进行卡方检验
# // 也没必要专门写函数因为已经有了，这里是为了更有结构感
def Chi2Test(data):
    chi2, p, dof, expected = chi2_contingency(data)
    return chi2, p, dof, expected


# ! 该函数负责把dataframe中的两列拼接在一起，然后按其分组制成列联表，默认组别只适用于本问题
def CrossTable(df, col_i, col_j):
    # ! df为dataframe格式，col_i,col_j为数字列索引
    Data1 = df.iloc[:, col_i].dropna()
    Data2 = df.iloc[:, col_j].dropna()
    # 将两列的列名合并
    Data2 = Data2.rename(Data1.name)
    # 转化Series为Dataframe格式
    Data1 = Data1.to_frame()
    Data2 = Data2.to_frame()
    # ! 组别设置，调用记得改
    Data1["是否颅内感染"] = False
    Data2["是否颅内感染"] = True
    # 上下拼接两个表格
    DataIntergrated = pd.concat([Data1, Data2], axis=0, ignore_index=True)
    Data = pd.crosstab(
        DataIntergrated.iloc[:, 1], DataIntergrated.iloc[:, 0], dropna=False
    )
    return Data


# ! 该函数为本文件主函数
if __name__ == "__main__":
    # 读入数据df
    df = pd.read_excel("IngratedTableForTest2.xlsx")

    # * 输入要检测的指标，选择相应行列，检验凝固性列索引选1,10，检验蛋白定性选列索引7,16，检验透明度选列索引8,17
    str = input("输入检验指标的一种：透明度，凝固性，蛋白定性：")
    if str == "透明度":
        col_i = 8
        col_j = 17
        temp = 1
    elif str == "蛋白定性":
        col_i = 7
        col_j = 16
        temp = 2
    elif str == "凝固性":
        col_i = 1
        col_j = 10
        temp = 3
    else:
        # 输入错误退出程序
        print("输入错误！请重新输入正确的值！")
        sys.exit()

    # ! 建立列联表，检验凝固性列索引选1,10，检验蛋白定性选列索引7,16，检验透明度选列索引8,17
    data = CrossTable(df, col_i, col_j)

    # ! 交换两列以获得有序分类变量,如果检验凝固性，检验另外两个就不需要
    if temp != 3:
        cols = data.columns.tolist()
        cols[0], cols[1] = cols[1], cols[0]
        data = data[cols]

    # 输出列联表
    print("\n")
    print("------- 列联表 -------")
    print(data)
    print("\n")

    # ! 进行广义Fisher精确检验，调用R语言库
    fisher_result = Generalized_FisherExactTest(data)
    print("广义fisher精确检验结果如下：")
    print(fisher_result)
    p1 = fisher_result.rx2("p.value")[0]
    if p1 <= 0.05:
        print(
            "p值%f小于置信水平0.05，说明两组数据的分布在统计学意义上存在显著差异"
            % (p1,)
        )
    else:
        print(
            "p值%f大于置信水平0.05，说明两组数据的分布在统计学意义上不存在显著差异"
            % (p1,)
        )
    print("\n")

    # ! 进行卡方检验，使用scipy库
    if temp != 3:
        chi2, p2, dof, expected = Chi2Test(data)
        print(
            "卡方检验结果如下：以下四个数据分别代表卡方统计量的值，p值，自由度和期望/理论频数："
        )
        print(chi2, p2, dof, "\n", expected)
        if p2 <= 0.05:
            print(
                "p值%f小于置信水平0.05，说明两组数据的分布在统计学意义上存在显著差异"
                % (p2,)
            )
        else:
            print(
                "p值%f大于置信水平0.05，说明两组数据的分布在统计学意义上不存在显著差异"
                % (p2,)
            )
        print("\n")
    elif temp == 3:
        print(
            "列联表单元格小于等于5的格子占比太高，不适用卡方检验！ 请参考广义Fisher精确检验！"
        )

    # ! 进行Cochran_ArmitageTrendTest（CAT）卡方趋势检验，调用R语言库
    print("\n", "\n")
    print("Cochran_ArmitageTrendTest（CAT）卡方趋势检验结果如下：")
    trend_test_result = Cochran_ArmitageTrendTest(data)
    print(trend_test_result)
    p3 = trend_test_result.rx2("p.value")[0]
    if p3 <= 0.05:
        print(
            "p值%f小于置信水平0.05，说明两组有序分类数据之间存在一个统计学上显著的线性趋势关联。"
            % (p3,)
        )
    else:
        print(
            "p值%f大于置信水平0.05，说明两组有序分类数据之间不存在统计学上显著的线性趋势关联。"
            % (p3,)
        )
    print("\n")
