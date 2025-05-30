import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import sys

# todo 本程序负责处理手术类型和是否感染的关系


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

OprationData = Record.drop_duplicates(
    subset=[
        "住院号",
        "手术名称1",
        "麻醉方式1",
        "手术名称2",
        "麻醉方式2",
        "手术名称3",
        "麻醉方式3",
        "手术名称4",
        "麻醉方式4",
        "是否患有颅内感染",
    ],
    keep="first",
)[
    [
        "手术名称1",
        "麻醉方式1",
        "手术名称2",
        "麻醉方式2",
        "手术名称3",
        "麻醉方式3",
        "手术名称4",
        "麻醉方式4",
        "是否患有颅内感染",
    ]
].copy()
# // OprationData.to_excel('OprationTypeData.xlsx')

OprationTypeData=OprationData.iloc[:,0:7]
all_Operation =set()
for col in OprationTypeData.columns:
    # 清理空字符串和NaN值，并确保是字符串类型
    valid_Opration = OprationTypeData[col].astype(str).str.strip()
    valid_Opration = valid_Opration[valid_Opration.ne('') & valid_Opration.ne('nan') & valid_Opration.notna()]
    all_Operation.update(valid_Opration.unique())
unique_Opration = sorted(list(all_Operation))

print(f"提取到的不重复手术或麻醉类型共有 {len(unique_Opration)} 种: {unique_Opration}")
print("-" * 50)

# --- 3. 为每种独特诊断创建列联表并进行卡方检验 ---
p_value_lessthanalpha = [] # 用于存储结果
for specific_Opration in unique_Opration:
    # print(f"\n正在分析诊断: 【{specific_diagnosis}】")

    # 重要：如果当前分析的 specific_diagnosis 就是 "颅内感染" 本身，
    # 并且 infection_status_col_name 列就是基于它生成的，
    # 那么它们之间必然存在完美关联，卡方检验可能无意义或出错。
    # 你可以根据需要选择跳过对 "颅内感染" 本身的分析，
    # 因为你关注的是 *其他* 诊断与颅内感染的关系。

    # 3a. 创建临时指示列：标记哪些患者具有当前正在分析的 specific_diagnosis
    #     检查 diagnosis_data 的所有列中是否包含当前的 specific_diagnosis
    series = OprationData.astype(str).eq(specific_Opration).any(axis=1)

    # 3b. 构建列联表
    #     行: 是否患有 specific_diagnosis (True/False)
    #     列: 是否患有颅内感染_目标 (True/False)
    contingency_table = pd.crosstab(series, OprationData['是否患有颅内感染'])
    fisher_result = Generalized_FisherExactTest(contingency_table)
    if fisher_result.rx2("p.value")[0]<=0.05:
        print(specific_Opration,'\n')
        print("列联表:")
        print(contingency_table)
        print('p值=',fisher_result.rx2("p.value")[0])
        p_value_lessthanalpha.append({
            '诊断': str(specific_Opration),
            'P值': fisher_result.rx2("p.value")[0],
            '列联表': contingency_table,
        })

print(p_value_lessthanalpha)
