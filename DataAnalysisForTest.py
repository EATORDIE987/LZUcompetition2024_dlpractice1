import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# 去除重复指标数据
Record = pd.read_excel("治疗过程记录.xlsx", header=1)
Filter_NoRepeatData = Record.iloc[:, np.r_[0:25, 27:35]].drop_duplicates()
Filter_NoRepeatData["是否诊断颅内感染"] = (
    Record.iloc[:, 4:15].eq("颅内感染").any(axis=1)
)

# 未颅内感染患者数据（假设检验用）
Filter_NoInfection = Filter_NoRepeatData[
    Filter_NoRepeatData["是否诊断颅内感染"] == False
]
LongToWide_NoInfection = Filter_NoInfection.pivot(columns="项目名称", values="定性结果")
NoInfectionDataForTest = LongToWide_NoInfection.apply(
    lambda col: pd.Series(col.dropna().values)
)
NoInfectionDataForTest["是否颅内感染"] = False
NoInfectionDataForTest.to_excel("NoInfectionDataForTest.xlsx")
NoInfectionDataForTest.to_csv("NoInfectionDataForTest.csv")

# 颅内感染患者数据（假设检验用）
Filter_Infection = Filter_NoRepeatData[Filter_NoRepeatData["是否诊断颅内感染"] == True]
LongToWide_Infection = Filter_Infection.pivot(columns="项目名称", values="定性结果")
InfectionDataForTest = LongToWide_Infection.apply(
    lambda col: pd.Series(col.dropna().values)
)
InfectionDataForTest["是否颅内感染"] = True
InfectionDataForTest.to_excel("InfectionDataForTest.xlsx")
InfectionDataForTest.to_csv("InfectionDataForTest.csv")

# 二者按行列融合为两个用于假设检验的新表
IngratedTableForTest1 = pd.concat([NoInfectionDataForTest, InfectionDataForTest], axis=0)
IngratedTableForTest1.to_excel("IngratedTableForTest1.xlsx")
IngratedTableForTest1.to_csv("IngratedTableForTest1.csv")
IngratedTableForTest2 = pd.concat([NoInfectionDataForTest, InfectionDataForTest], axis=1)
IngratedTableForTest2.to_excel("IngratedTableForTest2.xlsx")
IngratedTableForTest2.to_csv("IngratedTableForTest2.csv")
