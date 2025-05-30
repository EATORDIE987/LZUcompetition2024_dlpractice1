import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

Record = pd.read_excel("治疗过程记录.xlsx")
# todo 加入列‘是否患有颅内感染’
Record["是否患有颅内感染"] = Record.iloc[:, 4:15].eq("颅内感染").any(axis=1)
ProcessedData = Record.drop_duplicates(
    subset=[
        "住院号",
        "性别",
        "年龄",
        "住院天数",
        "主要诊断",
        "其他诊断1",
        "其他诊断2",
        "其他诊断3",
        "其他诊断4",
        "其他诊断5",
        "其他诊断6",
        "其他诊断7",
        "其他诊断8",
        "其他诊断9",
        "其他诊断10",
        "手术名称1",
        "麻醉方式1",
        "手术持续时间（h）",
        "手术名称2",
        "麻醉方式2",
        "手术名称3",
        "麻醉方式3",
        "手术名称4",
        "麻醉方式4",
        "手术持续时间4",
        "是否患有颅内感染",
    ],
    keep="first",
)[
    "住院号",
    "性别",
    "年龄",
    "住院天数",
    "主要诊断",
    "其他诊断1",
    "其他诊断2",
    "其他诊断3",
    "其他诊断4",
    "其他诊断5",
    "其他诊断6",
    "其他诊断7",
    "其他诊断8",
    "其他诊断9",
    "其他诊断10",
    "手术名称1",
    "麻醉方式1",
    "手术持续时间（h）",
    "手术名称2",
    "麻醉方式2",
    "手术名称3",
    "麻醉方式3",
    "手术名称4",
    "麻醉方式4",
    "手术持续时间4",
    "是否患有颅内感染",
]
ProcessedData.to_excel('ProcessedData.xlsx')
