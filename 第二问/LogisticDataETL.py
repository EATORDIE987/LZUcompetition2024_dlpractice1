import numpy as np
import pandas as pd

# 定义一个函数来获取序列的众数，如果众数不存在（例如，组内全为NaN），则返回NaN
def get_mode_or_na(series):
    mode = series.mode() # series.mode() 返回一个Series，可能包含多个众数
    if not mode.empty:
        return mode[0] # 如果有众数，返回第一个
    return np.nan # 如果没有众数（例如，该组内此列全为NaN），则返回NaN


# todo 读入治疗过程记录
Record = pd.read_excel("治疗过程记录.xlsx", header=1)
# todo 加入列‘是否患有颅内感染’
Record["是否患有颅内感染"] = Record.iloc[:, 4:15].eq("颅内感染").any(axis=1)
# todo 过滤多余行
Filtered_others = pd.concat(
    [Record.iloc[:,0],Record.iloc[:, 28], Record.iloc[:, 29], Record.iloc[:, 34], Record.iloc[:, 35]],
    axis=1,
)
# todo 保存为下一步ETL存为文件
Filtered_others.to_excel("Filtered_others.xlsx")
# todo 为构建完整的一一对应的数据集，只留下每个日期的检验项目首次出现的数据作为原材料
# ? .drop_duplicvates([],keep='first')是去掉重复行的意思，当某一行在选中的列上和另一行一样时，只保留第一行
Filtered_NotFirstOccurrence = Filtered_others.drop_duplicates(
    subset=['住院号',"项目名称", "报告日期"], keep="first"
)
Filtered_NotFirstOccurrence.to_excel("Filtered_NotFirstOccurrence.xlsx")
# todo 由于凝固性没有通过单因素分析，删去凝固性的数据
Filtered_OnePoint = Filtered_NotFirstOccurrence[
    Filtered_NotFirstOccurrence["项目名称"] != "凝固性"
]
Filtered_OnePoint["是否患有颅内感染"] = Filtered_OnePoint["是否患有颅内感染"].astype(
    int
)  # True 变为 1, False 变为 0
Filtered_OnePoint.to_excel("Filtered_OnePoint.xlsx")
X = Filtered_OnePoint.pivot_table(
    index=['住院号', '报告日期'],  # 将两列作为列表传递给 index
    columns='项目名称',
    values='定量结果',
    aggfunc='first' ,# 或者其他你需要的聚合函数
    sort=False
)
X.to_excel('X.xlsx')
Y = Filtered_OnePoint.groupby(['住院号','报告日期'])['是否患有颅内感染'].first()
# todo 确保 Y 的索引与 X 的索引一致
Y = Y.reindex(X.index)
Y.to_excel('Y.xlsx')
dataset_dataframe=pd.concat([X,Y],axis=1)
dataset_dataframe.to_excel('dataset_dataframe.xlsx')
num_col=['葡萄糖[Glu]','微量蛋白[MTP]','白细胞计数', '多个核细胞百分比', '氯[CL]']
str_col=['蛋白定性', '透明度']
for col in num_col:
    dataset_dataframe[col] = dataset_dataframe[col].astype('Float32')
    dataset_dataframe[col] = dataset_dataframe[col].fillna(
        dataset_dataframe.groupby('是否患有颅内感染')[col].transform('mean')
    )
for col in str_col:
    dataset_dataframe[col] = dataset_dataframe[col].astype('Float32')
    dataset_dataframe[col] = dataset_dataframe[col].fillna(
        dataset_dataframe.groupby('是否患有颅内感染')[col].transform(get_mode_or_na)
    )
dataset_dataframe['透明度']=dataset_dataframe['透明度']-242
dataset_dataframe['蛋白定性']=dataset_dataframe['蛋白定性']-248
dataset_dataframe.to_excel('dataset_fillna.xlsx')




