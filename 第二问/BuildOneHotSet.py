import pandas as pd

df = pd.read_excel('ProcessedData.xlsx')

# todo 定义高风险手术类型列表
high_risk_surgery = [
    '侧脑室脑池造口引流术',
    '内镜下面神经微血管减压术',
    '脑室Ommaya泵置入术',
    '脑室分流管去除术'
]

# todo 定义高风险诊断类型列表
high_risk_diagnosis = [
    '外展神经损伤',
    '脑实质出血继发蛛网膜下腔出血',
    '脑室腹腔分流管置入感染',
    '脑积水',
    '面肌痉挛',
    '额叶交界性肿瘤'
]

# todo 进行独热编码和列筛选所必需的原始列名
essential_cols_for_processing = ['住院号', '主要诊断', '其他诊断1', '其他诊断2', '其他诊断3', '其他诊断4', '其他诊断5', '其他诊断6', '其他诊断7', '其他诊断8', '其他诊断9', '其他诊断10', '手术名称1', '手术名称2', '手术名称3', '手术名称4']
        
# todo 创建一个新的DataFrame，首先只包含 '住院号' 列，后续将添加其他处理过的列
# ! 使用 .copy() 避免SettingWithCopyWarning
df_processed = df[['住院号']].copy() 

# todo 定义包含诊断信息和手术信息的列名列表
diagnosis_column_names = [
    '主要诊断', '其他诊断1', '其他诊断2', '其他诊断3', '其他诊断4', 
    '其他诊断5', '其他诊断6', '其他诊断7', '其他诊断8', '其他诊断9', '其他诊断10'
]
surgery_column_names = [
    '手术名称1', '手术名称2', '手术名称3', '手术名称4'
]

for diag_item in high_risk_diagnosis:
    one_hot_col_name = f'诊断_{diag_item}'
    # todo 初始化一个全为False的Series，长度与DataFrame相同
    combined_check_diag = pd.Series([False] * len(df), index=df.index)
    for col_name in diagnosis_column_names:
        # 将当前列的检查结果（是否包含特定诊断项）通过逻辑或（|）合并到总的检查结果中
        # .astype(str) 转换确保str.contains可以处理NaN等非字符串类型
        # na=False 表示NaN值不被视为包含该字符串
        combined_check_diag = combined_check_diag | df[col_name].astype(str).str.contains(diag_item, na=False, regex=False)
    df_processed[one_hot_col_name] = combined_check_diag.astype(int) # 将布尔结果转换为0或1

for surg_item in high_risk_surgery:
    one_hot_col_name = f'手术_{surg_item}'
    combined_check_surg = pd.Series([False] * len(df), index=df.index)
    for col_name in surgery_column_names:
        if col_name in df.columns:
            combined_check_surg = combined_check_surg | df[col_name].astype(str).str.contains(surg_item, na=False, regex=False)
    df_processed[one_hot_col_name] = combined_check_surg.astype(int)

# 将目标变量 '是否颅内感染' 添加到处理后的DataFrame中
df_processed['是否患有颅内感染'] = df['是否患有颅内感染']
df_processed.to_excel('df_processed.xlsx', index=False)