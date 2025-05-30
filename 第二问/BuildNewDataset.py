import pandas as pd

# 加载数据文件
df_processed = pd.read_excel("df_processed.xlsx")
df_processed2 = pd.read_excel("df_processed2.xlsx")

df_processed.drop(columns='是否患有颅内感染',inplace=True)
# 假设两个 DataFrame 中都有一个共同的列用于连接，例如 '住院号'
# 请确保 '住院号' 在两个 DataFrame 中的列名完全一致，包括大小写和空格
common_key_column = "住院号"  # 请根据你的实际列名修改

# 执行左连接 (left merge)
# df_processed2 是左表，保留其所有行和“住院号”的结构
# df_processed 是右表，将其信息匹配到左表
# on=common_key_column 指定使用哪个列作为连接键
# how='left' 表示以左表 (df_processed2) 为基准进行连接
merged_df = pd.merge(df_processed2, df_processed, on=common_key_column, how="left",indicator=True)

# 验证和解释：
# 1. df_processed2 中的所有行都会被保留。
# 2. 对于 df_processed2 中的每一行，会根据其 '住院号' 的值在 df_processed 中查找匹配的行。
# 3. 如果在 df_processed 中找到匹配的 '住院号'：
#    - df_processed 中该 '住院号' 对应行的所有其他列的值，会被复制到 merged_df 中当前行的相应新列中。
#    - 由于 df_processed 中每个 '住院号' 只有一行，而 df_processed2 中一个 '住院号' 可能有多行，
#      所以 df_processed 中的这唯一一行数据会被“广播”或“复制”到 merged_df 中所有具有相同 '住院号' 的行上。
# 4. 如果在 df_processed 中没有找到匹配的 '住院号'（理论上不应发生，如果 df_processed 包含了所有 df_processed2 中的住院号的话），
#    那么从 df_processed 合并过来的列在 merged_df 的对应行中将填充为 NaN。
# 获取要转换的列
cols_to_convert = merged_df.columns[9:15]
for col in cols_to_convert:
    merged_df[col] = merged_df[col].astype(int)
merged_df.to_excel("FinalDataset.xlsx")

# 检查合并后是否有因为右表（df_processed）列名与左表（df_processed2）除连接键外的其他列名重复而产生的 '_x', '_y' 后缀
# 如果有，你可能需要处理这些重复列名，例如通过重命名或选择保留哪一个。
# pd.merge 会自动处理同名列（非连接键）的情况，通常会给它们加上后缀 _x (来自左表) 和 _y (来自右表)。
# 如果 df_processed 和 df_processed2 除了 '住院号' 之外没有其他同名列，则不需要担心这个。
# 如果有，你可以在 merge 之前就重命名 df_processed 中的列（除了连接键），以避免后缀。
# 例如：
# df_processed_renamed = df_processed.rename(columns={'某列': '某列_来自df1'})
# merged_df = pd.merge(df_processed2, df_processed_renamed, on=common_key_column, how='left')

# (可选) 保存合并后的 DataFrame
# merged_df.to_csv("merged_data.csv", index=False)
# merged_df.to_excel("merged_data.xlsx", index=False)
# print("合并后的数据已保存到 merged_data.csv/xlsx (如果取消注释)")