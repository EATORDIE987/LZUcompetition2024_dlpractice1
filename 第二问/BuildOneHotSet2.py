import pandas as pd

df = pd.read_excel("dataset_fillna.xlsx")

# 指定需要进行独热编码的列名
columns_to_encode = ["透明度", "蛋白定性"]

# 执行独热编码
# pd.get_dummies() 会将指定的分类列转换为多个新的0/1数值列
df_encoded = pd.get_dummies(
    df,
    # ? 告诉函数哪些列需要被编码
    columns=columns_to_encode,  
    # ? 为新生成的独热编码列指定前缀。
    prefix=columns_to_encode,  
    # ? 例如，如果“透明度”列有值“清晰”，
    # ? 那么新列名会是“透明度_清晰”
    # ? dummy_na=False 表示不为 NaN (缺失值) 创建单独的指示列。
    # ? 如果此值为 True，则会为每个编码列中的 NaN 值,创建一个额外的列，如 "透明度_nan"。
    dummy_na=False,
) 
df_encoded['住院号'] = df_encoded['住院号'].ffill()
df_encoded.to_excel('df_processed2.xlsx')
