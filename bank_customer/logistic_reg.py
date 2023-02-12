import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 数据加载
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submission = pd.read_csv('data/submission.csv')

# 训练集、测试集合并
df = pd.concat([train, test], axis=0)
cat_columns = df.select_dtypes(include='O').columns

job_le = LabelEncoder()  # 将离散型的数据转换成 0 到 n − 1 之间的数
print(df['job'])
df['job'] = job_le.fit_transform(df['job'])

print(df['job'].value_counts())


