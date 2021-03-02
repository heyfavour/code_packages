import numpy as np
import pandas as pd

# data load
# 前20天 18项
train_data = pd.read_csv("train.csv", encoding='big5')
train_data.replace("NR", 0, inplace=True)

# test_data = pd.read_csv('test.csv', encoding='big5', names=['id', '測項', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
# test_data["id"] = test_data["id"].str.split("_", expand=True)[1].astype("int")
# test_data.replace("NR", 0, inplace=True)
# 纵表转横表
# 预数据处理
for num in range(0, 23):
    train_data[str(num)] = train_data[str(num)].astype(float)

train_data.drop('測站', axis=1, inplace=True)

train_data_temp = pd.DataFrame()

for i in range(24):
    train_data_hour = train_data[['日期', '測項', str(i)]].copy()
    train_data_hour['日期'] = pd.to_datetime(train_data_hour['日期'] + ' ' + str(i) + ':00:00')
    train_data_hour = train_data_hour.pivot(index='日期', columns='測項', values=str(i))
    train_data_temp = pd.concat([train_data_temp, train_data_hour])
train_data = train_data_temp.astype('float64').reset_index().set_index("日期").sort_index()
print(train_data)

#feature scaling for train_data
#(X-mean)/std
train_mean = train_data_new.mean().copy()
train_std = train_data_new.std().copy()
train_data_new1 = train_data_new.copy()
for liecolumn in train_data_new:
        train_data_new[liecolumn] = (train_data_new[liecolumn] - train_mean[liecolumn])/train_std[liecolumn]

