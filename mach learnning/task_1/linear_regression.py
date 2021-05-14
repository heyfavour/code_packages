import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt


def loan_data():
    # data load
    # 前20天 18项
    train_data = pd.read_csv("train.csv", encoding='big5')
    train_data.replace("NR", 0, inplace=True)
    # 纵表转横表
    # 预数据处理
    for num in range(0, 23):
        train_data[str(num)] = train_data[str(num)].astype(float)

    train_data.drop('測站', axis=1, inplace=True)

    train_data_by_time = pd.DataFrame()
    for i in range(24):
        train_data_hour = train_data[['日期', '測項', str(i)]].copy()
        train_data_hour['日期'] = pd.to_datetime(train_data_hour['日期'] + ' ' + str(i) + ':00:00')
        train_data_hour = train_data_hour.pivot(index='日期', columns='測項', values=str(i))
        train_data_by_time = pd.concat([train_data_by_time, train_data_hour])
    train_data = train_data_by_time.astype('float64').reset_index().set_index("日期").sort_index()

    # Y
    train_y = train_data.loc[:, ['PM2.5']]
    train_y.index = train_y.index - datetime.timedelta(hours=9)
    train_y.columns = ["Y"]

    # feature scaling for train_data
    # X-Xmin/(Xmax-Xmin)
    train_max = train_data.max()
    train_min = train_data.min()
    for col in train_data:
        train_data[col] = (train_data[col] - train_min[col]) / (train_max[col] - train_min[col])
    # x merge serval time rows => one row
    train_x = train_data.copy()
    train_x.columns = train_x.columns + "_0"
    for i in range(1, 9):
        train_data_merge = train_data.copy()
        train_data_merge.index = train_data_merge.index - datetime.timedelta(hours=i)
        train_x = pd.merge(train_x, train_data_merge, on="日期")

    # --------------test data
    test_x = load_test_data(train_max, train_min)

    return train_x, train_y, test_x


def data_sample(data_x, data_y):
    train_x = data_x.sample(frac=0.8)
    train_x = train_x[train_x.index.isin(data_y.index)]
    train_y = pd.merge(train_x, data_y, on="日期")[["Y"]]

    validation_x = data_x[~data_x.index.isin(train_x.index)]
    validation_x = validation_x[validation_x.index.isin(data_y.index)]
    validation_y = pd.merge(validation_x, data_y, on="日期")[["Y"]]

    # 去掉Y中没有的
    return train_x, train_y, validation_x, validation_y


# set y = x1*w1 + x2*w2 + x3*w3
def get_gradient(x, y, weight, regularzation_param, XTX, XTY):
    # 矩阵法求梯度下降 X.T@X@W-X.T@T
    # return ((train_x.T @ train_x) @ weight) - (train_x.T @ train_y)
    return (XTX @ weight) - XTY + (regularzation_param * weight)


def cal_loss(x, y, weight):
    num = x.shape[0]
    return np.sum(np.sqrt(np.power(x @ weight - y, 2)) / num)


def plt_show(train_loss_list, validation_loss_list):
    fig, ax = plt.subplots()
    ax.set_xlabel('step')
    ax.set_ylabel('loss')
    ax.plot([i[0] for i in train_loss_list], [i[1] for i in train_loss_list])
    ax.plot([i[0] for i in validation_loss_list], [i[1] for i in validation_loss_list])
    plt.show()


def load_test_data(train_max, train_min):
    test_data = pd.read_csv('test.csv', encoding='big5',
                            names=['id', '測項', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
    test_data["id"] = test_data["id"].str.split("_", expand=True)[1].astype("int")
    test_data.replace("NR", 0, inplace=True)
    # 纵表转横表
    # 预数据处理
    for num in range(0, 8):
        test_data[str(num)] = test_data[str(num)].astype(float)

    test_data_by_id = pd.DataFrame()
    for i in range(9):
        test_data_slice = test_data[['id', '測項', str(i)]].copy()
        test_data_slice = test_data_slice.pivot(index='id', columns='測項', values=str(i))
        test_data_slice.columns = test_data_slice.columns + "_" + str(i)
        test_data_by_id = pd.concat([test_data_by_id, test_data_slice], axis=1)
    test_data = test_data_by_id.astype('float64').reset_index().set_index("id").sort_index()

    for col in test_data:
        tran_name = col[:-2]
        test_data[col] = (test_data[col] - train_min[tran_name]) / (train_max[tran_name] - train_min[tran_name])
    return test_data


def predict_test(test_x, weight):
    test_x = np.hstack((test_x.values, np.ones((np.size(test_x.values, 0), 1), 'double')))
    test_y = test_x @ weight
    return test_y


if __name__ == '__main__':
    data_x, data_y, test_x = loan_data()
    train_x, train_y, validation_x, validation_y = data_sample(data_x, data_y)
    # pd_to_np
    train_x = np.hstack((train_x.values, np.ones((np.size(train_x.values, 0), 1), 'double')))
    train_y = train_y.values

    validation_x = np.hstack((validation_x.values, np.ones((np.size(validation_x.values, 0), 1), 'double')))
    validation_y = validation_y.values
    # weight
    weight = np.random.random((np.size(train_x, 1), 1))
    # learning_rate
    learning_rate = 100
    # regularzation
    regularzation_param = 1

    XTX = train_x.T @ train_x
    XTY = train_x.T @ train_y
    adagrad = np.zeros([train_x.shape[1], 1])

    train_loss_list = []
    validation_loss_list = []
    for step in range(100000):
        gradient = get_gradient(train_x, train_y, weight, regularzation_param, XTX, XTY)
        adagrad = adagrad + np.power(gradient, 2)
        weight = weight - (learning_rate / np.sqrt(adagrad + 1e-10)) * gradient
        if step % 100 == 0:  # loss record
            train_loss = cal_loss(train_x, train_y, weight)
            validation_loss = cal_loss(validation_x, validation_y, weight)
            train_loss_list.append([step, train_loss])
            validation_loss_list.append([step, validation_loss])
    # np.save('weight.npy', weight)
    # plt_show(train_loss_list,validation_loss_list)
    test_y = predict_test(test_x, weight)
    print(test_y)
