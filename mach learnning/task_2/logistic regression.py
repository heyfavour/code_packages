import numpy as np
import pandas as pd


def load_data():
    # train_data = pd.read_csv("train.csv",index_col=["id"])
    train_x = pd.read_csv("X_train")
    train_y = pd.read_csv("Y_train")
    test_x = pd.read_csv("X_test")
    return train_x, train_y, test_x


def data_sample(train_x, train_y):
    data = pd.merge(train_x, train_y, on="id")
    data = data.set_index("id")
    # data_max = data.max()
    # data_min = data.min()
    # for col in data:
    #     data[col] = (data[col] - data_min[col]) / (data_max[col] - data_min[col])
    train_x = data.sample(frac=0.8).sort_index().copy()
    validation_x = data[~data.index.isin(train_x.index)].copy()

    train_y = train_x[["label"]]
    train_x.drop('label', axis=1, inplace=True)

    validation_y = validation_x[["label"]]
    validation_x.drop('label', axis=1, inplace=True)

    train_x = np.hstack((train_x.values, np.ones((np.size(train_x.values, 0), 1), 'double')))
    train_y = train_y.values

    validation_x = np.hstack((validation_x.values, np.ones((np.size(validation_x.values, 0), 1), 'double')))
    validation_y = validation_y.values

    return train_x, train_y, validation_x, validation_y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def probability(x, weight):
    return sigmoid(x @ weight)


def cost_function(weight, x, y):
    # bernoulli distribution
    m = x.shape[0]
    f_y = probability(x, weight)
    total_cost = -(1 / m) * np.sum(y * np.log(f_y) + (1 - y) * np.log(1 - f_y))
    return total_cost


def gradient_descent(weight, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * (x.T @ (sigmoid(x @ weight) - y))


def cal_loss(x, y, weight):
    num = x.shape[0]
    f_y = probability(x, weight)
    return np.sum(np.sqrt(np.power(f_y - y, 2)) / num)


def cal_error_validation_percent(x, y, weight):
    f_y = probability(x, weight)
    f_y = np.array([[1] if i[0] >= 0.5 else [0] for i in f_y])
    diff = f_y - y
    diff = [i[0] for i in diff]
    right = diff.count(0)
    percent = right / len(diff)
    return percent


if __name__ == '__main__':
    train_x, train_y, test_x = load_data()
    train_x, train_y, validation_x, validation_y = data_sample(train_x, train_y)

    weight = np.random.random((np.size(train_x, 1), 1))

    learning_rate = 0.1
    adagrad = np.zeros([train_x.shape[1], 1])
    for step in range(100000):
        gradient = gradient_descent(weight, train_x, train_y)
        adagrad = adagrad + np.power(gradient, 2)
        weight = weight - (learning_rate / np.sqrt(adagrad + 1e-10)) * gradient
        if step % 100 == 0:
            train_loss = cal_loss(train_x, train_y, weight)
            validation_loss = cal_loss(validation_x, validation_y, weight)
            percent = cal_error_validation_percent(validation_x, validation_y, weight)
            print(step, train_loss, validation_loss, percent)
