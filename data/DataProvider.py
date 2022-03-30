import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


class DataProvider():

    def get_Data(self):
        print("Preprocess data ...")
        x, y = make_regression(n_samples=100000, n_features=5, n_targets=1,
                               effective_rank=None, tail_strength=0.5, noise=8.0, shuffle=True, coef=False,
                               random_state=5)

        xs_train, xs_test, ys_train, ys_test = train_test_split(x, y, test_size=0.33)

        train_data = xs_train, ys_train
        test_data = xs_test, ys_test

        return train_data, test_data
