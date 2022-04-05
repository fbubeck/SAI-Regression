import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.models import Sequential
from matplotlib import pyplot as plt
from time import time
from sklearn.metrics import mean_squared_error, r2_score


class TensorFlow_ANN:
    def __init__(self, train_data, test_data, learning_rate, n_epochs, opt, i):
        self.history = None
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.i = i
        self.opt = opt
        self.model = 0

    def train(self):
        # Training Data
        xs_train, ys_train = self.train_data

        n_inputs = xs_train.shape[1]
        n_outputs = ys_train.shape[0]

        # define model architecture
        self.model = Sequential()
        self.model.add(Dense(self.i, input_dim=n_inputs, activation='relu'))
        self.model.add(Dense(1))

        # Define Optimizer
        if self.opt == "SGD":
            opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        elif self.opt == "RMSprop":
            opt = tf.keras.optimizers.RMSprop(self.learning_rate)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # define loss and optimizer
        self.model.compile(optimizer=opt, loss='mean_squared_error')

        # Modeling
        start_training = time()
        self.history = self.model.fit(xs_train, ys_train, epochs=self.n_epochs, validation_split=0.33,
                                      batch_size=128, verbose=1)
        end_training = time()

        # Time
        duration_training = end_training - start_training
        duration_training = round(duration_training, 4)

        # Number of Parameter
        trainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        n_params = trainableParams + nonTrainableParams

        # Prediction for Training mse
        y_pred = self.model.predict(xs_train)
        error = r2_score(ys_train, y_pred)
        error *= 100
        error = round(error, 4)

        # Summary
        print('------ TensorFlow - ANN ------')
        print('Number of Neurons: ', self.i)
        print(f'Duration Training: {duration_training} seconds')
        print('R2 Score Training: ', error)
        print("Number of Parameter: ", n_params)

        return duration_training, error

    def test(self):
        # Test Data
        xs_test, ys_test = self.test_data

        # Predict Data
        start_test = time()
        y_pred = self.model.predict(xs_test)
        error = r2_score(ys_test, y_pred)
        error *= 100
        error = round(error, 4)
        end_test = time()

        # Time
        duration_test = end_test - start_test
        duration_test = round(duration_test, 4)

        print(f'Duration Inference: {duration_test} seconds')

        print("R2 Score Testing: %.2f" % error)
        print("")

        return duration_test, error

    def plot(self):
        # Plot loss and val_loss
        px = 1 / plt.rcParams['figure.dpi']
        __fig = plt.figure(figsize=(800 * px, 600 * px))
        plt.plot(self.history.history['loss'], 'blue')
        plt.plot(self.history.history['val_loss'], 'red')
        plt.title('Neural Network Training loss history')
        plt.ylabel('loss (log scale)')
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend(['train_loss', 'val_loss'], loc='upper right')
        url = f"plots/training-history/TensorFlow_{self.id}_Loss-Epochs-Plot.png"
        plt.savefig(url)
        # plt.show()
        print("TensorFlow loss Plot saved...")
        print("")
