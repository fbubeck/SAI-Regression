from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from time import time

from sklearn.metrics import mean_squared_error, r2_score


class RandomForest:
    def __init__(self, train_data, test_data, n_estimators):
        self.history = None
        self.train_data = train_data
        self.test_data = test_data
        self.n_estimators = n_estimators
        self.model = None

    def train(self):
        # Training Data
        xs_train, ys_train = self.train_data

        # Modeling
        self.model = RandomForestRegressor(n_estimators=self.n_estimators)
        start_training = time()
        self.model.fit(xs_train, ys_train)
        end_training = time()

        # Time
        duration_training = end_training - start_training
        duration_training = round(duration_training, 4)

        # Prediction for Training mse
        y_pred = self.model.predict(xs_train)
        error = r2_score(ys_train, y_pred)
        error *= 100
        error = round(error, 4)

        # Summary
        print('------ Random Forest ------')
        print('Number of Trees: ', self.n_estimators)
        print(f'Duration Training: {duration_training} seconds')
        print('R2 Score Training: ', error)

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
