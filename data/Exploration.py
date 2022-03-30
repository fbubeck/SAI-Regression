from matplotlib import pyplot as plt
import numpy as np


class Exploration:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def plot(self):
        xs_train, ys_train = self.train_data


        px = 1/plt.rcParams['figure.dpi']  
        __fig = plt.figure(figsize=(800*px, 600*px))
        plt.scatter(xs_train, ys_train, color='b', s=1, alpha=0.5)
        plt.title('Training Data (n=' + str(len(xs_train)) + ')')
        plt.ylabel('y (Output)')
        plt.xlabel('x (Input)')
        plt.savefig('plots/DataExploration.png')
        #plt.show()
        print("Exploration Plot saved...")
        print("")
