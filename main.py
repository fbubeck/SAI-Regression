from algorithms import Algorithm_1, Algorithm_2, Algorithm_3, Algorithm_4
from data import DataProvider, Exploration
import json
from matplotlib import pyplot as plt
import pandas as pd


def main():
    print("Starting...")
    print("")

    # read config.json
    with open('config/config.json') as file:
        config = json.load(file)

    # Get Parameters from config file
    algo1_lr = config["Algorithm 1"]["learning_rate"]
    algo1_epochs = config["Algorithm 1"]["n_epochs"]
    algo1_opt = config["Algorithm 1"]["opt"]

    # Get Sample Data
    sampleData = DataProvider.DataProvider()
    train_data, test_data = sampleData.get_Data()

    # # Data Exploration
    # DataExploration = Exploration.Exploration(train_data, test_data)
    # DataExploration.plot()

    ################################################################################################################
    # Artificial Neural Network
    ################################################################################################################
    NeuralNetwork_training = []
    NeuralNetwork_test = []

    for i in range(3, 30, 3):
        model = Algorithm_1.TensorFlow_ANN(train_data, test_data, algo1_lr, algo1_epochs, algo1_opt, i)
        duration_train, acc_train = model.train()
        duration_test, acc_test = model.test()

        NeuralNetwork_training.append(
            {'accuracy': acc_train,
             'duration': duration_train,
             'Run': i
             }
        )
        NeuralNetwork_test.append(
            {'accuracy': acc_test,
             'duration': duration_test,
             'Run': i
             }
        )
        model = None

    NeuralNetwork_training_df = pd.DataFrame(NeuralNetwork_training)
    NeuralNetwork_test_df = pd.DataFrame(NeuralNetwork_test)

    ################################################################################################################
    # DecisionTree
    ################################################################################################################
    DecisionTree_training = []
    DecisionTree_test = []

    for i in range(2, 40, 1):
        model = Algorithm_2.DecisionTree(train_data, test_data, i)
        duration_train, acc_train= model.train()
        duration_test, acc_test = model.test()

        DecisionTree_training.append(
            {'accuracy': acc_train,
             'duration': duration_train,
             'Run': i
             }
        )
        DecisionTree_test.append(
            {'accuracy': acc_test,
             'duration': duration_test,
             'Run': i
             }
        )
        model = None

    DecisionTree_training_df = pd.DataFrame(DecisionTree_training)
    DecisionTree_test_df = pd.DataFrame(DecisionTree_test)

    ################################################################################################################
    # Support Vector Machine
    ################################################################################################################
    SVM_training = []
    SVM_Regression_test = []

    for i in range(400, 1500, 50):
        model = Algorithm_3.Linear_Regression(train_data, test_data, i)
        duration_train, acc_train = model.train()
        duration_test, acc_test = model.test()

        SVM_training.append(
            {'accuracy': acc_train,
             'duration': duration_train,
             'Run': i
             }
        )
        SVM_Regression_test.append(
            {'accuracy': acc_test,
             'duration': duration_test,
             'Run': i
             }
        )
        model = None

    SVM_training_df = pd.DataFrame(SVM_training)
    SVM_test_df = pd.DataFrame(SVM_Regression_test)

    ################################################################################################################
    # Random Forest
    ################################################################################################################
    RandomForest_training = []
    RandomForest_test = []

    for i in range(1, 30, 1):
        model = Algorithm_4.RandomForest(train_data, test_data, i)
        duration_train, acc_train = model.train()
        duration_test, acc_test = model.test()

        RandomForest_training.append(
            {'accuracy': acc_train,
             'duration': duration_train,
             'Run': i
             }
        )
        RandomForest_test.append(
            {'accuracy': acc_test,
             'duration': duration_test,
             'Run': i
             }
        )
        model = None

    RandomForest_training_df = pd.DataFrame(RandomForest_training)
    RandomForest_test_df = pd.DataFrame(RandomForest_test)

    ################################################################################################################
    # Evaluation
    ################################################################################################################
    px = 1 / plt.rcParams['figure.dpi']

    fig = plt.figure(figsize=(1000 * px, 700 * px))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_xlabel('Model-Performance (R2-Score in %)', fontsize=10)
    ax1.set_ylabel('Duration [in seconds]', fontsize=10)
    ax2.set_xlabel('Model-Performance (R2-Score in %)', fontsize=10)
    ax2.set_ylabel('Duration [in seconds]', fontsize=10)
    # ax1.set_xlim([30, 100])
    # ax2.set_xlim([30, 100])
    ax1.plot(NeuralNetwork_test_df["accuracy"], NeuralNetwork_training_df["duration"], '-o', c='blue', alpha=0.6, markersize=4)
    ax2.plot(NeuralNetwork_test_df["accuracy"], NeuralNetwork_test_df["duration"], '-o', c='blue', alpha=0.6, markersize=4)
    ax1.plot(DecisionTree_test_df["accuracy"], DecisionTree_training_df["duration"], '-o', c='green', alpha=0.6,
             markersize=4)
    ax2.plot(DecisionTree_test_df["accuracy"], DecisionTree_test_df["duration"], '-o', c='green', alpha=0.6,
             markersize=4)
    ax1.plot(SVM_test_df["accuracy"], SVM_training_df["duration"], '-o', c='red', alpha=0.6,
             markersize=4)
    ax2.plot(SVM_test_df["accuracy"], SVM_test_df["duration"], '-o', c='red', alpha=0.6, markersize=4)
    ax1.plot(RandomForest_test_df["accuracy"], RandomForest_training_df["duration"], '-o', c='orange', alpha=0.6,
             markersize=4)
    ax2.plot(RandomForest_test_df["accuracy"], RandomForest_test_df["duration"], '-o', c='orange', alpha=0.6, markersize=4)
    ax1.title.set_text('Training')
    ax2.title.set_text('Inference')
    plt.legend(["Artificial Neural Network", "Decision Tree", "Support Vector Regressor", "Random Forest"],
               loc='lower center', ncol=4, bbox_transform=fig.transFigure, bbox_to_anchor=(0.5, 0))
    plt.savefig('plots/Algorithms_Evaluation.png', dpi=600)
    plt.clf()
    print("Evaluation Plot saved...")
    print("")


if __name__ == "__main__":
    main()
