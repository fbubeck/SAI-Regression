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
    algo1_a_lr = config["Algorithm 1 - Run A"]["learning_rate"]
    algo1_a_epochs = config["Algorithm 1 - Run A"]["n_epochs"]
    algo1_a_id = config["Algorithm 1 - Run A"]["id"]
    algo1_a_opt = config["Algorithm 1 - Run A"]["opt"]
    algo1_b_lr = config["Algorithm 1 - Run B"]["learning_rate"]
    algo1_b_epochs = config["Algorithm 1 - Run B"]["n_epochs"]
    algo1_b_id = config["Algorithm 1 - Run B"]["id"]
    algo1_b_opt = config["Algorithm 1 - Run B"]["opt"]
    algo1_c_lr = config["Algorithm 1 - Run C"]["learning_rate"]
    algo1_c_epochs = config["Algorithm 1 - Run C"]["n_epochs"]
    algo1_c_id = config["Algorithm 1 - Run C"]["id"]
    algo1_c_opt = config["Algorithm 1 - Run C"]["opt"]
    algo2_a_max_depth = config["Algorithm 2 - Run A"]["max_depth"]
    algo2_a_id = config["Algorithm 2 - Run A"]["id"]
    algo2_b_max_depth = config["Algorithm 2 - Run B"]["max_depth"]
    algo2_b_id = config["Algorithm 2 - Run B"]["id"]
    algo2_c_max_depth = config["Algorithm 2 - Run C"]["max_depth"]
    algo2_c_id = config["Algorithm 2 - Run C"]["id"]
    algo3_a_degree = config["Algorithm 3 - Run A"]["degree"]
    algo3_a_id = config["Algorithm 3 - Run A"]["id"]
    algo3_b_degree = config["Algorithm 3 - Run B"]["degree"]
    algo3_b_id = config["Algorithm 3 - Run B"]["id"]
    algo3_c_degree = config["Algorithm 3 - Run C"]["degree"]
    algo3_c_id = config["Algorithm 3 - Run C"]["id"]
    algo4_a_n_estimators = config["Algorithm 4 - Run A"]["n_estimators"]
    algo4_b_n_estimators = config["Algorithm 4 - Run B"]["n_estimators"]
    algo4_c_n_estimators = config["Algorithm 4 - Run C"]["n_estimators"]

    # Get Sample Data
    sampleData = DataProvider.DataProvider()
    train_data, test_data = sampleData.get_Data()

    # # Data Exploration
    # DataExploration = Exploration.Exploration(train_data, test_data)
    # DataExploration.plot()

    # Creating Algorithm Objects
    algo1_a = Algorithm_1.TensorFlow_ANN(train_data, test_data, algo1_a_lr, algo1_a_epochs, algo1_a_id, algo1_a_opt)
    algo1_b = Algorithm_1.TensorFlow_ANN(train_data, test_data, algo1_b_lr, algo1_b_epochs, algo1_b_id, algo1_b_opt)
    algo1_c = Algorithm_1.TensorFlow_ANN(train_data, test_data, algo1_c_lr, algo1_c_epochs, algo1_c_id, algo1_c_opt)

    algo2_a = Algorithm_2.DecisionTree(train_data, test_data,  algo2_a_max_depth, algo2_a_id)
    algo2_b = Algorithm_2.DecisionTree(train_data, test_data, algo2_b_max_depth, algo2_b_id)
    algo2_c = Algorithm_2.DecisionTree(train_data, test_data, algo2_c_max_depth, algo2_c_id)

    algo3_a = Algorithm_3.Linear_Regression(train_data, test_data, algo3_a_degree, algo3_a_id)

    algo4_a = Algorithm_4.RandomForest(train_data, test_data, algo4_a_n_estimators)
    algo4_b = Algorithm_4.RandomForest(train_data, test_data, algo4_b_n_estimators)
    algo4_c = Algorithm_4.RandomForest(train_data, test_data, algo4_c_n_estimators)

    # ANN
    algo1_a_trainingDuration, algo1_a_trainingError = algo1_a.train()
    algo1_a_testDuration, algo1_a_testError = algo1_a.test()
    algo1_a.plot()
    algo1_b_trainingDuration, algo1_b_trainingError = algo1_b.train()
    algo1_b_testDuration, algo1_b_testError = algo1_b.test()
    algo1_b.plot()
    algo1_c_trainingDuration, algo1_c_trainingError = algo1_c.train()
    algo1_c_testDuration, algo1_c_testError = algo1_c.test()
    algo1_c.plot()

    # Decision Tree
    algo2_a_trainingDuration, algo2_a_trainingError = algo2_a.train()
    algo2_a_testDuration, algo2_a_testError = algo2_a.test()
    algo2_b_trainingDuration, algo2_b_trainingError = algo2_b.train()
    algo2_b_testDuration, algo2_b_testError = algo2_b.test()
    algo2_c_trainingDuration, algo2_c_trainingError = algo2_c.train()
    algo2_c_testDuration, algo2_c_testError = algo2_c.test()

    # Linear Regression
    algo3_a_trainingDuration, algo3_a_trainingError = algo3_a.train()
    algo3_a_testDuration, algo3_a_testError = algo3_a.test()

    # Random Forest
    algo4_a_trainingDuration, algo4_a_trainingError = algo4_a.train()
    algo4_a_testDuration, algo4_a_testError = algo4_a.test()
    algo4_b_trainingDuration, algo4_b_trainingError = algo4_b.train()
    algo4_b_testDuration, algo4_b_testError = algo4_b.test()
    algo4_c_trainingDuration, algo4_c_trainingError = algo4_c.train()
    algo4_c_testDuration, algo4_c_testError = algo4_c.test()

    # Plots
    px = 1 / plt.rcParams['figure.dpi']

    training1 = {'error': [algo1_a_testError, algo1_b_testError, algo1_c_testError],
                 'duration': [algo1_a_trainingDuration, algo1_b_trainingDuration, algo1_c_trainingDuration],
                 'Run': ["A", "B", "C"]
                 }

    training2 = {'error': [algo2_a_testError, algo2_b_testError, algo2_c_testError],
                 'duration': [algo2_a_trainingDuration, algo2_b_trainingDuration, algo2_c_trainingDuration],
                 'Run': ["A", "B", "C"]
                 }

    training3 = {'error': [algo3_a_testError],
                 'duration': [algo3_a_trainingDuration],
                 'Run': ["A"]
                 }

    training4 = {'error': [algo4_a_testError, algo4_b_testError, algo4_c_testError],
                 'duration': [algo4_a_trainingDuration, algo4_b_trainingDuration, algo4_c_trainingDuration],
                 'Run': ["A", "B", "C"]
                 }

    inference1 = {'error': [algo1_a_testError, algo1_b_testError, algo1_c_testError],
                  'duration': [algo1_a_testDuration, algo1_b_testDuration, algo1_c_testDuration],
                  'Run': ["A", "B", "C"]
                  }

    inference2 = {'error': [algo2_a_testError, algo2_b_testError, algo2_c_testError],
                  'duration': [algo2_a_testDuration, algo2_b_testDuration, algo2_c_testDuration],
                  'Run': ["A", "B", "C"]
                  }

    inference3 = {'error': [algo3_a_testError],
                  'duration': [algo3_a_testDuration],
                  'Run': ["A"]
                  }

    inference4 = {'error': [algo4_a_testError, algo4_b_testError, algo4_c_testError],
                  'duration': [algo4_a_testDuration, algo4_b_testDuration, algo4_c_testDuration],
                  'Run': ["A", "B", "C"]
                  }

    data_training1 = pd.DataFrame(training1)
    data_training2 = pd.DataFrame(training2)
    data_training3 = pd.DataFrame(training3)
    data_training4 = pd.DataFrame(training4)
    data_inference1 = pd.DataFrame(inference1)
    data_inference2 = pd.DataFrame(inference2)
    data_inference3 = pd.DataFrame(inference3)
    data_inference4 = pd.DataFrame(inference4)
    data_training1.sort_values(by=['duration'], inplace=True)
    data_training2.sort_values(by=['duration'], inplace=True)
    data_training3.sort_values(by=['duration'], inplace=True)
    data_training4.sort_values(by=['duration'], inplace=True)
    data_inference1.sort_values(by=['duration'], inplace=True)
    data_inference2.sort_values(by=['duration'], inplace=True)
    data_inference3.sort_values(by=['duration'], inplace=True)
    data_inference4.sort_values(by=['duration'], inplace=True)

    fig = plt.figure(figsize=(900 * px, 700 * px))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_xlabel('Duration [in seconds]', fontsize=11)
    ax1.set_ylabel('Model-Performance (R2 Score)', fontsize=11)
    ax2.set_xlabel('Duration [in seconds]', fontsize=11)
    ax2.set_ylabel('Model-Performance (R2 Score)', fontsize=11)
    fig.suptitle('Efficiency of different ML-Algorithms and Parametersets', fontsize=14)
    ax1.plot(data_training1["duration"], data_training1["error"], '-o', c='blue', alpha=0.6)
    ax1.plot(data_training2["duration"], data_training2["error"], '-o', c='green', alpha=0.6)
    ax1.plot(data_training3["duration"], data_training3["error"], '-o', c='red', alpha=0.6)
    ax1.plot(data_training4["duration"], data_training4["error"], '-o', c='orange', alpha=0.6)
    ax2.plot(data_inference1["duration"], data_inference1["error"], '-o', c='blue', alpha=0.6)
    ax2.plot(data_inference2["duration"], data_inference2["error"], '-o', c='green', alpha=0.6)
    ax2.plot(data_inference3["duration"], data_inference3["error"], '-o', c='red', alpha=0.6)
    ax2.plot(data_inference4["duration"], data_inference4["error"], '-o', c='orange', alpha=0.6)
    ax1.title.set_text('Training')
    ax2.title.set_text('Inference')
    plt.legend(["TensorFlow ANN ", "Decision Tree", "Linear Regression", "Random Forest"], fontsize=11,
               loc='lower center',
               ncol=4, bbox_transform=fig.transFigure,
               bbox_to_anchor=(0.5, 0))
    # ax1.set_yscale('log')
    # ax2.set_yscale('log')
    for i in range(3):
        ax1.annotate(data_training1["Run"][i], xy=(data_training1["duration"][i], data_training1["error"][i]),
                     color='black',
                     fontsize=10, weight='heavy',
                     horizontalalignment='left',
                     verticalalignment='center')
        ax1.annotate(data_training2["Run"][i], xy=(data_training2["duration"][i], data_training2["error"][i]),
                     color='black',
                     fontsize=10, weight='heavy',
                     horizontalalignment='left',
                     verticalalignment='center')
        ax1.annotate(data_training4["Run"][i], xy=(data_training4["duration"][i], data_training4["error"][i]),
                     color='black',
                     fontsize=10, weight='heavy',
                     horizontalalignment='left',
                     verticalalignment='center')
        ax2.annotate(data_inference1["Run"][i], xy=(data_inference1["duration"][i], data_inference1["error"][i]),
                     color='black',
                     fontsize=10, weight='heavy',
                     horizontalalignment='left',
                     verticalalignment='center')
        ax2.annotate(data_inference2["Run"][i], xy=(data_inference2["duration"][i], data_inference2["error"][i]),
                     color='black',
                     fontsize=10, weight='heavy',
                     horizontalalignment='left',
                     verticalalignment='center')
        ax2.annotate(data_inference4["Run"][i], xy=(data_inference4["duration"][i], data_inference4["error"][i]),
                     color='black',
                     fontsize=10, weight='heavy',
                     horizontalalignment='left',
                     verticalalignment='center')
    ax1.annotate(data_training3["Run"][0], xy=(data_training3["duration"][0], data_training3["error"][0]),
                 color='black',
                 fontsize=10, weight='heavy',
                 horizontalalignment='left',
                 verticalalignment='center')
    ax2.annotate(data_inference3["Run"][0], xy=(data_inference3["duration"][0], data_inference3["error"][0]),
                 color='black',
                 fontsize=10, weight='heavy',
                 horizontalalignment='left',
                 verticalalignment='center')
    plt.savefig('plots/Algorithms_Evaluation.png', dpi=300)
    # plt.show()
    print("Evaluation Plot saved...")
    print("")


if __name__ == "__main__":
    main()
