import logging
import os
import traceback
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

DEBUG = False
FILE_NAME = "data.pkl"
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def get_data():
    path, dirs, files = next(os.walk("./csv/"))
    file_count = len(files)
    data_list = []
    for i in range(file_count):
        try:
            curr = pd.read_csv("./csv/"+files[i], header=None)
            data_list.append(curr)
        except Exception as err:
            print(err)
            logging.debug(traceback.format_exc())
        
    return data_list

def combine_data(data_list):
    return pd.concat(data_list)

def save_data(data):
    data.to_pickle(FILE_NAME)

def load_data():
    data = None
    try:
        data = pd.read_pickle(FILE_NAME)
    except Exception as err:
        print(err)
        logging.debug(traceback.format_exc())
    
    return data

def setup(data):
    logging.debug(f"type={type(data)}, shape={data.shape}")
    target = data[1]
    
    # Replace NaN with zero
    data = data.fillna(0)
    
    # Remove first two columns
    cols = [0,1]
    data.drop(data.columns[cols], axis=1, inplace=True)
    
    X = data
    # Remove column names
    X = X[1:]
    # Convert string to float
    X = X.replace('%','', regex=True).astype(np.float64)

    y = target
    # Remove column names
    y = y[1:]

    return X, y

def best_result(results, column):
    best_result = float('-inf')
    for result in results:
        if result[column] > best_result:
            best_result = result[column]
    return best_result

# K Nearest Neighbor algorithm
def run_knn(kMin, kMax, X, y):
    results = []
    for k in range(kMin, kMax+1):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)*100
        results.append([k,accuracy])
        logging.debug(f"k={k} Accuracy={accuracy:2f}")
    return results

# Plot accuracy vs k values
def plot_knn(results):
    x = [result[0] for result in results]
    y = [result[1] for result in results]
    plt.plot(x, y, 'ro')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.show()

# Run KNN on data 
def knn():
    category = ["Batting","Fielding","Pitching"]
    data_list = get_data()
    for i, data in enumerate(data_list):
        save_data(data_list[i])
        df = load_data()
        X, y = setup(df)

        results = run_knn(1,400,X,y)

        print(f"{category[i]} data:")

        best = best_result(results, 1)
        print(f"Highest accuracy = {best:2f}%")
    
        plot_knn(results)

# Multi-layer Perceptron, a supervised ML algorithm utilizing backpropagation
# Using 'sgd' solver which refers to stochastic gradient descent
def run_mlp(X, y, hidden_layers=(100,), learning_rate=0.001, epochs=200):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = MLPClassifier(solver='sgd', hidden_layer_sizes=hidden_layers, learning_rate_init=learning_rate, max_iter=epochs, alpha=1e-5, random_state=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)*100
    logging.debug(f"Accuracy={accuracy:2f}, hidden_layers={hidden_layers}, learning_rate={learning_rate}, epochs={epochs}")
    return accuracy

# Run MLP on data
def mlp():
    category = ["Batting","Fielding","Pitching"]
    data_list = get_data()

    hidden_layers_list = [(10),(10, 50),(10, 100),(10, 200),(50),(50, 100),(50, 200),(100),(100, 200),(200),(10, 50, 100),(10, 50, 200),(10, 100, 200),(50, 100, 200),(10, 50, 100, 200)]
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.2]
    epochs_list = [10, 100, 1000, 2000]

    for i, data in enumerate(data_list):
        save_data(data_list[i])
        df = load_data()
        X, y = setup(df)

        print(f"{category[i]} data:")
        results = []
        for hl in hidden_layers_list:
            for lr in learning_rates:
                for e in epochs_list:
                    accuracy = run_mlp(X, y, hidden_layers=hl, learning_rate=lr, epochs=e)
                    results.append([accuracy, hl,lr, e])

        best = best_result(results, 0)
        print(f"Highest accuracy = {best:2f}%")

def main():
    # knn()
    mlp()



if __name__ == "__main__":
    if(DEBUG):
        logging.basicConfig(level=logging.DEBUG)    
        logging.debug("Debugging Enabled.\n")

    main()
    logging.debug("Exiting Program.")
