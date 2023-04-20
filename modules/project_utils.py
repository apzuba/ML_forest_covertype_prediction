from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.utils import to_categorical
import numpy as np


def print_compare_predicitons(prediction, y_train, start_row=0):
    for i, v in enumerate(prediction):
        print(f"The model: {v}, Actual: {y_train[start_row+i][0]}, {v == y_train[start_row+i][0]}")


def distil_best_params(best_params):
    dropout = best_params['dropout']
    num_layers = best_params['num_layers']
    num_neurons = best_params['num_neurons'] 
    learning_rate = best_params['learning_rate']

    return dropout, num_layers, num_neurons, learning_rate

def adjust_data_for_tf(X_train, y_train, X_cv, y_cv, X_test, y_test, X_mini_test):

    # Decreasing the training dataset to speed up the time of model training.
    X_train, x_, y_train, y_ = train_test_split(X_train, y_train, test_size=0.98, random_state=1)
    X_cv, x_, y_cv, y_ = train_test_split(X_cv, y_cv, test_size=0.98, random_state=1)
    X_test, x_, y_test, y_ = train_test_split(X_test, y_test, test_size=0.98, random_state=1)

    del x_, y_      #deleting the temporary variable

    # Expanding y_train to 7 categories.
    y_train_conv = to_categorical(y_train - 1, num_classes=7)
    y_cv_conv = to_categorical(y_cv - 1, num_classes=7)
    y_test_conv = to_categorical(y_test - 1, num_classes=7)

    # Scaling the features.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_cv_scaled = scaler.transform(X_cv)
    X_test_scaled = scaler.transform(X_test)
    X_mini_test_scaled = scaler.transform(X_mini_test)

    # Return the scaled and decreased in size dataset.
    return X_train_scaled, y_train_conv, X_cv_scaled, y_cv_conv, X_test_scaled, y_test_conv, X_mini_test_scaled

def scale_features(X_train, X):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return scaler.transform(X)