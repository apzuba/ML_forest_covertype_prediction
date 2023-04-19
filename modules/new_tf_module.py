import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import LambdaCallback

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

import numpy as np
import matplotlib.pyplot as plt


# Create function that returns a compiled Keras model
def create_model(num_layers=1, num_neurons=24, dropout=0.3, learning_rate=0.01):
    model = Sequential()
    model.add(Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.3)))
    for i in range(num_layers-1):
        model.add(Dense(num_neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.3)))
        model.add(Dropout(dropout))
    model.add(Dense(7, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.3)))
    model.add(Dense(1, activation= 'linear'))
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), 
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        metrics=['accuracy'
                    ])
    return model

def search_fit(X_train_scaled, y_train, X_cv_scaled, y_cv):

    # Create KerasClassifier object.
    model = KerasClassifier(build_fn=create_model, verbose=0)

    # Set hyperparameters and value ranges to randomly test from.
    param_dist = {
        'num_layers': randint(1, 4),
        'num_neurons': randint(32, 156),
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate' : [0.0001, 0.001, 0.01, 0.1],
    }

    # Create RandomizedSearchCV object
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=5, cv=2, verbose=2)

    # train the network and store the history data
    search.fit(X=X_train_scaled, y=y_train,
        validation_data=(X_cv_scaled, y_cv),
        batch_size=8,
        epochs=20,
        )

    # Print the best score & return the best parameters.
    print(search.best_score_)

    return search.best_params_

    
def plot_the_best(model, X_train_scaled, y_train, X_cv_scaled, y_cv):

    # train the best model on the full training set
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_cv_scaled, y_cv),
        epochs=20, 
        batch_size=4
        )
    
    # DataFrame(history.history).plot(figsize=(8,5))
    # plt.show()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
