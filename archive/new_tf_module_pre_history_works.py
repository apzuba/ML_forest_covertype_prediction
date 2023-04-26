import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint


# Create function that returns a compiled Keras model
def create_model(num_layers=1, num_neurons=24, dropout=0.3, optimizer='adam'):
    model = Sequential()
    model.add(Dense(num_neurons, activation='relu'))
    for i in range(num_layers-1):
        model.add(Dense(num_neurons, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), 
        optimizer=optimizer, 
        metrics=['accuracy'
                    ])
    return model

def search_predict(X_train, y_train):

    # Decrease the training dataset to speed up the time of model training.
    X_train, x_, y_train, y_ = train_test_split(X_train, y_train, test_size=0.995, random_state=1)
    del x_, y_

    # Create KerasClassifier object
    model = KerasClassifier(build_fn=create_model, verbose=0)

    # Set hyperparameters
    param_dist = {
        'num_layers': randint(1, 5),
        'num_neurons': randint(32, 156),
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'optimizer': ['adam', 'rmsprop']
    }

    # Create RandomizedSearchCV object
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, verbose=2)

    # Fit the model
    search.fit(X_train, y_train)

    # Print the best parameters
    print(search.best_params_)
    print(search.best_score_)
