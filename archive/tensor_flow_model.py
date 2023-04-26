
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from scipy.stats import loguniform, randint

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error

import random

# suppress warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

class TF_Model:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        # self.model = self.build_model()


    def build_model(self, units=17, no_extra_layers=2, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)):

        # Decreasing the X_train size to speed up the time of model training.
        self.X_train, x_, self.y_train, y_ = train_test_split(self.X_train, self.y_train, test_size=0.995, random_state=1)
        del x_, y_

        # Scale the features using the z-score
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)

        # Setup the model
        self.model = Sequential()

        for i in range(no_extra_layers):
                self.model.add(Dense(units=units, activation = 'relu'))
        self.model.add(Dense(units=7, activation= 'relu')),
        self.model.add(Dense(units=1, activation = 'linear')),


        # Setup the loss and optimizer
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=[tf.keras.metrics.Accuracy()],
        )

        return self.model

    def find_best_params(self):
        
        # Setting the parameters to be varied in the fitting.
        param_distributions = {
            'units': randint(12, 32),
            'no_extra_layers': randint(0,4),
            'optimizer': [tf.keras.optimizers.Adam, 
                          tf.keras.optimizers.experimental.SGD],
        }

        # (learning_rate=random.uniform(0.001, 0.5))

        # Initiating the estimator class and setting the scoring metric.
        model = KerasClassifier(build_fn=self.build_model, verbose=0)
        scoring = make_scorer(accuracy_score)

        # Initiating the Random Search class & fitting the data.
        rand_search = RandomizedSearchCV(
             estimator=model,
             scoring=scoring,
             param_distributions=param_distributions, 
             n_iter=10, 
             cv=3, 
             verbose=2, 
             random_state=42
             )
        
        rand_search.fit(self.X_train, self.y_train)

        # Printing the best results.
        # print("Best Parameters:", self.best_model.best_params_)
        # print("Best Score:", self.best_model.best_score_)

        return rand_search







        # # Train the model
        # self.model.fit(
        #     self.X_train_scaled, self.y_train,
        #     epochs=50,
        #     verbose=0
        # )





    def make_prediction(self, x):
        self.x = x

        # Scaling features to fit the model.
        self.x_scaled = self.scaler.transform(self.x)

        # Making prediciton.
        prediction = self.model.predict(self.x_scaled)

        # Getting the highest score.
        pred_classes = tf.argmax(prediction, axis=1)


        return pred_classes
    

    def get_model_stats(self, X_cv, X_test):
        # Funciton to analyse the trained model stats on the cross validation and test samples.

        self.X_cv = X_cv
        self.X_test = X_test

        # Scale test dataset features.
        self.X_cv_caled = self.scaler.transform(self.X_cv)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Record the training MSEs
        self.yhat_train = self.model.predict(self.X_train_scaled)
        self.train_mse = mean_squared_error(self.y_train, self.yhat_train) / 2

        # Record the cross validation MSEs 
        yhat_cv = self.model.predict(self.X_cv_scaled)
        self.cv_mse = mean_squared_error(self.y_cv, yhat_cv) / 2

        return (self.train_mse, self.cv_mse)


