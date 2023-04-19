
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np

# from sklearn.cluster import KMeans 
# import plot_utils


# One vs Rest (OvR) Logistic Regression model
class OvR_Model:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        #scaling the features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)

        # Adjusting the y shape for 1-D and 2-D operations.
        # y_train = np.reshape(y_train, y_train.shape[0])

        # Define and train the OvR classifier
        self.ovr_clf = Pipeline([
        ('ovr', OneVsRestClassifier(LogisticRegression(max_iter=2000)))
        ])
        self.ovr_clf.fit(self.X_train, self.y_train)

    def make_prediction(self, x):
        self.x = x

        # Adjust the x array to scaled features.
        x_scaled = self.scaler.transform(self.x)

        # Make a prediction on x.
        y_pred = self.ovr_clf.predict(x_scaled)

        return y_pred
    
    def evaluate(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

        X_test_scaled = self.scaler.transform(self.X_test)

        y_pred = self.ovr_clf.predict(X_test_scaled)

        # calculate the accuracy of the predictions
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy


class SVC_Model:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = np.ravel(y_train)

        # Decreasing the X_train size to speed up the time of model training.
        self.X_train, x_, self.y_train, y_ = train_test_split(self.X_train, self.y_train, test_size=0.92, random_state=1)
        del x_, y_

        # Scaling the training data
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)

        # Creating an instance and training the model.
        self.svm = SVC(kernel='linear', probability=False)

        self.svm.fit(self.X_train, self.y_train)

    def make_prediction(self, x):
        self.x = x
        
        # Scaling the data to predict.
        self.x = self.scaler.transform(self.x)

        y_pred = self.svm.predict(self.x)

        return y_pred

    def evaluate(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

        X_test_scaled = self.scaler.transform(self.X_test)

        y_pred = self.svm.predict(X_test_scaled)

        # calculate the accuracy of the predictions
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy


# def K_means(X_train):
#     # k-nearest neighbours

#     km = KMeans(
#         n_clusters=7, init='random',
#         n_init=10, max_iter=300, 
#         tol=1e-04, random_state=0
#     )
#     y_km = km.fit_predict(X_train)

#     #choose the number of datapoints to plot
#     plot_data_no = 1000
#     plot_utils.plot_k_means(km, X_train[:plot_data_no], y_km[:plot_data_no])