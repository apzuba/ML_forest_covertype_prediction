# base operations
import pandas as pd
import numpy as np

# for building models and preparing data
from sklearn.model_selection import train_test_split

# REST API
from flask import Flask, jsonify, request

# custom-made local modules
from modules.loading_data import load_dataset
from modules.project_utils import print_compare_predicitons, distil_best_params, adjust_data_for_tf
import modules.heurestics as heur
import modules.scikit_models as scikit_models
import modules.new_tf_module as tf_module
import modules.plot_utils as plt_utils


# Loading the dataset

# Please use function with the dataset_url if you haven't downloaded the dataset in the folder.
# dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
raw_data = load_dataset()   


# Assigning the X attributes and y target parameters.
X = raw_data[:,:raw_data.shape[1]-1]
y = raw_data[:,-1]

# Converting 1-D array into 2-D for the use of later commands.
y = np.expand_dims(y, axis=1)

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
X_train, x_, y_train, y_ = train_test_split(X, y, test_size=0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
X_cv, X_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

del x_, y_


start_row, end_row = 0, 10    # choose the 'start' and 'end' row from the y_train dataset 
                             # to make printed predictions on.


# I. 1. First model : Simple heurestics

# A simple definition of the data fragment we will make a simple prediction on.
spearman_model = heur.Spearman_heurestics(X_train, y_train)

new_spearm_prediction = spearman_model.make_prediction(X[start_row: end_row])

# Print the spearman heurestics predictions
print_compare_predicitons(new_spearm_prediction, y_train, start_row)


# II. Scikit library two ML models: 
# II. 1. Logistic Regression Model applied in Scikit learn.

ovr_model = scikit_models.OvR_Model(X_train, y_train)

new_ovr_prediciton = ovr_model.make_prediction(X[start_row: end_row])

# Print the One vs Rest model predictions
print_compare_predicitons(new_ovr_prediciton, y_train, start_row)

# II. 2. Support Vector Machines (SVM) Model in Scikit learn.

svm_model = scikit_models.SVC_Model(X_train, y_train)

new_svm_prediction = svm_model.make_prediction(X[start_row: end_row])

# Print the SVM model predictions
print_compare_predicitons(new_svm_prediction, y_train, start_row)


# III. Tensorflow model.

# Caution, the training of below models may take a few minutes to hours. The below operation
# scales and reduces the size of dataset to speed up the process.
X_train_scaled, y_train, X_cv_scaled, y_cv, X_test_scaled, y_test, X_mini_test_scaled = adjust_data_for_tf(
    X_train, y_train, 
    X_cv, y_cv, 
    X_test, y_test, 
    X[start_row: end_row]
    )

# Finding the top parameters for the TF model.
best_parameters = tf_module.search_fit(X_train_scaled, y_train, X_cv_scaled, y_cv)

# Create model with new parameters - adjust the parameters based on the found results.
dropout, layers, neurons, learning_rate = distil_best_params(best_parameters)

new_tf_model = tf_module.create_model(
    num_layers=layers, 
    num_neurons=neurons, 
    dropout=dropout, 
    learning_rate=learning_rate
    )

# Plot the training curve
tf_module.plot_the_best(new_tf_model, X_train_scaled, y_train, X_cv_scaled, y_cv)


# IV. Evaluation of all models.

#Decreasing the test size.
X_test1, X_test, y_test1, y_test = train_test_split(X_test, y_test, test_size=0.95, random_state=1)

# 1. Spearman heurestics
spearm_test_results = spearman_model.make_prediction(X_test1)
spearm_eval = np.mean(spearm_test_results==y_test1)

# 2. Scikit OvR Model
ovr_eval = ovr_model.evaluate(X_test1, y_test1)

# 3. Scikit SVM Model
svm_eval = svm_model.evaluate(X_test1, y_test1)

# 3. Tensorflow Model
tf_eval = new_tf_model.evaluate(X_test1, y_test1)[1]

# Plot the four results
plt_utils.plot_accuracies(spearm_eval, ovr_eval, svm_eval, tf_eval)


# V. REST API
app = Flask(__name__)

# Define the API routes
@app.route('/predict', methods=['POST'])
def predict():
    model = request.json['model']
    input_features = request.json['input_features']
    
    if model == 'spearman':
        prediction = spearman_model.make_prediction(input_features)
    elif model == 'ovr':
        prediction = ovr_model.make_prediction(input_features)
    elif model == 'svm':
        prediction = svm_model.make_prediction(input_features)
    elif model == 'tf':
        predictions = new_tf_model.predict(input_features)
        prediction = np.argmax(predictions, axis=1) + 1  # Add 1 to offset the zero-based indexing   
    else:
        return jsonify({'error': 'Invalid model specified'})
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)


    
