A project working with the large, publicly available US forest data.
The project included and evaluated four trained prediction models: 
  - Spearman coefficient-based heurestic, 
  - sklearn OvR, 
  - sklearn SVM, 
  - multiclass TensorFlow.
Program structured modularely with parts such as: 
A) Data preprocessing, B) Hyperparameters self-learning function, 
C) plots of the training for the top parameters, D) models evalation, E) REST API.


The project tasks/objectives included:
1. Load the Covertype Data Set
○ https://archive.ics.uci.edu/ml/datasets/Covertype

2. Implement a very simple heuristic that will classify the data
○ It doesn't need to be accurate

3. Use Scikit-learn library to train two simple Machine Learning models
○ Choose models that will be useful as a baseline

4. Use TensorFlow library to train a neural network that will classify the data
○ Create a function that will find a good set of hyperparameters for the NN
○ Plot training curves for the best hyperparameters

5. Evaluate your neural network and other models
○ Choose appropriate plots and/or metrics to compare them

6. Create a very simple REST API that will serve your models
○ Allow users to choose a model
(heuristic, two other baseline models, or neural network)
○ Take all necessary input features and return a prediction
