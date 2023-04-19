
import numpy as np
from scipy import stats

class Spearman_heurestics:
    """The simple heurestics model using Spearman statistics to compare 
    the correlation between two vectors.

        Parameters
        ----------
        X_array : 2-Dimensional numpy array with X.shape[1] number of attributes.
        X : Training 2-D array with attribute values
        y : Training 2-D array with target values
        
        Returns
        -------
        y_p : Integer with the target type prediction.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

        # Adjusting the y shape for 1-D operations.
        self.y = np.reshape(self.y, self.y.shape[0])

        # Finding attributes averages for each cover type.
        self.unique_targets = np.unique(self.y)
        self.avg_by_type = np.zeros((len(self.unique_targets), self.X.shape[1]))

        for num, unique_target in enumerate(self.unique_targets):
            self.avg_by_type[num] = np.mean(self.X[self.y == unique_target], axis = 0)


    def make_prediction(self, x_array):
        self.x_array = x_array

        results = np.zeros(x_array.shape[0])

        # Applying the Spearman Coefficient correlation
        for j, xi_array in enumerate(self.x_array):
            spearman_coefcs = np.zeros((len(self.unique_targets)))

            for i, avg_val in enumerate(self.avg_by_type):
                res = stats.spearmanr(xi_array, avg_val)
                spearman_coefcs[i] = res.statistic

            # Choosing the highest score as the prediction.    
            y_p = np.where(spearman_coefcs == np.max(spearman_coefcs))[0][0]+1
            results[j] = y_p
        return results
    
    
    