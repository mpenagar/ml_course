from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

__all__ = ['BaseModel', 'SimpleRegressor', 'ComplexRegressor', 'ACMEClassifier']

class BaseModel(BaseEstimator, RegressorMixin):
    """
    Base wrapper class for all educational Machine Learning regression models.
    
    This class handles the standard interface (fit, predict, score) 
    by delegating the operations to the underlying scikit-learn model.
    """
    
    def predict(self, X):
        """
        Generates predictions for new data using the underlying model.
        """
        return self._model.predict(X)


class SimpleRegressor(BaseModel):
    """
    A really simple regression model without hyper-parameters.
    
    Parameters
    ----------
    """
    
    def fit(self, X, y):
        """
        Fits the underlying model using the provided training data.
        """        
        self._model = LinearRegression()
        self._model.fit(X, y)
        return self

        
class ComplexRegressor(BaseModel):
    """A somehow complex regression model with some hyper-parameters.
    
    A regression model that learns by continuously partitioning the dataset into smaller
    groups based on the input features. At each step, it randomly divides the data into
    more specific subsets, ultimately making a tailored prediction for each final group.
    While powerful, this model requires careful tuning of its hyper-parameters to ensure
    it learns general trends rather than just memorizing the training data.    

    Parameters
    ----------
    complexity : int, default=None
        Controls the maximum complexity of the model's internal structure. A higher value
        allows the model to learn more intricate patterns from the training data, but it 
        heavily increases the risk of overfitting (memorizing the specific training 
        examples instead of learning the general trend). If None, there is no limit on 
        how complex the model can become (use with caution!).
        
    min_split_size : int, default=2
        The minimum amount of data points required before the model is allowed to divide
        a group into even smaller pieces. Increasing this value stops the model from 
        creating overly complex and specific rules for very small sets of data.

    random_state : int, default=None
        Controls the randomness of the estimator. To obtain a deterministic behaviour 
        during fitting, random_state has to be fixed to an integer
    """
    
    def __init__(self, complexity=None, min_split_size=2, random_state=None):
        """
        Initializes the regressor.
        """
        self.complexity = complexity
        self.min_split_size = min_split_size
        self.random_state = random_state
        

    def fit(self, X, y):
        """
        Fits the model using the provided training data.
        """
        self._model = DecisionTreeRegressor(
            max_depth = self.complexity,
            min_samples_split = self.min_split_size,
            random_state = self.random_state
        )
        
        self._model.fit(X, y)
        return self


class ACMEClassifier(BaseEstimator, ClassifierMixin):
    """A simple classification model to distinguish between different categories.
    
    This model tries to draw a mathematical boundary between different classes
    in the dataset. It calculates the probability of a sample belonging to a 
    certain category and makes a decision based on those probabilities.
    
    Parameters
    ----------
    regularization_strength : float, default=1.0
        Controls how strictly the model is penalized for being overly complex. 
        Smaller values create simpler, more generalized boundaries (higher penalty), 
        while larger values allow the model to fit the training data more closely 
        (lower penalty), which might lead to overfitting.
        
    max_iterations : int, default=100
        The maximum number of attempts the model is allowed to make in order 
        to find the optimal boundary. If the model hasn't converged (found 
        the best solution) within this limit, it will stop learning.
    """
    
    def __init__(self, regularization_strength=1.0, max_iterations=100):
        """
        Initializes the classifier.
        """
        self.regularization_strength = regularization_strength
        self.max_iterations = max_iterations

    def fit(self, X, y):
        """
        Fits the underlying classification model using the provided training data.
        """
        self._model = LogisticRegression(
            C = self.regularization_strength,
            max_iter = self.max_iterations
        )
        
        self._model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts the class labels for the provided data.
        """
        return self._model.predict(X)
        
    def predict_proba(self, X):
        """
        Predicts the probability of each class for the provided data.
        Useful for students to understand how confident the model is.
        """
        return self._model.predict_proba(X)