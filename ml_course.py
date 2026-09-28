from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

__all__ = ['BaseModel', 'Regressor1', 'Regressor2']

class BaseModel:
    """
    Base wrapper class for all educational Machine Learning models.
    
    This class handles the standard interface (fit, predict, score) 
    by delegating the operations to the underlying scikit-learn model.
    """
    
    def __init__(self, model):
        """
        Initializes the base model wrapper.
        
        Parameters:
        - model: An instantiated scikit-learn model.
        """
        self._model = model

    def fit(self, X, y):
        """
        Fits the underlying model using the provided training data.
        """
        self._model.fit(X, y)
        return self

    def predict(self, X):
        """
        Generates predictions for new data using the underlying model.
        """
        return self._model.predict(X)

    def score(self, X, y):
        """
        Returns the performance metric (e.g., R-squared or accuracy) 
        from the underlying model.
        """
        return self._model.score(X, y)


class Regressor1(BaseModel):
    """
    A really simple regression model without hyper-parameters.
    
    """
    
    def __init__(self):        
        # Instantiate the specific scikit-learn model
        sklearn_model = LinearRegression()
        
        # Pass the instantiated model to the parent class
        super().__init__(model=sklearn_model)
        
class Regressor2(BaseModel):
    """
    A complex regression model with many hyper-parameters.
    
    Parameters:
    - complexity (int, default=None): Controls the maximum complexity of the model's decision rules. A higher value allows the model to learn more intricate patterns from the training data, but it heavily increases the risk of overfitting (memorizing the specific training examples instead of learning the general trend). If None, there is no limit on how complex the model can become (use with caution!).
    - min_split_size (int, default=2): The minimum amount of data points required before the model is allowed to divide a group into even smaller pieces. Increasing this value stops the model from creating overly complex and specific rules for very small sets of data.
    - random_state (int, default=None): Controls the randomness of the estimator. To obtain a deterministic behaviour during fitting, random_state has to be fixed to an integer
    """
    
    def __init__(self, complexity=None, min_split_size=2, random_state=None):
        # Instantiate the specific scikit-learn model
        sklearn_model = DecisionTreeRegressor(
            max_depth=complexity,
            min_samples_split=min_split_size,
            random_state=random_state
        )
        
        # Pass the instantiated model to the parent class
        super().__init__(model=sklearn_model)
