from sklearn.tree import DecisionTreeRegressor

__all__ = ['BaseModel', 'Regressor1']

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
    Educational regression model 1.
    
    Parameters:
    - param1 (int, default=None): Controls the maximum depth of the tree.
    - param2 (int, default=2): Controls the minimum number of samples required to split an internal node.
    """
    
    def __init__(self, param1=None, param2=2):
        self.param1 = param1
        self.param2 = param2
        
        # Instantiate the specific scikit-learn model
        sklearn_model = DecisionTreeRegressor(
            max_depth=self.param1,
            min_samples_split=self.param2
        )
        
        # Pass the instantiated model to the parent class
        super().__init__(model=sklearn_model)