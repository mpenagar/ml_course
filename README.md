# Intro ML Course

Python package containing educational regression and classification models for the Introduction to Machine Learning course.

## Installation

Run the following command in your terminal or Jupyter Notebook cell:

```bas
pip install git+https://github.com/your-username/intro_ml_course.git
```

## Usage Example

from ml_course import Regressor1

# Initialize the model
model = Regressor1()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(accuracy)