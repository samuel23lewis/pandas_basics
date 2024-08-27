import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Create a linear regression model
model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(model, test_size=0.2, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
print(model.score(X_test, y_test))

from sklearn.linear_model import LinearRegression
# Create a linear regression model
model = LinearRegression()
# Fit the model to the training data
model.fit(X_train, y_train)
# Evaluate the model on the test data
score = model.score(X_test, y_test)
# Print the score
print(score)