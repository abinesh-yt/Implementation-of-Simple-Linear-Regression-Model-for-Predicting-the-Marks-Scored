# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Abinesh A
RegisterNumber: 25017255

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Sample data: Hours studied vs Scores
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
Y = np.array([10, 20, 25, 35, 45, 50, 60, 65, 75, 80])

print("Hours:", *X.flatten())
print("Scores:", *Y)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Make predictions
Y_pred = regressor.predict(X_test)
print("Predicted Scores:", *Y_pred)
print("Actual Scores:", *Y_test)

# Plot Training set
plt.scatter(X_train, Y_train, color="orange")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.show()

# Plot Testing set
plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, Y_pred, color="green")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.show()

# Evaluate the model
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)




```

## Output:
<img width="696" height="607" alt="image" src="https://github.com/user-attachments/assets/ecf77519-133a-407a-b763-db3ec23ddc6d" />
<img width="631" height="592" alt="image" src="https://github.com/user-attachments/assets/adfc0d64-10ff-4657-882f-b9e20c7dee3d" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
