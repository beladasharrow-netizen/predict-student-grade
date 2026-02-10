# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

# Load data
data = pd.read_csv("student_data.csv")

# Features (inputs)
X = data[['study_hours', 'attendance', 'sleep_hours', 'previous_grade']]

# Target (output)
y = data['final_grade']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create AI model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Show predictions vs real results
print("Predicted Grades:", predictions)
print("Real Grades:", y_test.values)

# Evaluate model
error = mean_absolute_error(y_test, predictions)
print("Average Error:", error)
