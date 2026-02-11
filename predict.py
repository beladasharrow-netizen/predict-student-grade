import joblib
import numpy as np

# Load trained model
model = joblib.load("student_model.pkl")

# New student data
new_student = [[4, 90, 7, 80]]  
# study_hours, attendance, sleep, previous_grade

predicted_grade = model.predict(new_student)
print("Predicted Final Grade:", predicted_grade[0])
