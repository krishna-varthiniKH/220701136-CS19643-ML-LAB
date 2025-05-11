from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# Train the model (same code from previous answer)
data = {
    'CGPA': [7.5, 8.9, 7.3, 7.5, 8.3, 7.0, 7.7, 7.7, 6.5, 7.8, 8.3, 7.9, 8.3, 7.7, 7.5, 7.4, 8.6, 8.2, 6.8, 7.4, 7.6, 7.4, 7.7, 7.4, 7.4, 6.7],
    'Internships': [1, 0, 1, 1, 1, 0, 1, 2, 1, 1, 2, 1, 2, 1, 1, 0, 0, 2, 2, 1, 2, 1, 0, 1, 1, 1],
    'Projects': [1, 3, 2, 1, 2, 2, 1, 1, 1, 3, 3, 3, 3, 1, 1, 2, 3, 3, 3, 2, 1, 2, 1, 1, 2, 1],
    'Workshops/Certifications': [1, 2, 2, 2, 2, 2, 1, 0, 0, 2, 2, 2, 2, 1, 1, 1, 0, 2, 0, 0, 0, 1, 0, 0, 1, 1],
    'AptitudeTestScore': [65, 90, 82, 85, 86, 71, 76, 85, 84, 79, 90, 90, 90, 74, 66, 78, 84, 90, 73, 70, 84, 72, 77, 60, 90, 70],
    'SoftSkillsRating': [4.4, 4.0, 4.8, 4.4, 4.5, 4.2, 4.0, 3.5, 3.9, 4.4, 4.8, 4.8, 4.5, 4.6, 4.1, 4.4, 4.8, 4.7, 4.4, 4.4, 4.4, 3.5, 3.4, 3.8, 4.3, 4.0],
    'ExtracurricularActivities': [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    'PlacementTraining': [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    'SSC_Marks': [61, 78, 79, 81, 74, 55, 62, 59, 75, 85, 82, 71, 84, 63, 61, 63, 62, 84, 72, 58, 77, 82, 55, 55, 56, 68],
    'HSC_Marks': [79, 82, 80, 80, 88, 66, 65, 72, 71, 86, 88, 87, 83, 73, 66, 66, 78, 86, 82, 63, 75, 69, 72, 71, 69, 57],
    'PlacementStatus': [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Prepare features and target
X = df.drop(columns=['PlacementStatus'])
y = df['PlacementStatus']

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    cgpa = float(request.form['cgpa'])
    internships = int(request.form['internships'])
    projects = int(request.form['projects'])
    certifications = int(request.form['certifications'])
    aptitude_score = int(request.form['aptitude_score'])
    soft_skills = float(request.form['soft_skills'])
    extracurricular = int(request.form['extracurricular'])
    placement_training = int(request.form['placement_training'])
    ssc_marks = int(request.form['ssc_marks'])
    hsc_marks = int(request.form['hsc_marks'])
    
    # Prepare data for prediction
    input_data = np.array([[cgpa, internships, projects, certifications, aptitude_score, soft_skills,
                            extracurricular, placement_training, ssc_marks, hsc_marks]])
    
    # Make prediction
    prediction = model.predict(input_data)
    placement_status = 'Placed' if prediction[0] == 1 else 'Not Placed'
    
    return f"Predicted Placement Status: {placement_status}"

if __name__ == '__main__':
    app.run(debug=True)
