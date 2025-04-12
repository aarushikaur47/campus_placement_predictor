# campus_placement_pipeline.py

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("Campus_Selection.csv")

# Streamlit Title
st.title("Campus Placement Prediction")

# Show raw data
if st.checkbox("Show Raw Data"):
    st.write(data.head())

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target
target_column = 'status'
y = data[target_column]
X = data.drop(target_column, axis=1)

# Save feature column names for input alignment
feature_columns = X.columns

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation metrics
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# Display metrics
st.subheader("Model Evaluation")
st.write(f"Accuracy: {acc:.2f}")
st.write(pd.DataFrame(report).transpose())

# Plot confusion matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Predict on custom input
st.subheader("Try Custom Input")
custom_input = {
    'gender': st.selectbox("Gender", ['M', 'F']),
    'ssc_p': st.slider("SSC Percentage (10th)", 0.0, 100.0, 60.0),
    'hsc_p': st.slider("HSC Percentage (12th)", 0.0, 100.0, 60.0),
    'hsc_b': st.selectbox("HSC Board", ['Central', 'Others']),
    'hsc_s': st.selectbox("HSC Stream", ['Commerce', 'Science', 'Arts']),
    'degree_p': st.slider("Degree Percentage", 0.0, 100.0, 60.0),
    'degree_t': st.selectbox("Degree Type", ['Sci&Tech', 'Comm&Mgmt', 'Others']),
    'workex': st.selectbox("Work Experience", ['Yes', 'No']),
    'etest_p': st.slider("E-Test Percentage", 0.0, 100.0, 60.0),
    'specialisation': st.selectbox("MBA Specialisation", ['Mkt&HR', 'Mkt&Fin']),
    'mba_p': st.slider("MBA Percentage", 0.0, 100.0, 60.0)
}

if st.button("Predict Placement"):
    input_df = pd.DataFrame([custom_input])
    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    # Reindex the input_df to match training feature order
    input_df = input_df.reindex(columns=feature_columns)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    result = label_encoders[target_column].inverse_transform([prediction])[0] if target_column in label_encoders else prediction
    st.success(f"Predicted Placement Status: {result}")
