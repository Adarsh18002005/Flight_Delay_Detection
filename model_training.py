import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Load the dataset
df = pd.read_csv('flightdata.csv')

# Drop the 'Unnamed: 25' column
df = df.drop(columns=['Unnamed: 25'])

# Drop rows where 'ARR_DEL15' is NaN, as it is our target variable
df.dropna(subset=['ARR_DEL15'], inplace=True)

# Fill remaining missing numerical values with the mean of their respective columns
numerical_cols_with_nan = ['DEP_TIME', 'DEP_DELAY', 'DEP_DEL15', 'ARR_TIME', 'ARR_DELAY', 'ACTUAL_ELAPSED_TIME']
for col in numerical_cols_with_nan:
    if col in df.columns:
        df[col].fillna(df[col].mean(), inplace=True)

# Select features and target variable
features = [
    'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER',
    'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'DISTANCE',
    'CRS_ELAPSED_TIME', 'DEP_DELAY', 'CANCELLED', 'DIVERTED'
]
target = 'ARR_DEL15'

X = df[features]
y = df[target]

# Store the columns before one-hot encoding for later use in Flask app
# This is crucial for ensuring the Flask app processes inputs in the same way as the model was trained
original_columns = X.columns.tolist()

# Convert categorical features to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['UNIQUE_CARRIER', 'ORIGIN', 'DEST'], drop_first=True)

# Get the list of columns after one-hot encoding
# This list will be used to ensure consistent feature order during prediction
encoded_columns = X.columns.tolist()

# Save the list of encoded columns to a file for the Flask app
with open('encoded_columns.joblib', 'wb') as f:
    joblib.dump(encoded_columns, f)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'flight_delay_model.joblib')

print("Model training complete and model saved as 'flight_delay_model.joblib'")
print("Encoded columns saved as 'encoded_columns.joblib'")