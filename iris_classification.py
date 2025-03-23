import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from CSV file
file_path = "iris.csv"  # Ensure the dataset is in the same folder as the script
df = pd.read_csv(file_path, header=0)  # Skip the first row since it contains headers

# Print first few rows to verify correct loading
print("First few rows of the dataset:")
print(df.head())
print("Dataset shape:", df.shape)

# Convert numerical columns to float
df[["sepal_length", "sepal_width", "petal_length", "petal_width"]] = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].astype(float)

# Encode target labels
df["species"] = pd.Categorical(df["species"])
df["species_code"] = df["species"].cat.codes
species_mapping = dict(enumerate(df["species"].cat.categories))

# Prepare features and target variable
X = df.drop(columns=["species", "species_code"])
y = df["species_code"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Predict species for new data
new_data = [
    [5.1, 3.5, 1.4, 0.2], 
    [6.7, 3.1, 4.4, 1.4],  
    [5.8, 2.7, 5.1, 1.9],  
    [7.2, 3.6, 6.1, 2.5],  
    [4.9, 2.5, 4.5, 1.7]  
]

# Standardize the new data using the same scaler
new_data_scaled = scaler.transform(new_data)

# Make predictions on new data
predictions = model.predict(new_data_scaled)

# Map numeric predictions back to species names
predicted_species = [species_mapping.get(int(pred), "Unknown") for pred in predictions]

print("Predicted species for new data:", predicted_species)
