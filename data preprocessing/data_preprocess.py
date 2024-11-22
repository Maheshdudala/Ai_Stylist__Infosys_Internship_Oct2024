import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Identify numerical and categorical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# Debugging: Check the initial shape of the dataset
print("Initial Shape of Dataset:", df.shape)

# Debugging: Check the number of unique values in each categorical column
high_cardinality_cols = []
for col in categorical_features:
    unique_values = df[col].nunique()
    print(f"Number of unique values in '{col}':", unique_values)
    if unique_values > 100:  # Adjust threshold as needed
        high_cardinality_cols.append(col)

# Drop high-cardinality columns from categorical features
categorical_features = [col for col in categorical_features if col not in high_cardinality_cols]

# Define numerical and categorical transformers
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers in a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Separate features and target
X = df.drop('target_column', axis=1, errors='ignore')  # Replace 'target_column' with the actual target column name
y = df['target_column'] if 'target_column' in df.columns else None

# Apply the preprocessing pipeline to features
X_processed = preprocessor.fit_transform(X)

# Debugging: Check the shape after transformation
print("Shape after Transformation:", X_processed.shape)

# Encode target if it exists and is categorical
if y is not None:
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

# Convert processed features back into a DataFrame
X_processed_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())
if y is not None:
    X_processed_df['target'] = y

# Split the processed data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed_df.drop('target', axis=1, errors='ignore'), 
    X_processed_df['target'] if 'target' in X_processed_df.columns else None, 
    test_size=0.2, random_state=42
)

# Save the processed dataset
X_processed_df.to_csv('processed_dataset.csv', index=False)
print("Processed dataset saved as 'processed_dataset.csv'")
