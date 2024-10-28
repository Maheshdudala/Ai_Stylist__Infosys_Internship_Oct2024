import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('dataset.csv')

numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

scaler = StandardScaler()

X = df.drop('target_column', axis=1, errors='ignore')
y = df['target_column'] if 'target_column' in df.columns else None

X_processed = preprocessor.fit_transform(X)

if y is not None:
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

X_processed_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())
if y is not None:
    X_processed_df['target'] = y

X_train, X_test, y_train, y_test = train_test_split(X_processed_df.drop('target', axis=1, errors='ignore'), 
                                                    X_processed_df['target'] if 'target' in X_processed_df.columns else None, 
                                                    test_size=0.2, random_state=42)

X_processed_df.to_csv('processed_dataset.csv', index=False)
