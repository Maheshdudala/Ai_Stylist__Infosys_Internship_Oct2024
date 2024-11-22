import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Dataset.csv')
X = df.drop(columns=['link'])
y = df['link'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} rows")
print(f"Test set size: {X_test.shape[0]} rows")