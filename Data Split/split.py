import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')
X = df.drop(columns=['target_column'])
y = df['target_column']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
