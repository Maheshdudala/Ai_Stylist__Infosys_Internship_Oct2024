import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("dataset.csv")

# Initial exploration
print(df.head())
print(df.info())
print(df.describe(include='all'))

# Handle missing values
df = df.dropna()  # or use appropriate imputation methods

# Missing values analysis
missing_values = df.isnull().sum()
print(missing_values)
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Feature selection
selected_features = df[['gender', 'season', 'baseColour', 'usage', 'ratings']]

# Data visualization example
sns.countplot(data=selected_features, x='baseColour')
plt.title('Color Distribution')
plt.show()

# Distribution plots for numerical features
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
for feature in numerical_features:
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Box plots for numerical features
for feature in numerical_features:
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.show()

# Correlation analysis
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pair plots
sns.pairplot(df[numerical_features])
plt.title('Pair Plot')
plt.show()

# Categorical features analysis
categorical_features = df.select_dtypes(include=['object']).columns
for feature in categorical_features:
    sns.countplot(y=df[feature])
    plt.title(f'Distribution of {feature}')
    plt.show()

# Feature relationships
sns.scatterplot(data=df, x='ratings', y='baseColour', hue='gender')
plt.title('Ratings vs Base Colour by Gender')
plt.show()

# Clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(selected_features.select_dtypes(include='number'))
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(scaled_features)

# Visualize clusters (if applicable)
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=clusters)
plt.title('Item Clusters')
plt.show()