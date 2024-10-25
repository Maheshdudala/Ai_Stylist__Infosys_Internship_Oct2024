import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('updated_dataset.csv') 
df.sample(10)

df['id'] = df['id'].str.split('.').str[0]
df.head()

df.shape

df.info()
df.isnull().sum()
df.id = df.id.astype('int64')
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
df.describe()

#removing rows which contribute other masterCategory
df = df[~df['masterCategory'].isin(["Personal Care", "Home", "Free Items"])]
df = df[~df['subCategory'].isin(["Perfumes", "Water Bottle"])]

catnames = [ 'gender', 'masterCategory', 'subCategory', 'articleType',
       'baseColour', 'season','usage','Month']

for i in catnames:
    print(f'{i}: {df[i].unique()}')

df.shape

#numerical data
numnames = ["id","year","ratings","Price (USD)"]
dict_unique = {
    "numerical_columns": numnames,
    "unique_values": [df[i].nunique() for i in numnames]
}

df_unique = pd.DataFrame(dict_unique)
df_unique

for i in numnames:
    plt.figure(figsize=(10,5))
    sns.boxplot(df[i])

#IQR for removing outliers 
q25,q75 = np.percentile(df["Price (USD)"],[25,75])

iqr = q75 - q25
iqr

q0 = q25 - 1.5*iqr
q100 = q75 + 1.5*iqr
q0,q100

df = df[(df['Price (USD)'] >= q0) & (df['Price (USD)'] <= q100)]
df

#Boxplots for numerical values
for i in numnames:
    plt.figure(figsize=(8,4))
    sns.boxplot(df[i])
    plt.title(f'Boxplot of {i}')
    plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['Price (USD)'], bins=30, kde=True)
plt.title('Distribution of Prices (USD)')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.show()

#histograms to check the distribution of numerical data
from scipy.stats import skew,kurtosis
for i in numnames:
    plt.figure(figsize=(8,4))
    print(f"{i}: ",df[i].skew())
    sns.histplot(df[i],kde=True)

#Oversampling of year column for balancing the data

from sklearn.utils import resample

# Separate the data into a list of dataframes by year
dfs = [df[df['year'] == year] for year in df['year'].unique()]

# Determine the target number of samples for each year (e.g., the average count)
target_samples = int(np.mean([len(sub_df) for sub_df in dfs]))

# Resample each year group to the target size
dfs_resampled = [resample(sub_df, replace=True, n_samples=target_samples, random_state=42) for sub_df in dfs]

# Combine all resampled groups into a single dataframe
df_balanced = pd.concat(dfs_resampled)

df_balanced.year.value_counts()

sns.histplot(df_balanced["year"],kde=True)
df_balanced

# understanding the distribution and frequency of categorical columns(demographics)
print('Frequency Distribution of categorical columns') 
for i in catnames:
    if i == "articleType":
        continue
    plt.figure(figsize=(15, 4))
    ax = sns.countplot(data=df_balanced, x=i, palette='Set1')
    plt.xticks(rotation=90) 
    for j in ax.containers:
        ax.bar_label(j)

#BIVARIATE ANALYSIS
# Scatter plot of Price vs Ratings
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_balanced, x='Price (USD)', y='ratings', hue='masterCategory', alpha=0.7)
plt.title('Scatter Plot of Price vs Ratings')
plt.xlabel('Price (USD)')
plt.ylabel('Ratings')
plt.legend(loc='upper left')
plt.show()

sns.boxplot(df_balanced,x='masterCategory',y='Price (USD)',palette='rocket')
plt.title("Box Plot of Master category by Price(USD)")

plt.figure(figsize=(12,4))
sns.boxplot(df_balanced,x='subCategory',y='ratings',palette='Set2')
plt.title("Box Plot of Ratings by SubCategory")
plt.xticks(rotation=90)
plt.show()

# Count plot of Master Category vs Ratings
plt.figure(figsize=(12, 6))
ax = sns.countplot(data=df_balanced, x='masterCategory', hue='ratings')
for i in ax.containers:
    plt.bar_label(i)
plt.title('Count Plot of Master Category vs Ratings')
plt.xticks(rotation=45)
plt.show()

# Master Category vs Gender
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df_balanced, x='masterCategory', hue='gender')
for i in ax.containers:
    plt.bar_label(i)
plt.title('Master Category Distribution by Gender')
plt.xticks(rotation=45)
plt.show()

df_balanced.subCategory.unique()

plt.figure(figsize=(12,6))
df_grouped = df_balanced.groupby(['subCategory', 'gender']).size().reset_index(name='count')
ax = sns.barplot(data=df_grouped, x='subCategory', y='count', hue='gender', order=df_balanced['subCategory'].value_counts().index[:10])
for i in ax.containers:
    plt.bar_label(i)
plt.xticks(rotation=90)
plt.show()

# Master Category vs Gender
plt.figure(figsize=(12,6))
df_grouped = df_balanced.groupby(['masterCategory', 'gender']).size().reset_index(name='count')
ax = sns.barplot(data=df_grouped, x='masterCategory', y='count', hue='gender')
for i in ax.containers:
    plt.bar_label(i)
plt.xticks(rotation=90)
plt.show()

# subcategory vs season
plt.figure(figsize=(12,6))
df_grouped = df_balanced.groupby(['subCategory', 'season']).size().reset_index(name='count')
ax = sns.barplot(data=df_grouped, x='subCategory', y='count', hue='season', order=df_balanced['subCategory'].value_counts().index[:10])
for i in ax.containers:
    plt.bar_label(i)
plt.xticks(rotation=90)
plt.show()

# base colour vs usage
plt.figure(figsize=(18,6))
df_grouped = df_balanced.groupby(['baseColour', 'usage']).size().reset_index(name='count')
ax = sns.barplot(data=df_grouped, x='baseColour', y='count', hue='usage', order=df_balanced['baseColour'].value_counts().index[:10])
for i in ax.containers:
    plt.bar_label(i)
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(10,5))
sns.heatmap(df_balanced.corr(),annot=True)

#Considering the months column understanding the frequency distribution of purchases
#(trend projection analysis)

plt.figure(figsize=(10,5))
ax = sns.countplot(data=df_balanced, x='Month', order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
                   palette="Set2")
for i in ax.containers:
    plt.bar_label(i)
plt.title('Distribution of Purchases by Month')
plt.xticks(rotation=45)
plt.show()

#Considering the months column understanding the frequency distribution of price

plt.figure(figsize=(10,5))
ax = sns.barplot(data=df_balanced,x="Month",y="Price (USD)",palette="Set1",
                 order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
for i in ax.containers:
    plt.bar_label(i)
plt.xticks(rotation=90)
plt.show()

#ADVANCED EDA #CLUSTERING ANALYSIS

# label encoding categorical columns to numerical values ensuring for clustering
from sklearn.preprocessing import LabelEncoder,StandardScaler
le = LabelEncoder()
for i in catnames:
    df_balanced[i] = le.fit_transform(df_balanced[i])

df_balanced

# scaling the columns for apporitate clustering process
features_to_be_scaled = ['gender', 'masterCategory', 'subCategory', 'articleType', 
    'baseColour', 'season', 'ratings', 'Price (USD)', 'Month', 'year','usage']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_balanced[features_to_be_scaled])

df_scaled

# applying pca for  reducing the dimensions for future visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

# elbow method to decide clusters for k-means clustering analysis
from sklearn.cluster import KMeans
wcss = [] # within cluster sum of squares
for k in range(1,11):
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(pca_result)
    wcss.append(kmeans.inertia_)


plt.plot(range(1,11),wcss,marker='o')
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow method for optimal k")
plt.grid(True)
plt.show()

df_pca = pd.DataFrame(pca_result,columns=['pca1','pca2'])
df_pca
kmeans = KMeans(n_clusters=4,random_state=42)
df_pca["cluster"] = kmeans.fit_predict(df_pca)
df_pca
df_balanced["cluster"] = df_pca['cluster']

# Visualize the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_pca,
    x='pca1', y='pca2', 
    hue='cluster', 
    palette='viridis'
)
plt.title('Clustering of Products with PCA (Including Month and Year)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

###Content based filtering

from sklearn.metrics.pairwise import cosine_similarity

##finding similarity matrixes
# cosine similarity between all products
similarity_matrix = cosine_similarity(df_scaled)

# similarity DataFrame 
similarity_df = pd.DataFrame(similarity_matrix, index=df_balanced.index, columns=df_balanced.index)

print("Indices range:", similarity_df.index.min(), "to", similarity_df.index.max())

# Check if index  exists
if 10 in similarity_df.index:
    print("Index 10 exists.")
else:
    print("Index 10 does not exist.")


df_balanced.reset_index(drop=True, inplace=True)
similarity_df = pd.DataFrame(similarity_matrix, index=df_balanced.index, columns=df_balanced.index)

# Display the first few rows along with indices
print(df.head())

# Resetting the index to make it sequential
df.reset_index(drop=True, inplace=True)

# Check the new indices
print(df.index)

#expected range of indices based on the size of the DataFrame
expected_indices = set(range(df.index.min(), df.index.max() + 1))

#actual indices in the DataFrame
actual_indices = set(df.index)

# finding the missing indices
missing_indices = expected_indices - actual_indices

print(f"Missing indices: {missing_indices}")

#
valid_index = similarity_df.index[10]  # Choosing the first valid index
recommendations = get_recommendations(product_index=valid_index, similarity_df=similarity_df, num_recommendations=6) # type: ignore
print(f"Recommended products similar to product {valid_index}: {recommendations}")

def get_recommendations(product_index, similarity_df, num_recommendations=6):
    
    # Get the similarity scores for the product
    sim_scores = similarity_df[product_index]

    # Sort the products based on similarity scores (highest first), and exclude the input product itself
    sorted_similar_indices = sim_scores.sort_values(ascending=False).index.tolist()

    # Return top N similar products, excluding the first (which would be the product itself)
    return sorted_similar_indices[1:num_recommendations + 1]

# Example: Recommend 5 products similar to product at index 10
recommendations = get_recommendations(product_index=18, similarity_df=similarity_df, num_recommendations=6)
print(f"Recommended products similar to product 18: {recommendations}")

# List of specific index values
index_list = [18, 3064, 933, 6704, 2655, 2149]

# Retrieve the rows corresponding to the indices in index_list
data_for_indices = df.loc[index_list]

# Display the result
print(data_for_indices)


 